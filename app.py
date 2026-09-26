```python
import os
import re
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            padding: 1.2rem;
            border-radius: 10px;
            background-color: #f7f9fc;
            border: 1px solid #e1e5eb;
        }

        .source-box {
            padding: 0.8rem;
            border-radius: 8px;
            background-color: #fafafa;
            border-left: 4px solid #888;
            margin-bottom: 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📘 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an HR policy PDF and ask questions using RAG-powered search.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    return Groq(api_key=api_key)


try:
    embedding_model = load_embedding_model()
except Exception as e:
    st.error(f"Could not load embedding model: {e}")
    st.stop()


# ---------------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(uploaded_file):
    """
    Extract text from every page of the uploaded PDF.
    Returns a list of page dictionaries.
    """

    pdf_bytes = uploaded_file.getvalue()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        if text and text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    document.close()

    return pages


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    """
    Normalize unnecessary whitespace while preserving text.
    """

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

def create_chunks(pages, chunk_size=900, overlap=150):
    """
    Split policy text into overlapping chunks.

    Each chunk keeps its original PDF page number.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = clean_text(page_data["text"])

        if not text:
            continue

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ---------------------------------------------------------
# CREATE FAISS INDEX
# ---------------------------------------------------------

def build_faiss_index(chunks):
    """
    Create embeddings and build a FAISS cosine-similarity index.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype("float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------

def retrieve_chunks(query, index, chunks, top_k=4):
    """
    Retrieve the most relevant policy chunks.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):

        if index_number < 0:
            continue

        results.append(
            {
                "text": chunks[index_number]["text"],
                "page": chunks[index_number]["page"],
                "score": float(score)
            }
        )

    return results


# ---------------------------------------------------------
# GROQ ANSWER GENERATION
# ---------------------------------------------------------

def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using only the retrieved HR policy context.
    """

    client = load_groq_client()

    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"[Source {i} | PDF Page {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Answer the user's question using ONLY the HR policy context supplied
in the prompt.

Rules:

1. Do not invent HR policies.
2. Do not use outside knowledge as if it came from the uploaded policy.
3. If the policy does not contain enough information to answer the question,
   clearly say that the uploaded policy does not provide enough information.
4. Give a concise and practical answer.
5. When possible, mention the relevant policy section/page.
6. If the policy contains conditions, exceptions, limits, or approval
   requirements, include them.
7. Do not make legal claims that are not explicitly contained in the policy.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Provide the answer based strictly on the HR Policy Context.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Policy Document")

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"]
    )

    st.divider()

    top_k = st.slider(
        "Retrieved sections",
        min_value=2,
        max_value=8,
        value=4
    )

    st.caption(
        "The assistant searches the uploaded PDF first, "
        "then sends the relevant sections to Groq."
    )


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file is not None:

    if st.session_state.document_name != uploaded_file.name:

        with st.spinner("Processing HR policy PDF..."):

            try:
                pages = extract_pdf_text(uploaded_file)

                if not pages:
                    st.error(
                        "No readable text was found in this PDF. "
                        "If it is a scanned PDF, OCR may be required."
                    )
                    st.stop()

                chunks = create_chunks(pages)

                if not chunks:
                    st.error("Could not create searchable text chunks.")
                    st.stop()

                index = build_faiss_index(chunks)

                st.session_state.chunks = chunks
                st.session_state.faiss_index = index
                st.session_state.document_name = uploaded_file.name
                st.session_state.messages = []

                st.success(
                    f"Policy processed successfully: "
                    f"{len(pages)} pages, {len(chunks)} searchable sections."
                )

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.stop()


# ---------------------------------------------------------
# DOCUMENT STATUS
# ---------------------------------------------------------

if st.session_state.faiss_index is not None:

    st.info(
        f"📄 Active policy: **{st.session_state.document_name}**"
    )

else:

    st.info(
        "👈 Upload an HR Policy PDF from the sidebar to begin."
    )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("📚 Retrieved policy sections"):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-box">
                        <b>PDF Page {source['page']}</b>
                        &nbsp; | &nbsp;
                        Similarity: {source['score']:.3f}
                        <br><br>
                        {source['text']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ---------------------------------------------------------
# USER QUESTION
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about the HR policy..."
)


if question:

    if st.session_state.faiss_index is None:

        st.warning(
            "Please upload an HR Policy PDF first."
        )
        st.stop()

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant chunks
    with st.spinner("Searching the HR policy..."):

        retrieved = retrieve_chunks(
            question,
            st.session_state.faiss_index,
            st.session_state.chunks,
            top_k=top_k
        )

    # Generate answer
    with st.chat_message("assistant"):

        try:

            with st.spinner("Generating answer..."):

                answer = generate_answer(
                    question,
                    retrieved
                )

            st.markdown(answer)

            st.markdown("### 📚 Sources")

            for source in retrieved:

                st.markdown(
                    f"""
                    <div class="source-box">
                    <b>PDF Page {source['page']}</b>
                    &nbsp; | &nbsp;
                    Similarity: {source['score']:.3f}
                    <br><br>
                    {source['text']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved
                }
            )

        except Exception as e:

            error_message = f"Could not generate the answer: {e}"

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                }
            )
```

### 2. `requirements.txt`

```text
streamlit==1.50.0
groq>=0.31.0
PyMuPDF>=1.26.4
sentence-transformers>=5.1.0
faiss-cpu==1.12.0
numpy<2
```

### 3. `README.md`

````markdown
# 📘 HR Policy Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to
upload an HR Policy PDF and ask questions about the document.

The application searches the uploaded policy using semantic vector search
and then uses Groq's `openai/gpt-oss-20b` model to generate an answer based
on the retrieved policy sections.

## 🚀 Features

- Upload an HR Policy PDF
- Extract PDF text using PyMuPDF
- Split the document into overlapping chunks
- Generate semantic embeddings using Sentence Transformers
- Store embeddings in a FAISS vector index
- Retrieve the most relevant policy sections
- Generate answers using Groq
- Display the PDF pages used as sources
- Chat-style question and answer interface
- No API key stored in the source code

## 🧠 RAG Workflow

The application follows this workflow:

User uploads PDF
        ↓
PyMuPDF extracts text
        ↓
Text is split into chunks
        ↓
Sentence Transformers creates embeddings
        ↓
FAISS stores the embeddings
        ↓
User asks a question
        ↓
Question is converted into an embedding
        ↓
FAISS retrieves relevant policy sections
        ↓
Retrieved sections are sent to Groq
        ↓
Groq generates the answer
        ↓
Answer + source pages are displayed

## 🛠 Technology Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Groq
- `openai/gpt-oss-20b`

## 📁 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
````

Do NOT upload `secrets.toml` to GitHub.

## 🔐 Groq API Key

The application expects this Streamlit secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do not put your real API key inside `app.py`.

Do not commit your API key to GitHub.

## 🌐 Deploy on Streamlit Community Cloud

### Step 1 — Create the GitHub repository

Go to GitHub and create a new repository.

Example repository name:

```text
hr-policy-assistant
```

Keep the repository public if you are using the free Streamlit Community
Cloud workflow.

### Step 2 — Add the files

In the GitHub repository, click:

**Add file → Create new file**

Create:

```text
app.py
```

Paste the complete `app.py` code.

Then create:

```text
requirements.txt
```

Paste the requirements.

Then create:

```text
README.md
```

Paste this README.

Then create:

```text
.gitignore
```

Paste the `.gitignore` content supplied with this project.

### Step 3 — Deploy

Open Streamlit Community Cloud.

Sign in using GitHub.

Choose:

**Create app**

Select:

* Repository: your `hr-policy-assistant` repository
* Branch: `main`
* Main file path: `app.py`

Click:

**Deploy**

Streamlit will install the packages from `requirements.txt`.

### Step 4 — Add the Groq API key

After the application is created, open the Streamlit app settings.

Go to:

**Settings → Secrets**

Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Save the secret.

Then restart/reboot the application if Streamlit asks you to.

### Step 5 — Test the application

Open your deployed Streamlit application.

Upload an HR Policy PDF.

For example:

```text
Employee_Handbook.pdf
```

Wait for:

```text
Policy processed successfully
```

Then ask questions such as:

```text
How many annual leaves are employees entitled to?
```

```text
What is the maternity leave policy?
```

```text
What is the procedure for sick leave?
```

```text
Who approves annual leave?
```

```text
What is the company's working-hours policy?
```

The application will retrieve relevant sections before generating the answer.

## ⚠️ Important RAG Limitation

The assistant is designed to answer from the uploaded policy.

If the required information is not present in the retrieved policy context,
the assistant should say that the uploaded policy does not provide enough
information.

It should not invent a company policy.

## 📄 PDF Limitation

The current version works best with PDFs containing selectable text.

Scanned/image-only PDFs may not contain extractable text.

OCR can be added later if required.

## 🔒 Security

Never place the Groq API key directly inside `app.py`.

Never commit:

```text
.streamlit/secrets.toml
```

to GitHub.

Use Streamlit Cloud Secrets instead.

## 💻 Local Development

This project is designed to be deployed directly from GitHub using
Streamlit Community Cloud.

No Colab, VS Code, or terminal is required for the deployment workflow.

````

### 4. `.gitignore`

:::writing{variant="document" id="92547" title=".gitignore"}
```text
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
.venv/
venv/
env/
ENV/

# Streamlit secrets
.streamlit/secrets.toml

# Environment variables
.env
.env.*

# IDE files
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Python packaging
*.egg-info/
dist/
build/

# Logs
*.log

# Temporary files
*.tmp
*.temp
````

### 5. GitHub → Streamlit Cloud deployment

You **do not need to create `.streamlit/secrets.toml` in GitHub**. In fact, don't. The `.gitignore` above intentionally prevents it from being committed.

Your GitHub repository should look like:

```text
hr-policy-assistant
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Then:

**GitHub → create repository → add the 4 files → Streamlit Community Cloud → Create app → select repository → `app.py` → Deploy.**

After deployment, put your Groq key into **Streamlit Cloud → App settings → Secrets**:

```toml
GROQ_API_KEY = "your_actual_groq_key"
```

That is the important distinction from the TOML problem you encountered previously: **the key should be plain TOML with normal straight quotes**, not smart/curly quotes such as `“ ”`.

The resulting architecture is:

```text
                HR Policy PDF
                      │
                      ▼
                 PyMuPDF
                      │
                      ▼
                Text Chunks
                      │
                      ▼
          Sentence Transformers
                      │
                      ▼
                   FAISS
                      │
          ┌───────────┴───────────┐
          │                       │
     User Question          Relevant Chunks
          │                       │
          └───────────┬───────────┘
                      ▼
                  Groq API
          openai/gpt-oss-20b
                      │
                      ▼
              Answer + Sources
                      │
                      ▼
                 Streamlit UI
```

This version is intentionally **single-file and simple**, so you can manage the entire project through GitHub's web interface without needing Colab, VS Code, or a terminal.

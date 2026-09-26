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

Do NOT upload secrets.toml to GitHub.

🔐 Groq API Key

The application expects this Streamlit secret:

GROQ_API_KEY = "your_groq_api_key"

Do not put your real API key inside app.py.

Do not commit your API key to GitHub.

🌐 Deploy on Streamlit Community Cloud
Step 1 — Create the GitHub repository

Go to GitHub and create a new repository.

Example repository name:

hr-policy-assistant

Keep the repository public if you are using the free Streamlit Community
Cloud workflow.

Step 2 — Add the files

In the GitHub repository, click:

Add file → Create new file

Create:

app.py

Paste the complete app.py code.

Then create:

requirements.txt

Paste the requirements.

Then create:

README.md

Paste this README.

Then create:

.gitignore

Paste the .gitignore content supplied with this project.

Step 3 — Deploy

Open Streamlit Community Cloud.

Sign in using GitHub.

Choose:

Create app

Select:

Repository: your hr-policy-assistant repository
Branch: main
Main file path: app.py

Click:

Deploy

Streamlit will install the packages from requirements.txt.

Step 4 — Add the Groq API key

After the application is created, open the Streamlit app settings.

Go to:

Settings → Secrets

Add:

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

Save the secret.

Then restart/reboot the application if Streamlit asks you to.

Step 5 — Test the application

Open your deployed Streamlit application.

Upload an HR Policy PDF.

For example:

Employee_Handbook.pdf

Wait for:

Policy processed successfully

Then ask questions such as:

How many annual leaves are employees entitled to?
What is the maternity leave policy?
What is the procedure for sick leave?
Who approves annual leave?
What is the company's working-hours policy?

The application will retrieve relevant sections before generating the answer.

⚠️ Important RAG Limitation

The assistant is designed to answer from the uploaded policy.

If the required information is not present in the retrieved policy context,
the assistant should say that the uploaded policy does not provide enough
information.

It should not invent a company policy.

📄 PDF Limitation

The current version works best with PDFs containing selectable text.

Scanned/image-only PDFs may not contain extractable text.

OCR can be added later if required.

🔒 Security

Never place the Groq API key directly inside app.py.

Never commit:

.streamlit/secrets.toml

to GitHub.

Use Streamlit Cloud Secrets instead.

💻 Local Development

This project is designed to be deployed directly from GitHub using
Streamlit Community Cloud.

No Colab, VS Code, or terminal is required for the deployment workflow.

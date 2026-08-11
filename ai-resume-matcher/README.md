# 🤖 AI-Powered Resume & Job Description Matcher

An AI-powered application that analyzes a candidate's resume against a job description using:

- Python
- NLP preprocessing
- LangChain
- Hugging Face
- Sentence Transformers
- FAISS
- Retrieval-Augmented Generation
- Groq/OpenAI LLMs
- PyPDF
- python-docx
- Streamlit

## Architecture

Resume PDF/DOCX
        ↓
Document Parser
        ↓
Text Cleaning
        ↓
LangChain Text Chunking
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Vector Database
        ↓
Semantic Retrieval
        ↓
Relevant Resume Evidence
        ↓
Prompt Template
        ↓
Groq/OpenAI LLM
        ↓
Resume Match Analysis
        ↓
Streamlit UI

## Features

- PDF resume parsing
- DOCX resume parsing
- Text preprocessing
- Semantic chunking
- Hugging Face embeddings
- Sentence Transformer embeddings
- FAISS vector search
- Semantic retrieval
- RAG pipeline
- LLM-based resume analysis
- Matched skills
- Missing skills
- Evidence retrieval
- Improvement suggestions

## Installation

Create virtual environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create `.env`:

LLM_PROVIDER=groq

GROQ_API_KEY=your_key

GROQ_MODEL=llama-3.3-70b-versatile

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

TOP_K=5

Run:

streamlit run app.py

## RAG Pipeline

The application does not send the entire resume blindly to the LLM.

Instead:

1. Resume is parsed.
2. Resume is split into chunks.
3. Chunks are converted into embeddings.
4. Embeddings are stored in FAISS.
5. Job description is used as a retrieval query.
6. Relevant resume chunks are retrieved.
7. Retrieved chunks are inserted into the prompt.
8. LLM analyzes the evidence.
9. Results are displayed in Streamlit.

## Technologies

### Python

Main programming language.

### LangChain

Used for:

- text splitting
- prompt templates
- embeddings integration
- vector store integration
- LLM orchestration

### Hugging Face / Sentence Transformers

Used to convert resume and query text into semantic embeddings.

### FAISS

Used for vector similarity search.

### Groq/OpenAI

Used as the generative LLM layer.

### Streamlit

Used to build the user interface.

## Future Improvements

- Requirement-level retrieval
- Deterministic match percentage
- Skill extraction using an LLM
- Resume section detection
- Experience matching
- Education matching
- ATS score
- Docker
- FastAPI
- Cloud deployment
- Evaluation dataset
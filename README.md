# Study Chatbot

A simple RAG-based chatbot. Upload a PDF study document, and ask questions about
its contents. Answers are generated strictly from the uploaded document — if the
answer isn't in it, the bot says so.

## Stack

- **Backend**: FastAPI + LangChain (Groq LLM, HuggingFace `all-MiniLM-L6-v2`
  embeddings, Chroma vector store, pypdf text extraction)
- **Frontend**: React + Vite

## Setup

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Run:

```powershell
.\.venv\Scripts\python -m uvicorn main:app --port 8000 --app-dir backend
```

The first startup downloads the embedding model (~90 MB) and may take a few
minutes. Subsequent startups use the local cache.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Usage

1. Upload a PDF (drag & drop or browse) and wait for processing to finish.
2. Type questions about the document in the chat box.
3. Each answer shows the source file name and page number(s) below it.
4. Use "New document" to upload a different PDF (this clears the session).

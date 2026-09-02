# 📚 Study Chatbot

A **Retrieval-Augmented Generation (RAG)** based study assistant that allows users to upload PDF documents and ask questions about their contents.

The application extracts text from uploaded PDFs, splits the content into chunks, generates vector embeddings, retrieves the most relevant sections for a user's question, and uses an LLM to generate answers grounded in the uploaded document.

If the requested information is not available in the document, the chatbot clearly states that it could not find the answer instead of generating an unsupported response.

---

## 🚀 Live Demo

**Try the application:**

https://study-chatbot-rag.vercel.app/

---

## ✨ Features

- 📄 **PDF Upload**
  - Upload study materials directly through the web interface.

- 🖱️ **Drag & Drop Upload**
  - Supports drag-and-drop PDF uploads.

- 🔍 **PDF Text Extraction**
  - Extracts text and page information from uploaded PDF documents using PyPDF.

- 🧩 **Document Chunking**
  - Splits extracted text into smaller chunks for efficient retrieval.

- 🧠 **Retrieval-Augmented Generation**
  - Uses semantic similarity search to retrieve relevant document content before generating an answer.

- 💬 **Document-Based Q&A**
  - Ask natural-language questions about the uploaded document.

- 🎯 **Grounded Responses**
  - Answers are generated using the retrieved content from the uploaded document.

- 🚫 **Unsupported Answer Prevention**
  - If relevant information cannot be found in the document, the chatbot states that it could not find the answer.

- 📚 **Source References**
  - Displays the source file and relevant page references used for the response.

- 📝 **Markdown Rendering**
  - AI responses support GitHub-flavored Markdown, including tables and formatted text.

- 🔄 **New Document Sessions**
  - Upload a new document and start a fresh study session.

---

## 🏗️ How It Works

The application follows a standard **RAG pipeline**:

```text
                    ┌─────────────────┐
                    │    Upload PDF   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Extract Text   │
                    │     (PyPDF)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Split Document │
                    │  into Chunks    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Generate Vector │
                    │   Embeddings    │
                    │   (FastEmbed)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Chroma Vector   │
                    │      Store      │
                    └────────┬────────┘
                             │
                             │
                    User asks question
                             │
                             ▼
                    ┌─────────────────┐
                    │ Generate Query  │
                    │    Embedding    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Semantic Search │
                    │   / Retrieval   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relevant Context│
                    │  from Document  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Groq LLM     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Grounded Answer │
                    │ + Source Pages  │
                    └─────────────────┘

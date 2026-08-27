import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("study-chatbot")

load_dotenv()

app = FastAPI(title="Study Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service: RAGService | None = None


@app.on_event("startup")
def load_model():
    global rag_service
    logger.info("Loading embeddings and LLM (this may take a moment)...")
    try:
        rag_service = RAGService()
        logger.info("RAG service ready.")
    except Exception as e:
        logger.exception("Failed to initialise RAG service")
        raise


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": rag_service is not None}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")

    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")

    try:
        result = rag_service.process_pdf(filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to process PDF")
        raise HTTPException(status_code=500, detail="Failed to process the document.")

    return result


@app.post("/api/chat")
def chat(request: ChatRequest):
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return rag_service.ask(request.session_id, question)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))
    except Exception as e:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

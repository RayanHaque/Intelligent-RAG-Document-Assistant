import json
from contextlib import asynccontextmanager
from pathlib import Path
import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from .storage import init_db, save_document, all_chunks, list_documents
from .retrieval import retrieve
from .llm import generate_answer

load_dotenv()

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield

app = FastAPI(title="Intelligent Document Assistant", version="0.1.0", lifespan=lifespan)

class AskRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=2, max_length=2000)

@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/documents")
def documents():
    return list_documents()

@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds 10 MB limit")
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if len(pdf) > 100:
                raise HTTPException(status_code=413, detail="PDF exceeds 100-page limit")
            pages = [page.get_text("text") for page in pdf]
    except HTTPException:
        raise
    except (RuntimeError, ValueError, fitz.FileDataError):
        raise HTTPException(status_code=400, detail="Invalid or unreadable PDF")
    if not any(p.strip() for p in pages):
        raise HTTPException(status_code=422, detail="No selectable text found; scanned PDFs need OCR (not yet implemented)")
    filename = Path(file.filename).name
    return save_document(filename, pages)

def find_document_hits(data: AskRequest):
    if not any(doc["id"] == data.document_id for doc in list_documents()):
        raise HTTPException(status_code=404, detail="Document not found")
    return retrieve(data.question, all_chunks(data.document_id))


@app.post("/api/ask")
async def ask(data: AskRequest, accept: str = Header(default="")):
    hits = await run_in_threadpool(find_document_hits, data)
    sources = [
        {"page": chunk["page"], "excerpt": chunk["content"][:500], "score": chunk["score"]}
        for chunk in hits
    ]
    if accept == "application/x-ndjson":
        async def events():
            yield json.dumps({"type": "sources", "sources": sources}) + "\n"
            answer, mode = await generate_answer(data.question, hits)
            yield json.dumps({"type": "answer", "answer": answer, "mode": mode}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    answer, mode = await generate_answer(data.question, hits)
    return {"answer": answer, "mode": mode, "sources": sources}

# Intelligent Document Assistant — RAG Starter

Independently authored, university-level FastAPI project for PDF question answering with page-based source excerpts. **This is an original starter, not a copy of an existing GitHub RAG repository.**

## Working features
- Upload a **text-based PDF** up to 10 MB and 100 pages (`POST /api/documents`).
- Extract PDF text page-by-page using PyMuPDF; split into overlapping chunks stored in SQLite.
- Rank excerpts with **TF-IDF cosine similarity** over words and two-word phrases, normalize accents, and skip duplicate or zero-score excerpts.
- When `GEMINI_API_KEY` is present and evidence is available, generate a response with Gemini. Answers with missing or unsupported page references fall back to the retrieved excerpts.
- Without an API key, show retrieved excerpts in retrieval-only mode.
- Use the responsive document library and question form to upload PDFs and jump from inline citations to source excerpts. The interactive API is available at `/docs`.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

## Setup (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/ . Add your Gemini API key to `.env` to enable actual LLM responses.

## Tests
```bash
pytest -q
```

## Docker
```bash
docker build -t intelligent-rag .
docker run --rm -p 8000:8000 --env-file .env intelligent-rag
```

## Architecture
PDF → PyMuPDF page extraction → overlapping chunks → SQLite → TF-IDF search → Gemini grounded answer → page references.

## Honest limitations
- **OCR not implemented**: scanned documents without extractable text are rejected.
- **FAISS, neural embeddings, and visual citation highlights not implemented yet**. Implement them in your next iterations, and compare against TF-IDF.
- Citation checks verify the page number against the retrieved excerpts. Use the linked excerpt to check the claim itself.
- No user accounts, rate limiting, multi-tenant isolation or persistence of original PDFs. This is not a production service.
- Treat uploaded documents as untrusted; avoid uploading private files to third-party models.

## Authorship
This original starter is provided for educational customization. Keep a changelog of features and tests you personally implement. Reusing code from another repository later means complying with its license.

# Intelligent Document Assistant — RAG Starter

Independently authored, university-level FastAPI project for PDF question answering with page-based source excerpts. **This is an original starter, not a copy of an existing GitHub RAG repository.**

## Working features
- Upload a **text-based PDF** up to 10 MB and 100 pages (`POST /api/documents`).
- Extract PDF text page-by-page using PyMuPDF; split into overlapping chunks stored in SQLite.
- Retrieve relevant chunks using **TF-IDF cosine similarity** (a simple baseline, not FAISS/neural embeddings).
- When `GEMINI_API_KEY` is present, generate a grounded response with the Gemini API and ask it to cite pages.
- Without an API key, show retrieval-only results rather than pretending an LLM generated an answer.
- Minimal browser interface, `/docs` interactive API, and automated tests.

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
- Page references are grounded in retrieved chunk metadata; LLM page citations should still be verified.
- No user accounts, rate limiting, multi-tenant isolation or persistence of original PDFs. This is not a production service.
- Treat uploaded documents as untrusted; avoid uploading private files to third-party models.

## Authorship
This original starter is provided for educational customization. Keep a changelog of features and tests you personally implement. Reusing code from another repository later means complying with its license.

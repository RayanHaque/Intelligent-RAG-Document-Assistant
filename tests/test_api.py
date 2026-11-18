import fitz
from fastapi.testclient import TestClient
from app.main import app

def test_pdf_upload_and_question(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "rag.db"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((50, 50), "FastAPI makes it easy to build Python web APIs.")
    content = pdf.tobytes()
    pdf.close()
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        upload = client.post('/api/documents', files={'file': ('sample.pdf', content, 'application/pdf')})
        assert upload.status_code == 200, upload.text
        did = upload.json()['id']
        r = client.post('/api/ask', json={'document_id': did, 'question': 'What helps build Python APIs?'})
        assert r.status_code == 200
        assert r.json()['mode'] == 'retrieval-only'
        assert r.json()['sources'][0]['page'] == 1
        assert client.post('/api/ask', json={'document_id': 'unknown', 'question': 'hello there'}).status_code == 404

def test_upload_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_DB_PATH', str(tmp_path / 'rag.db'))
    with TestClient(app) as client:
        r = client.post('/api/documents', files={'file': ('test.txt', b'hello', 'text/plain')})
        assert r.status_code == 400

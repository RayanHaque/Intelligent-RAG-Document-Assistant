import asyncio
import json

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient
from app import llm
from app.main import app

def test_pdf_upload_and_question(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "rag.db"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / ".env")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((50, 50), "FastAPI makes it easy to build Python web APIs.")
    page = pdf.new_page()
    page.insert_text((50, 50), "Orchids need indirect sunlight and regular watering.")
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
        assert [source['page'] for source in r.json()['sources']] == [1]
        assert client.post('/api/ask', json={'document_id': 'unknown', 'question': 'hello there'}).status_code == 404
        monkeypatch.setenv("GEMINI_API_KEY", "unused-test-key")

        def unexpected_client(**kwargs):
            raise AssertionError("No Gemini request should be made without relevant excerpts")

        monkeypatch.setattr(httpx, "AsyncClient", unexpected_client)
        for question in ("Where are penguins found?", "and the or"):
            r = client.post('/api/ask', json={'document_id': did, 'question': question})
            assert r.status_code == 200
            assert r.json() == {
                "answer": "I couldn't find relevant text in this document.",
                "mode": "retrieval-only",
                "sources": [],
            }

def test_upload_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_DB_PATH', str(tmp_path / 'rag.db'))
    with TestClient(app) as client:
        r = client.post('/api/documents', files={'file': ('test.txt', b'hello', 'text/plain')})
        assert r.status_code == 400


@pytest.mark.parametrize("slow_response", [False, True])
def test_ask_stream_keeps_sources_with_gemini(tmp_path, monkeypatch, slow_response):
    monkeypatch.setenv("RAG_DB_PATH", str(tmp_path / "rag.db"))
    monkeypatch.setenv("GEMINI_API_KEY", "mock-test-key")
    monkeypatch.setenv("GEMINI_MODEL", llm.DEFAULT_MODEL)
    monkeypatch.setattr(llm, "GEMINI_TIMEOUT_SECONDS", 0.05)

    async def gemini(request):
        payload = json.loads(request.content)
        assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"
        assert "[Page 1]" in payload["contents"][0]["parts"][0]["text"]
        if slow_response:
            await asyncio.sleep(10)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"thought": True, "text": "Internal reasoning [Page 99]"},
            {"text": "There are 14 classes "},
            {"text": "[Page 1]."},
        ]}}]})

    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(gemini), **kwargs),
    )
    with fitz.open() as pdf:
        pdf.new_page().insert_text((50, 50), "ThoraxSense classifies 14 disease labels.")
        content = pdf.tobytes()

    with TestClient(app) as client:
        upload = client.post("/api/documents", files={"file": ("sample.pdf", content, "application/pdf")})
        assert upload.status_code == 200
        response = client.post(
            "/api/ask",
            json={"document_id": upload.json()["id"], "question": "How many labels does ThoraxSense classify?"},
            headers={"Accept": "application/x-ndjson"},
        )

    assert response.status_code == 200
    sources, answer = map(json.loads, response.text.splitlines())
    assert sources["type"] == "sources"
    assert sources["sources"][0]["page"] == 1
    assert answer["type"] == "answer"
    if slow_response:
        assert answer["mode"] == "retrieval-only"
        assert "took too long" in answer["answer"]
    else:
        assert answer["mode"] == "gemini"
        assert answer["answer"] == "There are 14 classes [Page 1]."

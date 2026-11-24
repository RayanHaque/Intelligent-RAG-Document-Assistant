import asyncio

import httpx
import pytest

from app import llm


@pytest.fixture
def mock_gemini(monkeypatch):
    async_client = httpx.AsyncClient

    def use(handler):
        def mock_client(**kwargs):
            return async_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", mock_client)

    return use


@pytest.mark.parametrize(
    ("model_text", "expected_mode"),
    [
        ("Orchids need indirect sunlight [Page 2].", "gemini"),
        ("Orchids need indirect sunlight [Page 2] [Page 99].", "retrieval-only"),
        ("Orchids need indirect sunlight.", "retrieval-only"),
    ],
)
def test_generate_answer_checks_citations(monkeypatch, mock_gemini, model_text, expected_mode):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-test-key")

    def fake_response(request):
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": model_text}]}}]}
        )

    mock_gemini(fake_response)

    answer, mode = asyncio.run(
        llm.generate_answer(
            "What kind of light do orchids need?",
            [{"page": 2, "content": "Orchids need indirect sunlight."}],
        )
    )
    assert mode == expected_mode
    if expected_mode == "gemini":
        assert answer == model_text
    else:
        assert "couldn't verify" in answer


def test_blank_process_key_uses_saved_key(monkeypatch, tmp_path, mock_gemini):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=saved-test-key\n")
    monkeypatch.setattr(llm, "ENV_FILE", env_file)
    monkeypatch.setenv("GEMINI_API_KEY", " ")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def fake_response(request):
        assert request.headers["x-goog-api-key"] == "saved-test-key"
        assert request.url.path.endswith("/gemini-3.5-flash-lite:generateContent")
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "Two [Page 1]."}]}}]}
        )

    mock_gemini(fake_response)
    answer = asyncio.run(llm.generate_answer("How many?", [{"page": 1, "content": "Two."}]))
    assert answer == ("Two [Page 1].", "gemini")


@pytest.mark.parametrize(
    ("status", "category"),
    [(403, "denied access"), (404, "configured model"), (429, "rate or quota limit")],
)
def test_gemini_error_falls_back_without_exposing_provider_body(monkeypatch, mock_gemini, status, category):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-test-key")
    mock_gemini(lambda request: httpx.Response(status, text="private provider details"))

    answer, mode = asyncio.run(
        llm.generate_answer("How many?", [{"page": 1, "content": "Two."}])
    )
    assert mode == "retrieval-only"
    assert category in answer
    assert f"HTTP {status}" in answer
    assert "private provider details" not in answer

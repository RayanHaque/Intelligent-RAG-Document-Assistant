import asyncio

import httpx
import pytest

from app.llm import generate_answer


@pytest.mark.parametrize(
    ("model_text", "expected_mode"),
    [
        ("Orchids need indirect sunlight [Page 2].", "gemini"),
        ("Orchids need indirect sunlight [Page 2] [Page 99].", "retrieval-only"),
        ("Orchids need indirect sunlight.", "retrieval-only"),
    ],
)
def test_generate_answer_checks_citations(monkeypatch, model_text, expected_mode):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-test-key")
    async_client = httpx.AsyncClient

    def fake_response(request):
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": model_text}]}}]}
        )

    def mock_client(**kwargs):
        return async_client(transport=httpx.MockTransport(fake_response), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)

    answer, mode = asyncio.run(
        generate_answer(
            "What kind of light do orchids need?",
            [{"page": 2, "content": "Orchids need indirect sunlight."}],
        )
    )
    assert mode == expected_mode
    if expected_mode == "gemini":
        assert answer == model_text
    else:
        assert "couldn't verify" in answer

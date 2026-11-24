"""Compare supported Gemini models on two synthetic document questions."""

import asyncio
import re
from statistics import median
from time import perf_counter

import httpx

from .llm import DEFAULT_MODEL, PAGE_CITATION, gemini_key, generate_answer
from .retrieval import retrieve

CANDIDATES = (
    DEFAULT_MODEL,
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
)
CHUNKS = [
    {"page": 1, "content": "The synthetic Project Alder classifier covers 14 categories."},
    {"page": 2, "content": "Project Alder earned a macro F1 of 0.87 on the held-out evaluation set."},
    {"page": 3, "content": "Project Birch covers 20 categories and earned a training macro F1 of 0.95."},
]
QUESTIONS = [
    ("How many categories does Project Alder cover?", r"\b(?:14|fourteen)\b", "1"),
    ("What macro F1 did Project Alder attain on the held-out evaluation set?", r"(?:\b0\.87\b|\b87\s*(?:%|percent))", "2"),
]


async def available_models(key):
    names = set()
    params = {"pageSize": "100"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": key},
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            for entry in payload.get("models", []):
                if "generateContent" in entry.get("supportedGenerationMethods", []):
                    names.add(entry["name"].removeprefix("models/"))
            if not payload.get("nextPageToken"):
                return names
            params["pageToken"] = payload["nextPageToken"]


async def main():
    key = gemini_key()
    if not key:
        print("No Gemini key loaded. Save GEMINI_API_KEY in the project's .env file.")
        return 1

    try:
        available = await available_models(key)
    except httpx.HTTPStatusError as error:
        print(f"Gemini model listing failed (HTTP {error.response.status_code}).")
        return 1
    except (httpx.RequestError, ValueError):
        print("Could not read the Gemini model list.")
        return 1

    models = [model for model in CANDIDATES if model in available][:3]
    if not models:
        print("The key did not list any of the benchmark's Gemini models.")
        return 1

    cases = [(question, pattern, page, retrieve(question, CHUNKS))
             for question, pattern, page in QUESTIONS]
    passed_models = []
    for model in models:
        print(f"\n{model}", flush=True)
        durations = []
        for index, (question, pattern, page, contexts) in enumerate(cases, start=1):
            started = perf_counter()
            answer, mode = await generate_answer(question, contexts, model=model)
            elapsed = perf_counter() - started
            if mode != "gemini":
                print(f"  Q{index} | {elapsed:.2f}s | {answer}", flush=True)
                continue
            citations = set(PAGE_CITATION.findall(answer))
            answer_text = PAGE_CITATION.sub("", answer)
            fact_ok = bool(re.search(pattern, answer_text, re.IGNORECASE))
            page_ok = citations == {page}
            print(f"  Q{index} | {elapsed:.2f}s | correct fact: {fact_ok} | correct page: {page_ok}")
            print(f"  {answer}", flush=True)
            if fact_ok and page_ok:
                durations.append(elapsed)
        if len(durations) == len(cases):
            passed_models.append((median(durations), model))

    if passed_models:
        latency, model = min(passed_models)
        print(f"\nFastest model that passed both questions: {model} ({latency:.2f}s median).")
    else:
        print("\nNo model passed both questions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

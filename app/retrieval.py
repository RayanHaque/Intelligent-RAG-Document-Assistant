from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(question: str, chunks: list[dict], k: int = 4):
    if not chunks or k <= 0:
        return []
    texts = [item["content"] for item in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english", strip_accents="unicode", ngram_range=(1, 2)
    )
    if not vectorizer.build_analyzer()(question):
        return []
    matrix = vectorizer.fit_transform(texts + [question])
    scores = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).ravel()
    top_indices = (-scores).argsort(kind="stable")
    results = []
    seen = set()
    for i in top_indices:
        if scores[i] <= 0:
            break
        chunk = chunks[int(i)]
        normalized = " ".join(chunk["content"].casefold().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append({
            **chunk,
            "score": round(float(scores[i]), 4)
        })
        if len(results) == k:
            break
    return results

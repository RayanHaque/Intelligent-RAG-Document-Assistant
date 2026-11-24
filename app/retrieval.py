from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(question: str, chunks: list[dict], k: int = 4):
    if not chunks:
        return []
    texts = [item["content"] for item in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    if not vectorizer.build_analyzer()(question):
        return []
    matrix = vectorizer.fit_transform(texts + [question])
    scores = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).ravel()
    top_indices = scores.argsort()[::-1][:k]
    results = []
    for i in top_indices:
        if scores[i] <= 0:
            break
        results.append({
            **chunks[int(i)],
            "score": round(float(scores[i]), 4)
        })
    return results

from app.retrieval import retrieve


def test_phrase_match_ranks_first_and_repeated_excerpts_are_skipped():
    chunks = [
        {"page": 1, "content": "The report tracks revenue and operating costs."},
        {"page": 2, "content": "The report tracks operating revenue and costs."},
        {"page": 3, "content": "The report tracks   operating revenue and costs."},
    ]

    results = retrieve("What was the operating revenue?", chunks)

    assert [result["page"] for result in results] == [2, 1]


def test_retrieval_matches_unaccented_question_against_accented_text():
    results = retrieve("What does the cafe serve?", [
        {"page": 1, "content": "The café sells croissants."},
        {"page": 2, "content": "The bakery sells bread."},
    ])

    assert [result["page"] for result in results] == [1]

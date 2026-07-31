import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection(name="transdom_readme")

# Every keyword below was confirmed with `grep` to actually exist in
# transdom_readme.md before being used here — learned the hard way that
# an unverified test case just measures whether your assumption was right,
# not whether retrieval works.
test_cases = [
    {"question": "How does LRU eviction work?", "expected_keyword": "MAX_LOADED_MODELS"},
    {"question": "What does the glossary do?", "expected_keyword": "do_not_translate"},
    {"question": "How many cached translations are kept in memory?", "expected_keyword": "MAX_TRANSLATION_CACHE_SIZE"},
    {"question": "How does semantic caching measure similarity?", "expected_keyword": "cosine similarity"},
    {"question": "How is CORS configured?", "expected_keyword": "ALLOWED_ORIGINS"},
]


def hit_rate_at_k(test_cases: list[dict], k: int = 3) -> float:
    hits = 0

    for case in test_cases:
        results = collection.query(query_texts=[case["question"]], n_results=k)
        retrieved_texts = results["documents"][0]

        found = any(case["expected_keyword"] in chunk for chunk in retrieved_texts)
        status = "HIT" if found else "MISS"
        print(f"[{status}] {case['question']!r} (expecting {case['expected_keyword']!r})")

        if found:
            hits += 1

    return hits / len(test_cases)


score = hit_rate_at_k(test_cases, k=3)
print(f"\nHit Rate @ 3: {score:.0%}")
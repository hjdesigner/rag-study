import chromadb

client = chromadb.PersistentClient(path="./chorma_data")
collection = client.get_or_create_collection(name="transdom_docs")

collection.add(
    documents =[
        "MAX_LOADED_MODELS limits how many translation models are kept in RAM at once, using an LRU eviction policy.",
        "The glossary lets you define terms that should never be translated, or terms with a custom translation.",
        "Semantic caching reuses a translation when a new text is close enough in meaning to something already translated.",
        "Transdom uses CTranslate2 with int8 quantization for faster, lighter translation inference.",
    ],
    ids=["doc1", "doc2", "doc3", "doc4"],
)

results = collection.query(
    query_texts=["What controls memory usage for loaded models?"],
    n_results=2,
)

for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"(distance={distance:.4f}) {doc}")
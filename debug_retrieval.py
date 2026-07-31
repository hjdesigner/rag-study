import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection(name="transdom_readme")

question = "How is CORS configured?"
results = collection.query(query_texts=[question], n_results=10)

for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
    has_keyword = "ALLOWED_ORIGINS" in doc
    print(f"[{i}] distance={distance:.4f} {'<-- has ALLOWED_ORIGINS' if has_keyword else ''}")
    print(f"    {doc[:100]}...")
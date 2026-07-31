import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection(name="transdom_readme")

question = "How does the LRU eviction work for loaded models?"

results = collection.query(query_texts=[question], n_results=3)

for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"(distance={distance:.4f})")
    print(doc)
    print("---")
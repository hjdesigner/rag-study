import requests
import chromadb

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

# Chunks father than this are considered "not actually relevant" and
# get dropped before reaching the prompt. this number came from looking
# at real distances in the previous step = not a formula.
DISTANCE_THRESHOLD = 1.45

def retrieve_relevant_chunks(question: str, n_results: int = 5) -> list[str]:
  client = chromadb.PersistentClient(path='./chroma_data')
  collection = client.get_collection(name="transdom_readme")

  results = collection.query(query_texts=[question], n_results=n_results)

  relevant = []
  for doc, distance in zip(results["documents"][0], results["distances"][0]):
      print(f" distance={distance:.4f} {'(kept)' if distance <= DISTANCE_THRESHOLD else 'dropped'}")
      if distance <= DISTANCE_THRESHOLD:
          relevant.append(doc)

  return relevant

def ask_ollama(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["response"].strip()

def rag_query(question: str) -> str:
    print(f"Question: {question}\n")

    print("Retrieving chunks...")
    chunks = retrieve_relevant_chunks(question)

    if not chunks:
        return "No relevant information found in the indexed document."

    context = "\n\n".join(chunks)

    # Simple, single-instruction prompt - this is deliberate, based on
    # what we learned in Module 1: small local models follow one clear
    # instruction much more reliably than a compound one.
    prompt = f"""Context:
{context}

Based on the context above, answer this question: {question}
"""
    print(f"\nSending {len(chunks)} chunk(s) to the model...\n")
    return ask_ollama(prompt)


if __name__ == "__main__":
    question = "What performance optimizations does Transdom use?"
    answer = rag_query(question)
    print("--- ANSWER ---")
    print(answer)
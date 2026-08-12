import requests
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
DISTANCE_THRESHOLD = 1.45
COLLECTION_NAME = "transdom_readme"

app = FastAPI(title="RAG Study API")

chroma_client = chromadb.PersistentClient(path="./chroma_data")


def retrieve_relevant_chunks(question: str, n_results: int = 5, relative_margin: float = 1.10):
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    results = collection.query(query_texts=[question], n_results=n_results)

    distances = results["distances"][0]
    if not distances:
        return []

    best_distance = min(distances)
    dynamic_threshold = best_distance * relative_margin

    relevant = []
    for doc, distance, metadata in zip(
        results["documents"][0], distances, results["metadatas"][0]
    ):
        print(f"  distance={distance:.4f} {'(kept)' if distance <= dynamic_threshold else '(dropped)'}")
        if distance <= dynamic_threshold:
            relevant.append({"text": doc, "distance": distance, "metadata": metadata})

    return relevant


def ask_ollama(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"""Context:
{context}

Based on the context above, answer this question: {question}
"""
    return ask_ollama(prompt)


BoundedQuestion = Annotated[str, Field(min_length=3, max_length=500)]


class AskRequest(BaseModel):
    question: BoundedQuestion


class Source(BaseModel):
    chunk_index: int
    distance: float
    text_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    chunks = retrieve_relevant_chunks(request.question)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant information found in the indexed document.",
        )

    answer = generate_answer(request.question, chunks)

    sources = [
        Source(
            chunk_index=c["metadata"]["chunk_index"],
            distance=round(c["distance"], 4),
            text_preview=c["text"][:120] + "...",
        )
        for c in chunks
    ]

    return AskResponse(answer=answer, sources=sources)


@app.get("/health")
def health_check():
    return {"status": "ok"}
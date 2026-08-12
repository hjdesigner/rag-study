# RAG Study

Learning project exploring Retrieval-Augmented Generation, building on
concepts from [Transdom](https://github.com/hjdesigner/transdom) (embeddings,
semantic similarity).

## Covered so far
- Vector embeddings and cosine similarity (conceptual + hands-on)
- Vector databases: local (ChromaDB) vs managed (Pinecone)
- Document chunking strategies (naive vs sentence-aware, with overlap)

## Setup
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install chromadb pinecone requests sentence-transformers python-dotenv
```

Copy `.env.example` to `.env` and add your Pinecone API key if testing that part.

## Known limitation: retrieval threshold trade-off

A single relative-distance margin can't perfectly distinguish "one relevant
chunk" from "many relevant chunks" — both cases can have similar distance
distributions with different meanings. `relative_margin = 1.10` was chosen
as a reasonable middle ground after testing both scenarios with real data,
not a perfect solution.

## Known limitation: generation can hallucinate even with correct retrieval

Example observed: asked about performance optimizations, the 3B model
correctly summarized semantic caching from the retrieved context, but also
invented an unrelated "time-series caching" claim not present anywhere in
the source document — a partial hallucination mixed with a partially
correct answer. This is why the faithfulness check (Module 5) exists as a
separate audit tool rather than running on every request: catching this
reliably required an 8B+ model, which is too slow to run synchronously on
every API call.
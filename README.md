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
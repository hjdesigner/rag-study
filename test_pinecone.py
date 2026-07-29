import time
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "transdom-docs"

# Pinecone (unlike ChromaDB) requires you to generate embeddings yourself
# and declare their exact dimensionality upfront when creating the index.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = embedding_model.get_embedding_dimension()

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Index creation is asynchronous — wait until it's ready.
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

documents = [
    "MAX_LOADED_MODELS limits how many translation models are kept in RAM at once, using an LRU eviction policy.",
    "The glossary lets you define terms that should never be translated, or terms with a custom translation.",
    "Semantic caching reuses a translation when a new text is close enough in meaning to something already translated.",
    "Transdom uses CTranslate2 with int8 quantization for faster, lighter translation inference.",
]

vectors = [
    {"id": f"doc{i}", "values": embedding_model.encode(doc).tolist(), "metadata": {"text": doc}}
    for i, doc in enumerate(documents)
]
index.upsert(vectors=vectors)

query = "What controls memory usage for loaded models?"
query_embedding = embedding_model.encode(query).tolist()

results = index.query(vector=query_embedding, top_k=2, include_metadata=True)

for match in results["matches"]:
    print(f"(score={match['score']:.4f}) {match['metadata']['text']}")
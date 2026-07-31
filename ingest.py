import re
import chromadb

SOURCE_FILE = "transdom_readme.md"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def sentence_aware_chunk_with_overlap(text: str, max_size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?]) +", text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:]
        current_chunk += sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def load_document(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    print(f"Loading {SOURCE_FILE}...")
    text = load_document(SOURCE_FILE)

    print("Chunking...")
    chunks = sentence_aware_chunk_with_overlap(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")

    print("Indexing into ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_data")
    # Delete any previous version of this collection, so re-running this
    # script doesn't just keep appending duplicate chunks on top of old ones.
    client.delete_collection(name="transdom_readme") if "transdom_readme" in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(name="transdom_readme")

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_index": i, "source": SOURCE_FILE} for i in range(len(chunks))],
    )

    print(f"Done. {collection.count()} chunks indexed.")


if __name__ == "__main__":
    main()
import re

sample_text = (
    "transdom.js scans the page's DOM for text nodes, skipping script and style tags. "
    "It watches for new content added later via MutationObserver. "
    "It sends batches of text to the server's translate endpoint. "
    "The server runs the appropriate open-source translation model. "
    "The client swaps the translated text back into the page in place." 
)

def naive_chunk(text: str, size: int) -> list[str]:
    """Cuts the text every `size` characters, with zero regard for word
    or sentence boundaries. This is tthe 'bad' baselinem, on purpose."""
    return [text[i:i + size] for i in range(0, len(text), size)]

def sentence_aware_chunk(text: str, max_size: int) -> list[str]:
    """"Splits into sentecens first, then greedily packs whole sentences
    into chuncks, never exceeding max_size and never curtting a sentence."""
    sentences = re.split(r"(?<=[.!?]) +", text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Would adding this sentence push us over the limit?
        if len(current_chunk) + len(sentence) + 1 > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def sentence_aware_chunk_with_overlap(text: str, max_size: int, overlap: int) -> list[str]:
    """Same sentence-packing logioc, but each new chunk starts by repeating
    the tail end of the previous chuck - so an idea split across a chunk
    boundary still has some surrounding context in both chunks"""
    sentences = re.split(r"(?<=[.!?]) +", text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start the next chunkk with the last `overlap` characters
            # of the chuck we just closed, intead of starting empty.
            current_chunk = current_chunk[-overlap:]
        current_chunk += sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

print("--- NAIVE (cuts blindly every 80 chars) ---")
for i, chunk in enumerate(naive_chunk(sample_text, 80)):
    print(f"[{i}] {chunk!r}")

print("\n--- SENTENCE-AWARE (respects sentence boundaries, max 80 chars) ---")
for i, chunk in enumerate(sentence_aware_chunk(sample_text, 80)):
    print(f"[{i}] {chunk!r}")

print("\n--- SENTENCE-AWARE WITH OVERLAP (max 80 chars, 20 char overlap) ---")
for i, chunk in enumerate(sentence_aware_chunk_with_overlap(sample_text, 80, 20)):
    print(f"[{i}] {chunk!r}")
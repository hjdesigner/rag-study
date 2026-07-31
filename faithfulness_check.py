import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"


def ask_ollama(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def check_faithfulness(context: str, answer: str) -> str:
    prompt = f"""Context:
{context}

Answer given: {answer}

Question: Does the answer use only information present in the context above, or does it add something not supported by the context?

Think it through briefly, then end your response with exactly one line in this exact format:
Verdict: YES
or
Verdict: NO
"""
    raw_response = ask_ollama(prompt)

    for line in raw_response.splitlines():
        if line.strip().startswith("Verdict:"):
            return line.strip()

    return f"(no verdict line found) {raw_response}"

context = """
Transdom uses CTranslate2 with int8 quantization for translation, giving a
~6x speedup and roughly half the disk/memory footprint compared to plain
PyTorch, with no observable quality loss on test sentences.
"""

# Case 1: a faithful answer — accurately reflects the context above.
faithful_answer = (
    "Transdom uses CTranslate2 with int8 quantization, which makes translation "
    "about 6 times faster and roughly halves the memory footprint."
)

# Case 2: an UNFAITHFUL answer — invented on purpose. Nothing in the context
# above mentions Chinese support; this is a fabricated claim to test whether
# the judge actually catches hallucination instead of rubber-stamping everything.
unfaithful_answer = (
    "Transdom uses CTranslate2 with int8 quantization, and also supports "
    "real-time translation to Chinese and Japanese out of the box."
)

print("--- Case 1: faithful answer ---")
print(check_faithfulness(context, faithful_answer))

print("\n--- Case 2: unfaithful answer (fabricated claim) ---")
print(check_faithfulness(context, unfaithful_answer))
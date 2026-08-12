import pytest
from fastapi.testclient import TestClient

import main as m
from main import app

client = TestClient(app)


class FakeCollection:
    """Simulates ChromaDB's query() output with hand-picked distances,
    so we can test the relative-margin filter with known, controlled numbers
    instead of depending on a real embedding model."""

    def __init__(self, documents, distances):
        self._documents = documents
        self._distances = distances

    def query(self, query_texts, n_results):
        return {
            "documents": [self._documents[:n_results]],
            "distances": [self._distances[:n_results]],
            "metadatas": [[{"chunk_index": i} for i in range(len(self._documents))][:n_results]],
        }


@pytest.fixture(autouse=True)
def mock_heavy_dependencies(monkeypatch):
    monkeypatch.setattr(m, "ask_ollama", lambda prompt: "FAKE ANSWER based on the context.")
    yield


def use_fake_collection(monkeypatch, documents, distances):
    fake_collection = FakeCollection(documents, distances)
    monkeypatch.setattr(m.chroma_client, "get_collection", lambda name: fake_collection)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(monkeypatch):
    use_fake_collection(
        monkeypatch,
        documents=["Chunk about LRU eviction policy."],
        distances=[1.0],
    )

    response = client.post("/ask", json={"question": "How does LRU work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "FAKE ANSWER based on the context."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["distance"] == 1.0


def test_relative_margin_filters_out_distant_chunks(monkeypatch):
    # One clearly best match, and one far worse — the far one should be dropped.
    use_fake_collection(
        monkeypatch,
        documents=["Best match chunk.", "Much worse match chunk."],
        distances=[1.0, 2.0],  # 2.0 is 100% farther than the best — well over the margin
    )

    response = client.post("/ask", json={"question": "Some question"})

    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) == 1
    assert sources[0]["distance"] == 1.0


def test_relative_margin_keeps_multiple_close_chunks(monkeypatch):
    # Several chunks all close to the best one — all should be kept.
    use_fake_collection(
        monkeypatch,
        documents=["Chunk A.", "Chunk B.", "Chunk C."],
        distances=[1.00, 1.03, 1.05],  # all within ~5% of the best
    )

    response = client.post("/ask", json={"question": "Some question"})

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 3


def test_no_relevant_chunks_returns_404(monkeypatch):
    use_fake_collection(monkeypatch, documents=[], distances=[])

    response = client.post("/ask", json={"question": "Some question"})

    assert response.status_code == 404


def test_question_too_short_is_rejected():
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_question_too_long_is_rejected():
    response = client.post("/ask", json={"question": "a" * 501})
    assert response.status_code == 422
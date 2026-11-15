"""Policy retrieval over the company knowledge base with ChromaDB (Lab-01 recipe).

Documents are split at their numbered sections (one rule block per chunk),
embedded with all-MiniLM-L6-v2 (ChromaDB's default ONNX embedder, runs on the
CPU, no API call) and stored on disk. Policy text never leaves the server
for embedding.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache

import chromadb

from src.config import CHROMA_DIR, KB_DIR

COLLECTION = "fleet_policies"
_lock = threading.Lock()
_SECTION = re.compile(r"^(?:\d+\.|Section\s+\d+(?:\.\d+)?:)", re.MULTILINE)


def chunk_document(text: str) -> list[str]:
    """Split at numbered headings ('1. ZONE A ...', 'Section 4.1: ...'); keep the title on each chunk."""
    lines = text.strip().splitlines()
    title = lines[0].strip()
    body = "\n".join(lines[1:])
    starts = [m.start() for m in _SECTION.finditer(body)] or [0]
    chunks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(body)
        piece = body[start:end].strip()
        if piece:
            chunks.append(f"{title}\n{piece}")
    return chunks


@lru_cache(maxsize=1)
def _collection():
    with _lock:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        if col.count() == 0:
            ids, docs, metas = [], [], []
            for path in sorted(KB_DIR.glob("*.txt")):
                for i, chunk in enumerate(chunk_document(path.read_text(encoding="utf-8"))):
                    ids.append(f"{path.stem}-{i}")
                    docs.append(chunk)
                    metas.append({"source": path.name, "chunk": i})
            col.add(ids=ids, documents=docs, metadatas=metas)
        return col


def search_policies(query: str, k: int = 4) -> dict:
    col = _collection()
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    matches = [
        {"source": meta["source"], "text": doc, "similarity": round(1 - dist, 3)}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    return {"query": query, "matches": matches}


def corpus_stats() -> dict:
    col = _collection()
    return {"chunks": col.count(), "files": len(list(KB_DIR.glob("*.txt")))}

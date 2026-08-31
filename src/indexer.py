"""
indexer.py

Runs ONCE (or whenever knowledge-base/ changes) to build the retrieval
index. It does NOT run per user query -- that's retriever.py's job.

Pipeline:
  chunk_knowledge_base()  -> list of chunk dicts (from chunker.py)
        |
  for each chunk, one at a time:
        text -> local sentence-transformers model -> vector
        |
  save everything to derived/embeddings.npy + derived/chunks.jsonl

Why local embeddings (all-MiniLM-L6-v2):
  - No API key, no billing, no account/region restrictions -- this
    replaces OpenAI (no credits) and Gemini (project access denied),
    both of which were account-level blockers, not code bugs.
  - Runs fine on 8GB RAM for 51 short chunks; ~80MB model, downloaded
    once and cached locally after the first run.

Why separate calls per chunk (not batched):
  - If one chunk fails, you know exactly which one, instead of
    debugging a failed batch of 51.

Why save to disk:
  - We embed once, cache the result, and retriever.py just loads the
    cached vectors instead of re-embedding the whole KB on every query.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from chunker import chunk_knowledge_base

EMBED_MODEL = "all-MiniLM-L6-v2"
DERIVED_DIR = Path("derived")


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    """Embeds a single string locally. One call per chunk."""
    return model.encode(text, convert_to_numpy=True)


def build_index(kb_dir: str = "knowledge-base"):
    print(f"Loading local embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    chunks = chunk_knowledge_base(kb_dir)
    print(f"Chunked {len(chunks)} sections. Embedding one at a time...\n")

    vectors = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            vector = embed_text(model, chunk["text"])
            vectors.append(vector)
            print(f"  [{i}/{len(chunks)}] OK  "
                  f"{chunk['source_file']} :: {chunk['heading']}")
        except Exception as e:
            print(f"  [{i}/{len(chunks)}] FAILED  "
                  f"{chunk['source_file']} :: {chunk['heading']}")
            print(f"    error: {e}")
            raise

    DERIVED_DIR.mkdir(exist_ok=True)

    vector_array = np.array(vectors, dtype=np.float32)
    np.save(DERIVED_DIR / "embeddings.npy", vector_array)

    with open(DERIVED_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, default=str) + "\n")

    print(f"\nSaved {len(vectors)} embeddings -> derived/embeddings.npy")
    print(f"Saved {len(chunks)} chunks -> derived/chunks.jsonl")


if __name__ == "__main__":
    build_index()

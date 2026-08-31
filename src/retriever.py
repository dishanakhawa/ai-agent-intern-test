"""
retriever.py

Runs at QUERY TIME (every time a user asks something) -- unlike
chunker.py and indexer.py, which run once, offline, ahead of time.

Pipeline:
  user query (string)
        |
  embed query -> local sentence-transformers model -> query vector
        |
  cosine similarity vs every stored chunk vector (loaded from disk)
        |
  take top-k most similar chunks (above a similarity floor)
        |
  PRECEDENCE FILTER:
    - hard-exclude any chunk with policy_authority != "official"
      (this is what keeps doc 14's injection text out of the prompt
      entirely -- not "the LLM sees it and is told to ignore it",
      but "the LLM never sees it")
    - sort so active-status chunks come before superseded ones
        |
  CANDIDATE CONFLICT DETECTION (final judgment made in agent.py):
    - pairwise: 2 chunks from different docs, both active/official,
      neither supersedes the other, AND neither uses conditional/
      eligibility-scoping language (member, eligible, active when...)
      -- this filters out most non-conflicts (supersession cases,
      conditional exceptions like TrailPlus vs standard) but does NOT
      by itself prove two remaining docs actually disagree (e.g. doc
      01 vs doc 03 can both survive this filter while just restating
      the same policy). The retriever flags these as CANDIDATES only;
      agent.py reads the actual chunk text and makes the final call
      on whether it's a genuine contradiction. (An earlier attempt to
      resolve this with embedding similarity between the two chunks
      was tested and rejected -- MiniLM embeddings capture topic, not
      stance, so genuine conflicts and genuine agreements scored in
      the wrong relative order. See project notes.)
        |
  return: {"chunks": [...], "candidate_conflict": bool, "candidate_conflict_chunks": [...]}
"""

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
DERIVED_DIR = Path("derived")
TOP_K = 10

_model = None  # loaded once, reused across calls


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_index(derived_dir: Path = DERIVED_DIR):
    """Loads the pre-built embeddings + chunk metadata from disk."""
    vectors = np.load(derived_dir / "embeddings.npy")
    chunks = []
    with open(derived_dir / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return vectors, chunks


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Standard cosine similarity: dot(a,b) / (norm(a) * norm(b))"""
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return doc_norms @ query_norm


def precedence_filter(ranked_chunks: list[dict]) -> list[dict]:
    """
    Hard-excludes anything not policy_authority=='official'.
    This is a pre-filter, not a prompt instruction -- excluded chunks
    never reach the LLM at all.
    """
    return [
        c for c in ranked_chunks
        if c["metadata"].get("policy_authority") == "official"
    ]


CONDITIONAL_SCOPE_PATTERN = re.compile(
    r"\b(member(ship)?|eligib\w*|provided that|only if|as long as|"
    r"must have been (active|valid)|active when|when the (order|membership))\b",
    re.IGNORECASE,
)


def _has_conditional_scope(text: str) -> bool:
    """
    Detects language that scopes a claim to a specific eligibility
    condition (e.g. "TrailPlus members", "must have been active when
    the order was placed"). Text-pattern check on the retrieved
    chunk's own CONTENT, not on doc IDs/titles/the question -- so it
    generalizes to paraphrased queries without hardcoding which
    specific documents are involved.
    """
    return bool(CONDITIONAL_SCOPE_PATTERN.search(text))


def detect_candidate_conflict(chunks: list[dict]) -> tuple[bool, list[dict]]:
    """
    Flags a CANDIDATE conflict -- not a confirmed one. A pair of
    chunks from different source documents, both status=active and
    policy_authority=official, where:
      (a) neither document supersedes the other, AND
      (b) neither chunk's text uses conditional/eligibility-scoping
          language (member, eligible, provided that, active when...)

    This removes the two known non-conflict shapes detectable from
    metadata/text-pattern alone: supersession, and conditional
    exceptions. It does NOT confirm the remaining pairs are genuine
    contradictions -- e.g. doc 01 vs doc 03 can both survive this
    filter while actually agreeing. That judgment requires reading
    the actual text, which is agent.py's job.
    """
    active_official = [
        c for c in chunks
        if c["metadata"].get("status") == "active"
        and c["metadata"].get("policy_authority") == "official"
    ]

    seen_docs = {}
    for c in active_official:
        doc_id = c["metadata"].get("document_id", c["source_file"])
        if doc_id not in seen_docs:
            seen_docs[doc_id] = c
    doc_list = list(seen_docs.values())

    if len(doc_list) < 2:
        return False, []

    candidate_docs = {}
    for a, b in combinations(doc_list, 2):
        md_a, md_b = a["metadata"], b["metadata"]

        if md_a.get("document_id") in {md_b.get("supersedes"), md_b.get("superseded_by")}:
            continue
        if md_b.get("document_id") in {md_a.get("supersedes"), md_a.get("superseded_by")}:
            continue

        if _has_conditional_scope(a["text"]) or _has_conditional_scope(b["text"]):
            continue

        for c in (a, b):
            key = c["metadata"].get("document_id", c["source_file"])
            candidate_docs[key] = c

    if candidate_docs:
        return True, list(candidate_docs.values())

    return False, []


def retrieve(query: str, top_k: int = TOP_K, derived_dir: Path = DERIVED_DIR,
             similarity_floor: float = 0.15):
    """
    Full retrieval pipeline for one user query.

    similarity_floor: chunks scoring below this are dropped even if
    they'd otherwise fit in top_k -- prevents an unrelated but
    active/official doc from leaking into candidate-conflict checks
    just for being one of the top_k nearest by rank.
    """
    model = get_model()
    query_vec = model.encode(query, convert_to_numpy=True)

    vectors, chunks = load_index(derived_dir)
    scores = cosine_similarity(query_vec, vectors)

    ranked_indices = np.argsort(scores)[::-1][:top_k]        # ← TOP_K SLICE HAPPENS FIRST
    ranked_chunks = [
    chunks[i] for i in ranked_indices if scores[i] >= similarity_floor
    ]
    filtered_chunks = precedence_filter(ranked_chunks)         # ← FILTER RUNS AFTER

    filtered_chunks.sort(
        key=lambda c: 0 if c["metadata"].get("status") == "active" else 1
    )

    candidate_conflict, candidate_conflict_chunks = detect_candidate_conflict(filtered_chunks)

    return {
        "chunks": filtered_chunks,
        "candidate_conflict": candidate_conflict,
        "candidate_conflict_chunks": candidate_conflict_chunks,
    }


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What is your return policy?"
    result = retrieve(query)
    print(f"Query: {query!r}\n")
    print(f"Retrieved {len(result['chunks'])} usable chunks (after precedence filter):")
    for c in result["chunks"]:
        print(f"  - [{c['source_file']}] {c['heading']} "
              f"(status={c['metadata'].get('status')})")
    if result["candidate_conflict"]:
        print(f"\nCANDIDATE CONFLICT (needs LLM judgment) between:")
        for c in result["candidate_conflict_chunks"]:
            print(f"  - [{c['source_file']}] {c['heading']}")
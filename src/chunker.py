"""
chunker.py

Reads every .md file in knowledge-base/, splits front matter (YAML)
from the body, then splits the body on '##' headings.

Each chunk = one heading section, tagged with the FULL front-matter
metadata of its parent file. This is what lets retriever.py later do
precedence filtering (status=active, policy_authority=official) and
conflict detection WITHOUT re-opening the source files.

Output: a list of dicts, one per chunk. Later saved to derived/chunks.jsonl
by indexer.py (this file only chunks — it does not embed or save).
"""

import re
from pathlib import Path
import yaml


def split_front_matter(raw_text: str) -> tuple[dict, str]:
    """
    Splits a markdown file's content into (metadata_dict, body_text).
    Front matter is the YAML block between the first two '---' lines.
    """
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw_text, re.DOTALL)
    if not match:
        return {}, raw_text

    front_matter_raw, body = match.groups()
    metadata = yaml.safe_load(front_matter_raw) or {}
    return metadata, body


def split_into_sections(body: str) -> list[dict]:
    """
    Splits the body on '##' (H2) headings.
    Returns a list of {"heading": str, "text": str}.
    """
    parts = re.split(r"\n(?=## )", body.strip())

    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            heading_line, _, rest = part.partition("\n")
            heading = heading_line.replace("## ", "").strip()
            text = rest.strip()
        else:
            # Pre-heading content (the '# Title' line). Skipped below
            # since it carries no answerable content — just the title.
            heading = None
            text = part.strip()

        if text and heading is not None:  # drop bare-title chunks
            sections.append({"heading": heading, "text": text})

    return sections


def chunk_file(filepath: Path) -> list[dict]:
    """
    Chunks a single markdown file into a list of chunk dicts, each with:
      - source_file: filename (for citations)
      - heading: the ## heading this chunk came from
      - text: the section's body text
      - metadata: full front-matter dict (status, policy_authority, etc.)
    """
    raw_text = filepath.read_text(encoding="utf-8")
    metadata, body = split_front_matter(raw_text)
    sections = split_into_sections(body)

    chunks = []
    for section in sections:
        chunks.append({
            "source_file": filepath.name,
            "heading": section["heading"],
            "text": section["text"],
            "metadata": metadata,
        })
    return chunks


def chunk_knowledge_base(kb_dir: str = "knowledge-base") -> list[dict]:
    """
    Chunks every .md file in the knowledge-base directory.
    Returns a flat list of all chunks across all files.
    """
    kb_path = Path(kb_dir)
    all_chunks = []

    for filepath in sorted(kb_path.glob("*.md")):
        file_chunks = chunk_file(filepath)
        all_chunks.extend(file_chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_knowledge_base()
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[{c['source_file']}] heading={c['heading']!r} "
              f"status={c['metadata'].get('status')} "
              f"authority={c['metadata'].get('policy_authority')}")
        print(f"  text preview: {c['text'][:70]!r}")
        print()
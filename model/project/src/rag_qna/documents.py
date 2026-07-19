from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    text: str


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Load plain text and markdown documents from a directory tree."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")

    supported_suffixes = {".txt", ".md", ".markdown"}
    documents: list[tuple[str, str]] = []

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported_suffixes:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((str(path), text))

    if not documents:
        raise ValueError(f"No .txt or .md documents found in: {docs_dir}")

    return documents


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    normalized = "\n".join(paragraphs)
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = end - overlap

    return chunks


def build_chunks(docs_dir: Path, chunk_size: int = 700, overlap: int = 120) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for source, text in load_documents(docs_dir):
        for index, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(DocumentChunk(id=f"{Path(source).stem}-{index}", source=source, text=chunk))

    if not chunks:
        raise ValueError("Document loading succeeded but no chunks were created")

    return chunks


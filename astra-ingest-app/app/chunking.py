"""Paragraph-aware text chunking, ported unchanged from ``ingest_tools.py``.

``chunk_text`` / ``split_long_paragraph`` are pure functions. ``chunk_document``
supplies the size and overlap from ``Settings``.
"""

from __future__ import annotations

import re
from typing import List

from .config import Settings


def split_long_paragraph(text: str, size: int, overlap: int) -> List[str]:
    """Split a long paragraph respecting sentence and word boundaries where possible."""
    step = max(size - overlap, 1)
    if len(text) <= size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        chunks: List[str] = []
        current = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) > size:
                words = s.split()
                w_current = ""
                for w in words:
                    cand = f"{w_current} {w}" if w_current else w
                    if len(cand) <= size:
                        w_current = cand
                    else:
                        if w_current:
                            chunks.append(w_current)
                        w_current = w
                if w_current:
                    chunks.append(w_current)
            else:
                cand = f"{current} {s}" if current else s
                if len(cand) <= size:
                    current = cand
                else:
                    if current:
                        chunks.append(current)
                    current = s
        if current:
            chunks.append(current)
        if chunks:
            return chunks

    return [text[i : i + size] for i in range(0, len(text), step)]


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Paragraph-aware semantic chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_paragraph(para, size, overlap))
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks


def chunk_document(text: str, settings: Settings) -> List[str]:
    """Chunk ``text`` using ``settings.chunk_size`` / ``settings.chunk_overlap``."""
    return chunk_text(text, settings.chunk_size, settings.chunk_overlap)

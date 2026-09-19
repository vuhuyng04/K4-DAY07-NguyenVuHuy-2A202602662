from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    # Split *after* a terminator followed by whitespace, so punctuation stays
    # attached to its sentence (a plain `[.!?]\s+` split would swallow it).
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in self._SENTENCE_BOUNDARY.split(text) if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        step = self.max_sentences_per_chunk
        for start in range(0, len(sentences), step):
            group = sentences[start : start + step]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [c for c in self._split(text, self.separators) if c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: already small enough.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case 2: no separators left (or "" separator) -> hard cut by size.
        if not remaining_separators or remaining_separators[0] == "":
            return self._hard_cut(current_text)

        sep, rest = remaining_separators[0], remaining_separators[1:]
        parts = current_text.split(sep)
        if len(parts) == 1:
            # Separator not present at this level; try the next one.
            return self._split(current_text, rest)

        # Re-attach the separator so no characters are lost between pieces.
        pieces = [p + sep for p in parts[:-1]] + [parts[-1]]

        # Merge upward: pack adjacent small pieces until just under chunk_size.
        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                # Recurse downward on the oversized piece with finer separators.
                chunks.extend(self._split(piece, rest))
            elif len(buffer) + len(piece) <= self.chunk_size:
                buffer += piece
            else:
                chunks.append(buffer)
                buffer = piece
        if buffer:
            chunks.append(buffer)
        return chunks

    def _hard_cut(self, text: str) -> list[str]:
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]


class HeadingChunker:
    """
    Split Markdown text into chunks at heading boundaries (one section per chunk).

    Design rationale (K4-L3A, university regulations): policy documents are
    already organised into sections ("## Điều 4 — ...") by their authors, and
    each section is a self-contained semantic unit. Splitting there keeps the
    rule, its conditions and its numbers together.

    Rules:
        - A section = heading line + everything up to the next heading.
        - Sections longer than chunk_size are split further with
          RecursiveChunker, and the heading is re-attached to every piece so
          that no piece loses the "what is this section about" context.
        - Text without any heading falls back to RecursiveChunker.
    """

    _HEADING = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)

    def __init__(self, chunk_size: int = 800, min_level: int = 1, max_level: int = 6) -> None:
        self.chunk_size = chunk_size
        self.min_level = min_level
        self.max_level = max_level
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections = self._split_sections(text)
        if not sections:
            return self._fallback.chunk(text)

        chunks: list[str] = []
        for heading, body in sections:
            full = f"{heading}\n{body}".strip() if heading else body.strip()
            if not full:
                continue
            if len(full) <= self.chunk_size:
                chunks.append(full)
                continue
            # Too long: split the body, then prefix the heading onto each piece.
            for piece in self._fallback.chunk(body.strip()):
                piece = piece.strip()
                if piece:
                    chunks.append(f"{heading}\n{piece}" if heading else piece)
        return chunks

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Return [(heading_line, body)] — heading may be "" for a preamble."""
        matches = [
            m
            for m in self._HEADING.finditer(text)
            if self.min_level <= len(m.group(1)) <= self.max_level
        ]
        if not matches:
            return []

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = m.group(0).strip()
            body = text[m.end() : end].strip()
            sections.append((heading, body))
        return sections


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = min(50, chunk_size // 10)
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        result: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result

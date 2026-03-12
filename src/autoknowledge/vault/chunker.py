"""Heading-based document chunker with token-cap and overlap."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from autoknowledge.config import IndexConfig
from autoknowledge.types import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Shared encoder (cl100k_base is a good proxy for most embedding model tokenizers)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens])


@dataclass
class _Section:
    heading_path: list[str]
    lines: list[str]
    start_line: int


def _split_into_sections(content: str) -> list[_Section]:
    """Split document content into sections based on Markdown headings."""
    lines = content.splitlines()
    sections: list[_Section] = []
    current_heading_path: list[str] = []
    current_lines: list[str] = []
    current_start = 0
    current_depth: list[int] = []

    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            if current_lines or current_heading_path:
                sections.append(_Section(
                    heading_path=list(current_heading_path),
                    lines=current_lines,
                    start_line=current_start,
                ))
            depth = len(m.group(1))
            title = m.group(2).strip()

            # Trim heading path to current depth, then append
            while current_depth and current_depth[-1] >= depth:
                current_depth.pop()
                if current_heading_path:
                    current_heading_path.pop()
            current_depth.append(depth)
            current_heading_path.append(title)

            current_lines = []
            current_start = i + 1
        else:
            current_lines.append(line)

    if current_lines or current_heading_path:
        sections.append(_Section(
            heading_path=list(current_heading_path),
            lines=current_lines,
            start_line=current_start,
        ))

    return sections


def _make_chunk_id(doc_path: str, heading_path: list[str], start_line: int) -> str:
    key = f"{doc_path}:{'/'.join(heading_path)}:{start_line}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _build_chunk(
    doc: Document,
    heading_path: list[str],
    text: str,
    start_line: int,
    end_line: int,
) -> Chunk:
    doc_path = str(doc.path)
    return Chunk(
        chunk_id=_make_chunk_id(doc_path, heading_path, start_line),
        doc_path=doc_path,
        heading_path=heading_path,
        content=text.strip(),
        token_count=_count_tokens(text),
        start_line=start_line,
        end_line=end_line,
        metadata={
            "title": doc.title,
            "tags": doc.tags,
            "wikilinks": doc.wikilinks,
            "heading": " > ".join(heading_path) if heading_path else "",
            "doc_path": doc_path,
        },
    )


def chunk_document(doc: Document, config: IndexConfig) -> list[Chunk]:
    """Split a Document into Chunks respecting heading structure and token limits."""
    if not doc.raw_content.strip():
        return []

    sections = _split_into_sections(doc.raw_content)
    chunks: list[Chunk] = []
    overlap_text = ""

    for section in sections:
        body = "\n".join(section.lines).strip()
        if not body:
            continue

        # Prefix with overlap from previous chunk
        full_text = (overlap_text + "\n\n" + body).strip() if overlap_text else body

        if _count_tokens(full_text) <= config.chunk_max_tokens:
            chunk = _build_chunk(
                doc, section.heading_path, full_text,
                section.start_line, section.start_line + len(section.lines),
            )
            chunks.append(chunk)
        else:
            # Split large section at paragraph boundaries
            sub_chunks = _split_section(
                doc, section, full_text, config,
            )
            chunks.extend(sub_chunks)

        # Prepare overlap for next chunk
        overlap_text = _truncate_to_tokens(body, config.chunk_overlap_tokens)

    return chunks


def _split_section(
    doc: Document,
    section: _Section,
    text: str,
    config: IndexConfig,
) -> list[Chunk]:
    """Split a section that exceeds chunk_max_tokens at paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    line_offset = section.start_line

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = _count_tokens(para)

        if para_tokens > config.chunk_max_tokens:
            # Single paragraph too long: flush current, then split by sentence
            if current_parts:
                text_block = "\n\n".join(current_parts)
                chunks.append(_build_chunk(
                    doc, section.heading_path, text_block, line_offset,
                    line_offset + text_block.count("\n"),
                ))
                line_offset += text_block.count("\n") + 1
                current_parts = []
                current_tokens = 0

            for sentence_chunk in _split_by_sentences(
                doc, section.heading_path, para, config, line_offset,
            ):
                chunks.append(sentence_chunk)
                line_offset = sentence_chunk.end_line
            continue

        if current_tokens + para_tokens > config.chunk_max_tokens and current_parts:
            text_block = "\n\n".join(current_parts)
            chunks.append(_build_chunk(
                doc, section.heading_path, text_block, line_offset,
                line_offset + text_block.count("\n"),
            ))
            line_offset += text_block.count("\n") + 1
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        text_block = "\n\n".join(current_parts)
        chunks.append(_build_chunk(
            doc, section.heading_path, text_block, line_offset,
            line_offset + text_block.count("\n"),
        ))

    return chunks


def _split_by_sentences(
    doc: Document,
    heading_path: list[str],
    text: str,
    config: IndexConfig,
    start_line: int,
) -> list[Chunk]:
    """Last resort: split text at sentence boundaries, then by token count."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        # If even a single sentence exceeds the max, hard-split by tokens
        if sent_tokens > config.chunk_max_tokens:
            if current_parts:
                block = " ".join(current_parts)
                chunks.append(_build_chunk(doc, heading_path, block, start_line, start_line))
                current_parts = []
                current_tokens = 0
            tokens = _ENCODER.encode(sent)
            for i in range(0, len(tokens), config.chunk_max_tokens):
                block = _ENCODER.decode(tokens[i : i + config.chunk_max_tokens])
                chunks.append(_build_chunk(doc, heading_path, block, start_line, start_line))
            continue

        if current_tokens + sent_tokens > config.chunk_max_tokens and current_parts:
            block = " ".join(current_parts)
            chunks.append(_build_chunk(doc, heading_path, block, start_line, start_line))
            current_parts = []
            current_tokens = 0

        current_parts.append(sent)
        current_tokens += sent_tokens

    if current_parts:
        block = " ".join(current_parts)
        chunks.append(_build_chunk(doc, heading_path, block, start_line, start_line))

    return chunks

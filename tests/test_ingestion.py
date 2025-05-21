import pathlib

import pytest

from pdf_ingestion.ingest import StructuredChunking, FixedSizeRollingWindow, SentenceOverlapChunking, process_document

SAMPLE_TEXT = (
    "# Title\n\n"
    "Introduction paragraph one. More text here.\n\n"
    "## Section A\n\n"
    "Sentence one. Sentence two. Sentence three.\n\n"
    "## Section B\n\n"
    "Another paragraph follows. It has several sentences."
)


def test_structured_chunking():
    chunks = StructuredChunking().process(SAMPLE_TEXT)
    # Should split into at least 3 chunks (Title, Section A, Section B)
    assert len(chunks) >= 3
    assert all("content" in c for c in chunks)


def test_fixed_size_window():
    chunks = FixedSizeRollingWindow(window_size=5, stride=3).process(SAMPLE_TEXT)
    # With very small window, should create multiple chunks
    assert len(chunks) > 1
    assert chunks[0]["start_idx"] == 0


def test_sentence_overlap():
    chunks = SentenceOverlapChunking(chunk_size=2, overlap=1).process(SAMPLE_TEXT)
    # Overlap chunking should produce overlapping segments
    assert len(chunks) >= 2
    first_end = chunks[0]["end_sentence"]
    second_start = chunks[1]["start_sentence"]
    # Because overlap=1, second_start should be first_end -1
    assert second_start == max(0, first_end - 1)


def test_process_document_dispatch():
    results = process_document(SAMPLE_TEXT)
    # Expect 3 strategies returned
    assert len(results) == 3
    for strategy_name, chunks in results.items():
        assert len(chunks) > 0, f"No chunks for {strategy_name}" 
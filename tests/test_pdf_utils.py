import tempfile
from pathlib import Path

from pdf_ingestion.utils import load_document, is_pdf


def test_load_txt_document_roundtrip():
    sample_text = "Hello world\nThis is a test."
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "doc.txt"
        p.write_text(sample_text, encoding="utf-8")
        loaded = load_document(p)
    assert loaded == sample_text
    assert not is_pdf(p) 
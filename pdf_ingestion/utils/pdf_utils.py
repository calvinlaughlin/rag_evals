"""
PDF Utilities
-------------
Utility helpers for working with PDF files.
Currently provides a simple wrapper around ``PyPDF2`` for extracting
text from PDF documents so they can be passed into the existing
chunking strategies.

Note: For large or scanned PDFs, consider swapping for ``pdfminer.six``
or ``pymupdf``.
"""

from __future__ import annotations

import pathlib
from typing import Union, List, Tuple, Optional

import PyPDF2
import logging


__all__ = ["pdf_to_text", "is_pdf", "load_document", "extract_pdf_figures"]


PathLike = Union[str, pathlib.Path]

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:  # pragma: no cover – optional dependency
    fitz = None  # type: ignore
    _PYMUPDF_AVAILABLE = False


def is_pdf(path: PathLike) -> bool:
    """Return ``True`` if *path* points to a PDF file (based solely on suffix)."""
    return pathlib.Path(path).suffix.lower() == ".pdf"


def pdf_to_text(pdf_path: PathLike) -> str:
    """Extract the raw text from *pdf_path* and return it as a single string.

    Parameters
    ----------
    pdf_path:
        Path to the PDF document.

    Returns
    -------
    str
        All text extracted from the pages of the PDF, concatenated with
        newline separators.
    """
    pdf_path = pathlib.Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text_parts: list[str] = []

    # ``PyPDF2`` reads PDF files in binary mode.
    with pdf_path.open("rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page_number, page in enumerate(reader.pages, start=1):
            # ``page.extract_text`` may return ``None`` for pages without
            # extractable text (e.g. scanned images). We guard against that
            # to avoid ``TypeError`` when attempting to join later.
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

    return "\n".join(text_parts)


def load_document(file_path: PathLike) -> str:
    """Load a plain-text or PDF document and return its text."""
    if is_pdf(file_path):
        return pdf_to_text(file_path)

    with open(file_path, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Image extraction – optional multimodal support
# ---------------------------------------------------------------------------

def extract_pdf_figures(pdf_path: PathLike) -> List[Tuple[int, bytes]]:
    """Return list of *(page_number, image_bytes)* tuples extracted from *pdf_path*.

    Extraction uses **PyMuPDF** if available; otherwise we return an empty list
    and log a message.  Page numbers are 1-indexed to match human references in
    the PDF.
    """
    if not _PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not installed – figure extraction skipped. Run `pip install pymupdf` to enable.")
        return []

    pdf_path = pathlib.Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    figures: List[Tuple[int, bytes]] = []
    doc = fitz.open(pdf_path)  # type: ignore[arg-type]
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image.get("image")
            if img_bytes:
                figures.append((page_index + 1, img_bytes))  # page_number 1-indexed
    doc.close()
    return figures


__all__.extend(["extract_pdf_figures"]) 
from __future__ import annotations

"""figure_adapter.py
Utility helpers for converting PDF figure images into pseudo-text chunks so
that they participate in the standard embedding / retrieval flow.

The core helper :func:`build_figure_chunks` takes the list returned by
``pdf_ingestion.utils.pdf_utils.extract_pdf_figures`` and produces chunk
dictionaries compatible with the other ingestion strategies, using the
``CLIPImageProcessor`` for image understanding.
"""

from typing import List, Dict, Any, Tuple

from pdf_ingestion.utils.image_utils import CLIPImageProcessor


_PROCESSOR: CLIPImageProcessor | None = None  # lazy singleton


def _get_processor() -> CLIPImageProcessor:
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = CLIPImageProcessor()
    return _PROCESSOR


def build_figure_chunks(figures: List[Tuple[int, bytes]]) -> List[Dict[str, Any]]:
    """Return list of chunk dictionaries representing *figures*.

    Parameters
    ----------
    figures
        Iterable of ``(page_number, image_bytes)`` extracted from the PDF.

    Returns
    -------
    list of dict
        Each dict has the same schema as text chunks with at least
        ``content`` and ``strategy`` keys so they can pass directly into
        :pyclass:`pdf_ingestion.retrieval.ChunkRetriever`.
    """
    if not figures:
        return []

    proc = _get_processor()
    chunks: List[Dict[str, Any]] = []
    for idx, (page_num, img_bytes) in enumerate(figures, start=1):
        result = proc.process_image(img_bytes)
        desc = proc.extract_structured_data(result)
        chunks.append(
            {
                "content": desc,
                "header": f"Figure {idx} (page {page_num})",
                "page": page_num,
                "strategy": "Figure Chunk",  # special marker for downstream use
            }
        )
    return chunks 
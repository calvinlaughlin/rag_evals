"""PDF Ingestion

A toolkit for experimenting with PDF document chunking strategies and evaluating
their effectiveness for retrieval-augmented generation.
"""

__version__ = "0.1.0"

# Core chunking strategies
from pdf_ingestion.ingest.strategies import (  # noqa
    StructuredChunking,
    FixedSizeRollingWindow,
    SentenceOverlapChunking,
    get_all_strategies,
    process_document,
    process_pdf,
)

# Retrieval
from pdf_ingestion.retrieval.retriever import ChunkRetriever  # noqa

# Utilities
from pdf_ingestion.utils.pdf_utils import load_document, pdf_to_text, is_pdf  # noqa
from pdf_ingestion.utils.image_utils import CLIPImageProcessor  # noqa

# Note: we no longer import ``rag_grade`` here to avoid hard dependency on an
# OpenAI key at *import time*.  Users can call `from pdf_ingestion.evaluation
# import grade_answer` which performs a lazy import when (and only when) they
# have configured the key. This keeps base package import lightweight and test
# friendly.

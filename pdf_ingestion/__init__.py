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
)

# Retrieval
from pdf_ingestion.retrieval.retriever import ChunkRetriever  # noqa

# Utilities
from pdf_ingestion.utils.pdf_utils import load_document, pdf_to_text, is_pdf  # noqa
from pdf_ingestion.utils.image_utils import CLIPImageProcessor  # noqa

# Evaluation helpers
from pdf_ingestion.evaluation.rag_grade import grade_answer  # noqa

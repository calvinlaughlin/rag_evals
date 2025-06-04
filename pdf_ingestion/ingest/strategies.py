"""
PDF Ingestion Strategies Implementation

This module implements various PDF ingestion strategies for comparison:
1. Structured chunking (Reducto-style)
2. Fixed-size rolling windows
3. Sentence-level overlap chunking
"""

import re
import os
from typing import List, Dict, Any

class BaseIngestStrategy:
    """Base class for all ingestion strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, text: str) -> List[Dict[str, Any]]:
        """Process text and return chunks"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def __str__(self) -> str:
        return self.name

class StructuredChunking(BaseIngestStrategy):
    """
    Simulates Reducto-style structured chunking based on document layout
    and semantic boundaries like headers, paragraphs, and sections.
    """
    
    def __init__(self):
        super().__init__("Structured Chunking (Reducto-style)")
    
    def process(self, text: str) -> List[Dict[str, Any]]:
        # Identify headers using regex patterns
        header_pattern = r'#{1,6}\s+(.+)$|([A-Z][A-Za-z\s]+:)'
        
        # Split by potential section boundaries
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_header = "Introduction"
        
        for line in lines:
            # Check if line is a header
            if re.match(header_pattern, line):
                # If we have content in the current chunk, save it
                if current_chunk:
                    chunks.append({
                        "content": '\n'.join(current_chunk),
                        "header": current_header,
                        "strategy": self.name
                    })
                    current_chunk = []
                
                # Update the current header
                current_header = line.strip('# :')
            
            # Add line to current chunk if it's not empty
            if line.strip():
                current_chunk.append(line)
            
            # If we have a significant gap (multiple newlines), consider it a chunk boundary
            elif current_chunk and not current_chunk[-1].strip():
                chunks.append({
                    "content": '\n'.join(current_chunk),
                    "header": current_header,
                    "strategy": self.name
                })
                current_chunk = []
        
        # Add the last chunk if there's content
        if current_chunk:
            chunks.append({
                "content": '\n'.join(current_chunk),
                "header": current_header,
                "strategy": self.name
            })
        
        return chunks

class FixedSizeRollingWindow(BaseIngestStrategy):
    """
    Implements fixed-size rolling window chunking with configurable
    window size and stride.
    """
    
    def __init__(self, window_size: int = 500, stride: int = 250):
        super().__init__(f"Fixed-Size Rolling Window (size={window_size}, stride={stride})")
        self.window_size = window_size
        self.stride = stride
    
    def process(self, text: str) -> List[Dict[str, Any]]:
        # Split text into tokens (words)
        tokens = text.split()
        chunks = []
        
        # Create chunks using rolling window
        for i in range(0, len(tokens), self.stride):
            # Get window of tokens
            window = tokens[i:i + self.window_size]
            
            if window:
                chunks.append({
                    "content": ' '.join(window),
                    "start_idx": i,
                    "end_idx": i + len(window),
                    "strategy": self.name
                })
        
        return chunks

class SentenceOverlapChunking(BaseIngestStrategy):
    """
    Implements sentence-level overlap chunking with configurable
    chunk size (in sentences) and overlap.
    """
    
    def __init__(self, chunk_size: int = 10, overlap: int = 3):
        super().__init__(f"Sentence Overlap Chunking (size={chunk_size}, overlap={overlap})")
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process(self, text: str) -> List[Dict[str, Any]]:
        # Simple sentence splitting (for demonstration)
        sentences = [s.strip() + '.' for s in text.replace('\n', ' ').split('.') if s.strip()]
        chunks = []
        
        # Create chunks with sentence overlap
        for i in range(0, len(sentences), self.chunk_size - self.overlap):
            # Get chunk of sentences
            chunk_sentences = sentences[i:i + self.chunk_size]
            
            if chunk_sentences:
                chunks.append({
                    "content": ' '.join(chunk_sentences),
                    "start_sentence": i,
                    "end_sentence": i + len(chunk_sentences),
                    "strategy": self.name
                })
        
        return chunks

def get_all_strategies() -> List[BaseIngestStrategy]:
    """Return all available ingestion strategies, pulling optional hyper-parameters
    from environment variables so they can be tuned from the command line
    without touching the source.  If an env-var is missing or invalid, the
    default hard-coded values are used.

    Supported environment variables
    --------------------------------
    • WINDOW_SIZE   – integer (default 500)
    • STRIDE        – integer (default 250)
    • SENT_CHUNK_SIZE – integer (default 10)
    • SENT_OVERLAP    – integer (default 3)
    """

    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except ValueError:
            # Fallback to default if conversion fails
            return default

    window_size = _int_env("WINDOW_SIZE", 500)
    stride = _int_env("STRIDE", 250)
    chunk_size = _int_env("SENT_CHUNK_SIZE", 10)
    overlap = _int_env("SENT_OVERLAP", 3)

    return [
        StructuredChunking(),
        FixedSizeRollingWindow(window_size=window_size, stride=stride),
        SentenceOverlapChunking(chunk_size=chunk_size, overlap=overlap),
    ]

def process_document(text: str, strategies=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a document using multiple ingestion strategies
    
    Args:
        text: The document text to process
        strategies: List of strategies to use (defaults to all)
        
    Returns:
        Dictionary mapping strategy names to their chunks
    """
    if strategies is None:
        strategies = get_all_strategies()
    
    results = {}
    for strategy in strategies:
        results[str(strategy)] = strategy.process(text)
    
    return results 

# ---------------------------------------------------------------------------
# Multimodal helpers – attach figure chunks to every strategy
# ---------------------------------------------------------------------------

from pdf_ingestion.utils.pdf_utils import is_pdf, extract_pdf_figures, pdf_to_text
from pdf_ingestion.ingest.figure_adapter import build_figure_chunks


def _attach_figure_chunks(
    chunks_by_strategy: Dict[str, List[Dict[str, Any]]],
    figure_chunks: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Return *chunks_by_strategy* with *figure_chunks* appended to each list."""
    if not figure_chunks:
        return chunks_by_strategy

    for strat, ch_list in chunks_by_strategy.items():
        # We copy to avoid mutating in-place when callers might reuse reference
        chunks_by_strategy[strat] = ch_list + list(figure_chunks)
    return chunks_by_strategy


def process_pdf(pdf_path: str, strategies=None) -> Dict[str, List[Dict[str, Any]]]:
    """End-to-end processing for a **PDF** file including figure extraction.

    1. Extract plain text (``pdf_to_text``)
    2. Chunk text via :func:`process_document`
    3. Extract images via :func:`extract_pdf_figures` + CLIP and convert to
       pseudo-text figure chunks
    4. Append figure chunks to every strategy so they are included during
       retrieval.
    """
    if not is_pdf(pdf_path):
        raise ValueError("process_pdf can only be used with .pdf files")

    # --- Text + standard chunking ---
    text = pdf_to_text(pdf_path)
    chunks_by_strategy = process_document(text, strategies=strategies)

    # --- Figures ---
    figures = extract_pdf_figures(pdf_path)
    figure_chunks = build_figure_chunks(figures)

    # --- Combine ---
    return _attach_figure_chunks(chunks_by_strategy, figure_chunks) 
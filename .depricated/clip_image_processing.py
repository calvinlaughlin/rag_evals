"""
Real CLIP-based image understanding utilities for PDFs.

This module replaces the earlier simulation.  It provides a
``CLIPImageProcessor`` that:
    1. classifies an image into one of a small set of chart/figure types
       using OpenAI/CLIP (via the HuggingFace transformers implementation),
    2. extracts any visible text from the image with Tesseract OCR, and
    3. returns a dictionary that can be converted to structured text for
       ingestion alongside regular PDF chunks.

Dependencies (added to requirements.txt):
    torch, torchvision, transformers, pillow, pytesseract

Example
-------
>>> from pathlib import Path
>>> from clip_image_processing import CLIPImageProcessor
>>> processor = CLIPImageProcessor()
>>> with Path("figure1.png").open("rb") as fh:
...     result = processor.process_image(fh.read(), image_context="Retrieval performance chart")
>>> print(result["description"])
Bar chart detected. Top-5 OCR tokens: ['Structured', 'Fixed-Size', ...]
"""
from __future__ import annotations

import io
import logging
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pytesseract

logger = logging.getLogger(__name__)

# Pre-defined figure categories we care about for research purposes.
LABELS: List[str] = [
    "bar chart",
    "line graph",
    "pie chart",
    "scatter plot",
    "table",
    "flowchart",
    "diagram",
    "infographic",
]


def _load_clip(device: torch.device):
    """Load CLIP model + processor on the given *device*."""
    logger.info("Loading CLIP model on %s", device)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Pre-compute embedding for label texts (normalised)
    with torch.no_grad():
        txt_tokens = processor(text=LABELS, padding=True, return_tensors="pt").to(device)
        text_features = model.get_text_features(**txt_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, processor, text_features


class CLIPImageProcessor:
    """Lightweight wrapper around CLIP + Tesseract for figure understanding."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model, self.processor, self.text_features = _load_clip(self.device)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def process_image(self, image_bytes: bytes, image_context: str | None = None) -> Dict[str, Any]:
        """Classify *image_bytes* and return metadata.

        Parameters
        ----------
        image_bytes:
            Raw image bytes (e.g., extracted from a PDF).
        image_context:
            Optional textual context surrounding the figure in the PDF.
        """
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # ------------------ CLIP classification ------------------
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_feat = self.model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            # cosine similarity
            sims = (img_feat @ self.text_features.T)[0]
            idx = int(torch.argmax(sims))
            image_type = LABELS[idx]
            confidence = float(sims[idx])
        # ------------------ OCR extraction ------------------
        ocr_text = pytesseract.image_to_string(pil_img).strip()
        # ------------------ Build description ------------------
        desc_parts = [f"{image_type.title()} detected (CLIP conf={confidence:.2f})."]
        if ocr_text:
            # show up to first 30 tokens of OCR for brevity
            tokens = ocr_text.split()
            desc_parts.append("Top-5 OCR tokens: " + str(tokens[:5]))
        if image_context:
            desc_parts.append(f"Context snippet: {image_context[:80]}")
        description = " ".join(desc_parts)

        return {
            "image_type": image_type,
            "confidence": confidence,
            "description": description,
            "ocr_text": ocr_text,
        }

    def extract_structured_data(self, processing_result: Dict[str, Any]) -> str:
        """Convert *processing_result* to a plain string for LLM ingestion."""
        return (
            f"Figure Type: {processing_result['image_type']}\n"
            f"CLIP Confidence: {processing_result['confidence']:.2f}\n"
            f"OCR Extract (truncated): {processing_result['ocr_text'][:200]}\n"
        )

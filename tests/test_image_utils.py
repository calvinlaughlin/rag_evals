from io import BytesIO
from PIL import Image
from pdf_ingestion.utils.image_utils import CLIPImageProcessor


def _dummy_png_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color="blue")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_processor_mocked():
    processor = CLIPImageProcessor()
    res = processor.process_image(_dummy_png_bytes(), image_context="context")
    assert "description" in res and "image_type" in res 
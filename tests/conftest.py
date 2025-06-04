import types
import builtins
import pytest
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    """Auto-mock OpenAI client so tests run offline."""
    dummy_json = '{"relevance": 1, "faithfulness": 1, "depth": 1}'
    dummy_choice = types.SimpleNamespace(message=types.SimpleNamespace(content=dummy_json))
    dummy_usage = types.SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    dummy_resp = types.SimpleNamespace(choices=[dummy_choice], usage=dummy_usage)

    class MockChatCompletions:
        def create(self, *args, **kwargs):
            return dummy_resp

    class MockClient:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(completions=MockChatCompletions())

    import importlib
    openai = importlib.import_module("openai")
    monkeypatch.setattr(openai, "OpenAI", MockClient)
    yield


@pytest.fixture(autouse=True)
def mock_clip(monkeypatch):
    """Mock the heavy CLIP model load in image_utils to speed up tests."""
    from pdf_ingestion.utils import image_utils as iu

    def _fake_load_clip(device):
        class _Dummy:
            def get_image_features(self, **kwargs):
                import torch
                return torch.zeros((1, 512))
        import torch
        return _Dummy(), types.SimpleNamespace(**{"__call__": lambda *a, **k: {"pixel_values": torch.zeros((1, 3, 224, 224))}}), torch.zeros((len(iu.LABELS), 512))

    monkeypatch.setattr(iu, "_load_clip", _fake_load_clip)
    yield


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """Mock SentenceTransformer to avoid HF downloads."""
    import numpy as np

    class _MockModel:
        def __init__(self, *args, **kwargs):
            self.dim = 384

        def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
            n = len(texts)
            embeddings = np.random.RandomState(42).randn(n, self.dim).astype("float32")
            
            # Make embeddings deterministic based on content for testing
            for i, text in enumerate(texts):
                if isinstance(text, str):
                    # Create a simple hash-based embedding that's consistent
                    # Put "Reducto" content first by giving it a specific pattern
                    if "Reducto" in text:
                        embeddings[i, :10] = 1.0  # High values in first dimensions
                    elif "Fixed-size" in text:
                        embeddings[i, 10:20] = 1.0  # Different pattern
                    elif "Sentence" in text:
                        embeddings[i, 20:30] = 1.0  # Different pattern
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
            
            return embeddings

    import importlib
    st_mod = importlib.import_module("sentence_transformers")
    monkeypatch.setattr(st_mod, "SentenceTransformer", _MockModel)
    import pdf_ingestion.retrieval.retriever as retriever_mod
    monkeypatch.setattr(retriever_mod, "SentenceTransformer", _MockModel)
    yield 
import types
import builtins
import pytest


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    """Auto-mock openai.ChatCompletion.create so tests run offline."""
    dummy_json = '{"relevance": 1, "faithfulness": 1, "depth": 1}'
    dummy_choice = types.SimpleNamespace(message=types.SimpleNamespace(content=dummy_json))
    dummy_resp = types.SimpleNamespace(choices=[dummy_choice], usage={"prompt_tokens": 0, "completion_tokens": 0})

    def _fake_create(*args, **kwargs):
        return dummy_resp

    import importlib
    openai = importlib.import_module("openai")
    monkeypatch.setattr(openai.ChatCompletion, "create", _fake_create)
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
            return np.zeros((n, self.dim), dtype="float32")

    import importlib
    st_mod = importlib.import_module("sentence_transformers")
    monkeypatch.setattr(st_mod, "SentenceTransformer", _MockModel)
    import pdf_ingestion.retrieval.retriever as retriever_mod
    monkeypatch.setattr(retriever_mod, "SentenceTransformer", _MockModel)
    yield 
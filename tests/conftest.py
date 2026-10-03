import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

SAMPLES = Path(__file__).parent / "samples"


@pytest.fixture
def samples(tmp_path) -> Path:
    d = tmp_path / "mail"
    shutil.copytree(SAMPLES, d)
    return d


@pytest.fixture
def fake_gliner(monkeypatch):
    """Replace the GLiNER model with a keyword matcher so tests need no model download."""
    from mailpipe import extract

    def _load(self, want):
        self.device = "cpu"
        self.model = object()

    def _predict(self, texts):
        out = []
        for t in texts:
            ents = []
            for word, label in (("Contoso", "company"), ("Fabrikam", "company"), ("Bob Chen", "person"),
                                ("AB-12345", "product model"), ("10月15日", "deadline")):
                i = t.find(word)
                if i >= 0:
                    ents.append({"start": i, "end": i + len(word), "text": word, "label": label, "score": 0.9})
            out.append(ents)
        return out

    monkeypatch.setattr(extract.GlinerRunner, "_load", _load)
    monkeypatch.setattr(extract.GlinerRunner, "_predict", _predict)

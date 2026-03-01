from pathlib import Path

import train


def test_train_model_returns_when_dataset_yaml_missing(monkeypatch):
    class FakeYOLO:
        def __init__(self, _name):
            self.train_called = False

        def train(self, **_kwargs):
            self.train_called = True
            return object()

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train.os.path, "exists", lambda _p: False)

    result = train.train_model()

    assert result is False


def test_train_model_uses_expected_training_arguments(monkeypatch):
    recorded = {}

    class FakeYOLO:
        def __init__(self, model_name):
            recorded["model_name"] = model_name

        def train(self, **kwargs):
            recorded["train_kwargs"] = kwargs
            return {"ok": True}

    def fake_exists(path):
        normalized = Path(path).as_posix()
        if normalized.endswith("TVD-2/data.yaml"):
            return True
        if normalized.endswith("runs/train/traffic_violation_large/weights/best.pt"):
            return False
        return True

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train.os.path, "exists", fake_exists)

    result = train.train_model()

    assert recorded["model_name"] == "yolov8l.pt"
    assert recorded["train_kwargs"]["data"].replace("\\", "/").endswith("TVD-2/data.yaml")
    assert recorded["train_kwargs"]["epochs"] == 50
    assert recorded["train_kwargs"]["imgsz"] == 640
    assert recorded["train_kwargs"]["device"] in (0, "cpu")
    assert recorded["train_kwargs"]["batch"] == 8
    assert recorded["train_kwargs"]["amp"] is True
    assert recorded["train_kwargs"]["seed"] == 42
    assert result is True


def test_train_model_propagates_model_load_error(monkeypatch):
    class BrokenYOLO:
        def __init__(self, _name):
            raise RuntimeError("corrupted model")

    monkeypatch.setattr(train, "YOLO", BrokenYOLO)

    try:
        train.train_model()
        assert False, "Expected RuntimeError from YOLO init"
    except RuntimeError as exc:
        assert "corrupted model" in str(exc)

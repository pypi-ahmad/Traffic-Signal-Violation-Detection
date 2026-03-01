import train


def test_training_pipeline_copies_best_model(monkeypatch):
    copied = {}

    class FakeYOLO:
        def __init__(self, _name):
            pass

        def train(self, **_kwargs):
            return {"trained": True}

    def fake_exists(path):
        path = str(path).replace("\\", "/")
        if path.endswith("TVD-2/data.yaml"):
            return True
        if path.endswith("runs/train/traffic_violation_large/weights/best.pt"):
            return True
        return False

    def fake_copy(src, dst):
        copied["src"] = src
        copied["dst"] = dst

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train.os.path, "exists", fake_exists)
    monkeypatch.setattr(train.shutil, "copy", fake_copy)

    train.train_model()

    assert copied["src"].replace("\\", "/").endswith("runs/train/traffic_violation_large/weights/best.pt")
    assert copied["dst"].replace("\\", "/").endswith("/best.pt")


def test_training_pipeline_handles_train_failure(monkeypatch):
    class FakeYOLO:
        def __init__(self, _name):
            pass

        def train(self, **_kwargs):
            raise RuntimeError("train failed")

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train.os.path, "exists", lambda p: True if str(p).replace("\\", "/").endswith("TVD-2/data.yaml") else False)

    result = train.train_model()

    assert result is False

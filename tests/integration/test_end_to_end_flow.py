import download_data
import train


def test_end_to_end_download_then_train(monkeypatch):
    state = {"downloaded": False, "trained": False}

    class FakeDataset:
        location = "./TVD-2"

    class FakeVersion:
        def download(self, fmt):
            assert fmt == "yolov8"
            state["downloaded"] = True
            return FakeDataset()

    class FakeProject:
        def version(self, number):
            assert number == 2
            return FakeVersion()

    class FakeWorkspace:
        def project(self, name):
            assert name == "tvd-kp9qw"
            return FakeProject()

    class FakeRoboflow:
        def __init__(self, api_key):
            assert api_key == "rf-key"

        def workspace(self, name):
            assert name == "traffic-violation-detection"
            return FakeWorkspace()

    class FakeYOLO:
        def __init__(self, model_name):
            assert model_name == "yolov8l.pt"

        def train(self, **kwargs):
            assert kwargs["data"].replace("\\", "/").endswith("TVD-2/data.yaml")
            state["trained"] = True
            return {"ok": True}

    monkeypatch.setenv("ROBOFLOW_API_KEY", "rf-key")
    monkeypatch.setattr(download_data, "Roboflow", FakeRoboflow)

    def fake_exists(path):
        path = str(path).replace("\\", "/")
        if path.endswith("TVD-2/data.yaml"):
            return True
        if path.endswith("runs/train/traffic_violation_large/weights/best.pt"):
            return False
        return False

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train.os.path, "exists", fake_exists)

    download_data.download_dataset()
    train.train_model()

    assert state["downloaded"] is True
    assert state["trained"] is True

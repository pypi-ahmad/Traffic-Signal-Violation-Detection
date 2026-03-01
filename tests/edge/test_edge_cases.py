import download_data
import train

from tests.conftest import FakeModel, FakeResult, FakeVideoCapture


def test_edge_train_missing_best_weights_does_not_copy(monkeypatch):
    copied = {"called": False}

    class FakeYOLO:
        def __init__(self, _name):
            pass

        def train(self, **_kwargs):
            return {"ok": True}

    def fake_exists(path):
        path = str(path).replace("\\", "/")
        if path.endswith("TVD-2/data.yaml"):
            return True
        if path.endswith("runs/train/traffic_violation_large/weights/best.pt"):
            return False
        return False

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    monkeypatch.setattr(train.os.path, "exists", fake_exists)
    monkeypatch.setattr(train.shutil, "copy", lambda *_a, **_k: copied.update({"called": True}))

    result = train.train_model()

    assert copied["called"] is False
    assert result is True


def test_edge_download_raises_system_exit_on_roboflow_error(monkeypatch):
    class BrokenRoboflow:
        def __init__(self, _api_key):
            raise RuntimeError("service unavailable")

    monkeypatch.setenv("ROBOFLOW_API_KEY", "x")
    monkeypatch.setattr(download_data, "Roboflow", BrokenRoboflow)

    try:
        download_data.download_dataset()
        assert False, "Expected SystemExit on Roboflow failure"
    except SystemExit as exc:
        assert exc.code == 1


def test_edge_inference_with_empty_capture_frame_stream(app_module):
    app = app_module
    app.model_vehicles = FakeModel(results=[FakeResult()])
    app.model_custom = None

    cap = FakeVideoCapture(frames=[])
    app.process_video(cap, 0.5)

    assert cap._released is True


def test_edge_inference_with_invalid_video_dimensions(app_module):
    """When video source reports zero dimensions, process_video exits gracefully."""
    app = app_module
    app.model_vehicles = FakeModel(results=[FakeResult()])
    app.model_custom = None

    cap = FakeVideoCapture(frames=[], width=0, height=0)
    app.process_video(cap, 0.5)

    assert cap._released is True

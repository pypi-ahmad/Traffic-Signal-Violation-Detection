import importlib
import sys
import types

import numpy as np

from tests.conftest import FakeBoxes, FakeModel, FakeResult, FakeVideoCapture


def _make_fake_streamlit():
    """Build a self-contained fake streamlit module for isolated import tests."""

    class _CtxCol:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeStreamlit(types.ModuleType):
        def __init__(self):
            super().__init__("streamlit")
            self.sidebar = types.SimpleNamespace(
                title=lambda *_a, **_k: None,
                slider=lambda *_a, **_k: 0.35,
                markdown=lambda *_a, **_k: None,
                radio=lambda _label, opts, index=0: opts[index],
            )
            self._stopped = False

        def cache_resource(self, fn):
            return fn

        def set_page_config(self, **_kwargs):
            return None

        def title(self, *_a, **_k):
            return None

        def info(self, *_a, **_k):
            return None

        def warning(self, *_a, **_k):
            return None

        def error(self, *_a, **_k):
            return None

        def stop(self):
            self._stopped = True
            raise _StopExecution()

        def empty(self):
            return types.SimpleNamespace(
                metric=lambda *_a, **_k: None,
                error=lambda *_a, **_k: None,
                success=lambda *_a, **_k: None,
                image=lambda *_a, **_k: None,
            )

        def columns(self, n):
            return tuple(_CtxCol() for _ in range(n))

        def file_uploader(self, *_a, **_k):
            return None

        def button(self, *_a, **_k):
            return False

    return FakeStreamlit()


def _make_fake_cv2():
    class FakeCv2(types.ModuleType):
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        COLOR_BGR2RGB = 1
        FONT_HERSHEY_SIMPLEX = 0

        def cvtColor(self, frame, _code):
            return frame

        def line(self, *_a, **_k):
            return None

        def putText(self, *_a, **_k):
            return None

        def rectangle(self, *_a, **_k):
            return None

        def VideoCapture(self, _source):
            return FakeVideoCapture()

    return FakeCv2("cv2")


class _StopExecution(Exception):
    """Raised by FakeStreamlit.stop() to halt module-level execution."""
    pass


def test_load_models_returns_none_for_custom_when_second_model_fails(monkeypatch):
    calls = []

    def fake_yolo(name):
        calls.append(name)
        if name == "best.pt":
            raise RuntimeError("bad model")
        return FakeModel(name=name)

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = fake_yolo

    monkeypatch.setitem(sys.modules, "streamlit", _make_fake_streamlit())
    monkeypatch.setitem(sys.modules, "cv2", _make_fake_cv2())
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)

    sys.modules.pop("app", None)
    app = importlib.import_module("app")

    assert calls[0] == "yolov8n.pt"
    assert calls[1] == "best.pt"
    assert app.model_custom is None


def test_load_models_stops_app_when_primary_model_fails(monkeypatch):
    """When yolov8n.pt fails to load, load_models returns (None, None) and st.stop() is called."""
    calls = []

    def fake_yolo(name):
        calls.append(name)
        raise RuntimeError("model not found")

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = fake_yolo

    fake_st = _make_fake_streamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setitem(sys.modules, "cv2", _make_fake_cv2())
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)

    sys.modules.pop("app", None)
    try:
        app = importlib.import_module("app")
    except _StopExecution:
        pass

    assert calls == ["yolov8n.pt"]
    assert fake_st._stopped is True


def test_process_video_calls_tracking_models_and_releases_capture(app_module):
    app = app_module

    veh_result = FakeResult(
        boxes=FakeBoxes(xyxy=[[10, 10, 100, 120]], ids=[1], clss=[2]),
        names={2: "car"},
    )
    custom_result = FakeResult(
        boxes=FakeBoxes(xyxy=[[20, 20, 110, 130]], ids=[2], clss=[0]),
        names={0: "No helmet"},
    )

    vehicle_model = FakeModel(name="yolov8n.pt", results=[veh_result])
    custom_model = FakeModel(name="best.pt", results=[custom_result])

    app.model_vehicles = vehicle_model
    app.model_custom = custom_model
    app.is_red = True
    app.confidence = 0.35

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = FakeVideoCapture(frames=[frame])

    app.process_video(cap, 0.5)

    assert len(vehicle_model.calls) == 1
    assert vehicle_model.calls[0]["classes"] == [2, 3, 5, 7]
    assert len(custom_model.calls) == 1
    assert custom_model.calls[0]["conf"] == 0.35
    assert cap._released is True


def test_process_video_handles_empty_input_without_model_calls(app_module):
    app = app_module

    vehicle_model = FakeModel(name="yolov8n.pt", results=[FakeResult()])
    custom_model = FakeModel(name="best.pt", results=[FakeResult()])
    app.model_vehicles = vehicle_model
    app.model_custom = custom_model

    cap = FakeVideoCapture(frames=[])
    app.process_video(cap, 0.6)

    assert len(vehicle_model.calls) == 0
    assert len(custom_model.calls) == 0
    assert cap._released is True

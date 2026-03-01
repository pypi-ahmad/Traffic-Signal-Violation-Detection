import importlib
import sys
import types

import numpy as np
import pytest


class _FakeTensor:
    def __init__(self, values):
        self._values = np.array(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeBoxes:
    def __init__(self, xyxy=None, ids=None, clss=None):
        self.xyxy = _FakeTensor(xyxy or [])
        self.id = _FakeTensor(ids) if ids is not None else None
        self.cls = _FakeTensor(clss or [])


class FakeResult:
    def __init__(self, boxes=None, names=None):
        self.boxes = boxes or FakeBoxes()
        self.names = names or {}


class FakeModel:
    def __init__(self, name=None, results=None):
        self.name = name
        self.results = results if results is not None else [FakeResult()]
        self.calls = []
        self.names = {0: "No helmet", 1: "Triple riding", 2: "Using mobile", 3: "Wheeling"}

    def track(self, frame, **kwargs):
        self.calls.append({"frame": frame, **kwargs})
        return self.results


class FakeVideoCapture:
    def __init__(self, frames=None, width=640, height=480):
        self._frames = list(frames) if frames is not None else []
        self._released = False
        self._width = width
        self._height = height

    def isOpened(self):
        return not self._released and (len(self._frames) >= 0)

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def get(self, prop):
        if prop == 3:
            return self._width
        if prop == 4:
            return self._height
        return 0

    def release(self):
        self._released = True


class _FakeSidebar:
    def __init__(self):
        self._radio_calls = 0

    def title(self, *_args, **_kwargs):
        return None

    def slider(self, label, _min_value, _max_value, default):
        if label == "Detection Confidence":
            return 0.35
        if label == "Line Position":
            return 0.6
        return default

    def markdown(self, *_args, **_kwargs):
        return None

    def radio(self, _label, options, index=0):
        self._radio_calls += 1
        return options[index]


class _FakePlaceholder:
    def __init__(self):
        self.metrics = []
        self.errors = []
        self.successes = []
        self.images = []

    def metric(self, label, value):
        self.metrics.append((label, value))

    def error(self, text):
        self.errors.append(text)

    def success(self, text):
        self.successes.append(text)

    def image(self, image_data, **kwargs):
        self.images.append((image_data, kwargs))


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeStreamlitModule(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.sidebar = _FakeSidebar()
        self._placeholders = []

    def cache_resource(self, fn):
        return fn

    def set_page_config(self, **_kwargs):
        return None

    def title(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def stop(self):
        return None

    def empty(self):
        ph = _FakePlaceholder()
        self._placeholders.append(ph)
        return ph

    def columns(self, n):
        return tuple(_FakeColumn() for _ in range(n))

    def file_uploader(self, *_args, **_kwargs):
        return None

    def button(self, *_args, **_kwargs):
        return False


class FakeCv2Module(types.ModuleType):
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    COLOR_BGR2RGB = 1
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self):
        super().__init__("cv2")

    def VideoCapture(self, source):
        if isinstance(source, FakeVideoCapture):
            return source
        return FakeVideoCapture()

    def cvtColor(self, frame, _code):
        return frame

    def line(self, *_args, **_kwargs):
        return None

    def putText(self, *_args, **_kwargs):
        return None

    def rectangle(self, *_args, **_kwargs):
        return None


@pytest.fixture
def fake_result_factory():
    def _make(ids=None, boxes=None, classes=None, names=None):
        boxes_obj = FakeBoxes(
            xyxy=boxes or [[10, 20, 100, 150]],
            ids=ids,
            clss=classes or [0],
        )
        return FakeResult(boxes=boxes_obj, names=names or {0: "car"})

    return _make


@pytest.fixture
def app_module(monkeypatch):
    fake_st = FakeStreamlitModule()
    fake_cv2 = FakeCv2Module()

    class FakeYOLO:
        calls = []

        def __new__(cls, model_name):
            cls.calls.append(model_name)
            model = FakeModel(name=model_name)
            if model_name == "yolov8n.pt":
                model.names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
            return model

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = FakeYOLO

    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)

    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    return module

import numpy as np

from tests.conftest import FakeBoxes, FakeModel, FakeResult, FakeVideoCapture


def test_inference_pipeline_processes_frames_and_updates_ui(app_module):
    app = app_module
    app.is_red = True
    app.confidence = 0.4

    veh = FakeResult(boxes=FakeBoxes(xyxy=[[5, 5, 50, 120]], ids=[1], clss=[2]), names={2: "car"})
    custom = FakeResult(boxes=FakeBoxes(xyxy=[[10, 10, 60, 130]], ids=[9], clss=[0]), names={0: "No helmet"})

    app.model_vehicles = FakeModel(name="veh", results=[veh])
    app.model_custom = FakeModel(name="custom", results=[custom])

    cap = FakeVideoCapture(frames=[np.zeros((480, 640, 3), dtype=np.uint8)])
    app.process_video(cap, 0.5)

    placeholders = app.st._placeholders
    assert len(placeholders) >= 3
    metric_calls = placeholders[1].metrics
    assert metric_calls
    assert metric_calls[-1][0] == "🚨 Total Violations"


def test_inference_pipeline_supports_no_custom_model(app_module):
    app = app_module
    app.is_red = False
    app.model_vehicles = FakeModel(
        name="veh",
        results=[FakeResult(boxes=FakeBoxes(xyxy=[[5, 5, 20, 30]], ids=[1], clss=[2]), names={2: "car"})],
    )
    app.model_custom = None

    cap = FakeVideoCapture(frames=[np.zeros((360, 480, 3), dtype=np.uint8)])
    app.process_video(cap, 0.6)

    assert len(app.model_vehicles.calls) == 1

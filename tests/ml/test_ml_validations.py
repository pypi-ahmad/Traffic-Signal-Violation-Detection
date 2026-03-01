import numpy as np

from tests.conftest import FakeBoxes, FakeModel, FakeResult, FakeVideoCapture


def test_ml_model_loader_initializes_vehicle_and_custom_models(app_module):
    app = app_module

    vehicle_model, custom_model = app.load_models()

    assert vehicle_model is not None
    assert custom_model is not None


def test_ml_prediction_paths_accept_expected_result_tensor_shapes(app_module):
    app = app_module
    app.is_red = True

    veh = FakeResult(
        boxes=FakeBoxes(xyxy=[[0, 0, 100, 200], [10, 10, 110, 210]], ids=[1, 2], clss=[2, 3]),
        names={2: "car", 3: "motorcycle"},
    )
    custom = FakeResult(
        boxes=FakeBoxes(xyxy=[[15, 15, 120, 230]], ids=[5], clss=[0]),
        names={0: "No helmet"},
    )

    app.model_vehicles = FakeModel(results=[veh])
    app.model_custom = FakeModel(results=[custom])

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    app.process_video(FakeVideoCapture(frames=[frame]), 0.5)

    assert len(app.model_vehicles.calls) == 1
    assert len(app.model_custom.calls) == 1


def test_ml_predictions_with_no_tracking_ids_are_accepted(app_module):
    app = app_module
    app.is_red = False

    empty_ids_result = FakeResult(
        boxes=FakeBoxes(xyxy=[[0, 0, 10, 10]], ids=None, clss=[2]),
        names={2: "car"},
    )
    app.model_vehicles = FakeModel(results=[empty_ids_result])
    app.model_custom = None

    app.process_video(FakeVideoCapture(frames=[np.zeros((100, 100, 3), dtype=np.uint8)]), 0.4)

    assert len(app.model_vehicles.calls) == 1

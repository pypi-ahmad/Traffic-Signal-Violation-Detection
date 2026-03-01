# TEST REPORT

Date: 2026-03-01
Repository: Traffic-Signal-Violation-Detection

## 1) System Overview

- Main application entry point: `app.py` (Streamlit video inference UI).
- Training pipeline entry point: `train.py` (YOLOv8l training + best-weight copy).
- Dataset ingestion entry point: `download_data.py` (Roboflow download, version 2, YOLOv8 export).
- Dataset config: `TVD-2/data.yaml`.
- Dependency manifest: `requirements.txt`.
- Automated tests: `tests/` (unit, integration, ML, edge).

## 2) Issues Found

Evidence was collected from code inspection and stress/test execution.

### Critical issues identified before fixes

- `app.py`: missing required model (`yolov8n.pt`) caused import-time crash.
- `app.py`: uploaded temp video file was not reliably closed/deleted.
- `app.py`: invalid `st.image(width='stretch')` usage.
- `app.py`: null frame could reach inference and crash.
- `train.py`: hardcoded `device=0` (no CPU fallback).
- `TVD-2/data.yaml`: split paths pointed outside dataset directory.
- `requirements.txt`: invalid CUDA index entry and unnecessary dependency drift.

### Major/minor issues identified before fixes

- `app.py`: broad exception handling on model load (`except:`).
- `app.py`: red-light logic mixed with custom behavioral-violation detections.
- `download_data.py`: non-interactive shells could block on input prompt.
- Dead/unused artifacts existed (`rescue_model.py`, `yolo26n.pt`).

## 3) Tests Created

A complete suite was created under `tests/`:

- `tests/conftest.py`
- `tests/unit/test_app_unit.py`
- `tests/unit/test_train_unit.py`
- `tests/unit/test_download_data_unit.py`
- `tests/integration/test_inference_integration.py`
- `tests/integration/test_train_integration.py`
- `tests/integration/test_end_to_end_flow.py`
- `tests/ml/test_ml_validations.py`
- `tests/edge/test_edge_cases.py`

Final test execution evidence:

- Command: `python -m pytest -q`
- Result: `21 passed`

## 4) Stress Results

Stress was executed in multiple passes (system/data and ML), including real runtime probes.

### System/Data stress evidence

- Large/repeated inference + rapid toggles: `SYSTEM_DATA_STRESS_OK`
- Null-frame handling: `DATA_STRESS_OK null-frame handled`

### ML stress evidence

- CPU/GPU switching runtime probe: `ML_SWITCH_OK cpu+gpu`
- Missing/corrupted model scenarios: `ML_EDGE_STRESS_OK missing/corrupted scenarios`
- Corrupted-weight load check produced expected loader failure behavior (no silent success).

### Stability evidence from validation loop

- Full test rerun: `21 passed`
- Stress reruns completed with success markers above and no unresolved failures.

## 5) Fixes Applied

### Application/runtime fixes

- `app.py`
  - Added guarded required-model load and clean stop behavior when `yolov8n.pt` is unavailable.
  - Replaced broad `except:` with explicit exception handling and messages.
  - Added invalid-dimension guard for video source.
  - Added null-frame skip logic.
  - Corrected `st.image` call to `use_container_width=True`.
  - Added deterministic temp-file close + cleanup for uploaded videos.
  - Separated red-light logic from custom behavioral detections.
  - Ensured capture release via `finally`.

### Training pipeline fixes

- `train.py`
  - Switched to absolute project-root paths for dataset/artifacts.
  - Added GPU/CPU auto-device handling (`0` if CUDA else `cpu`).
  - Added deterministic seed (`seed=42`).
  - Added boolean success/failure returns and process exit code contract.

### Data ingestion fixes

- `download_data.py`
  - Added non-interactive shell guard when API key is missing.

### Dataset/dependency correctness

- `TVD-2/data.yaml`
  - Fixed split paths to in-folder locations:
    - `train: train/images`
    - `val: valid/images`
    - `test: test/images`

- `requirements.txt`
  - Removed invalid CUDA index line.
  - Removed unused `pandas` entry.
  - Kept required runtime/test dependencies including `pytest`.

### Documentation alignment

- `README.md` updated to match actual code behavior, commands, dependencies, and repository contents.

## 6) Cleanup Done

- Deleted dead script: `rescue_model.py`
- Deleted unused artifact: `yolo26n.pt`
- Removed temporary stress helper script after validation run (`_phase6_stress_check.py`).

## 7) Final Stability

Final state is stable based on executed evidence:

- Automated tests: pass (`21 passed`).
- Validation-loop stress reruns: pass (system/data + ML success markers).
- No active regression signals remained in the validated paths.

---

This report includes only evidence observed from repository code, executed tests, and executed stress runs in this workspace session.

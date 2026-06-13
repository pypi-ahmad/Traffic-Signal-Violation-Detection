# Traffic-Signal-Violation-Detection

## Overview

A dual-model computer vision application that detects traffic violations in real time using YOLOv8 object tracking and a Streamlit dashboard. The system enforces red-light stop-line compliance for vehicles and independently flags behavioral infractions such as riding without a helmet, triple riding, mobile phone usage, and wheeling.

## Tech Stack

- Python (requirements.txt based)

## Repository Structure

- `.gitignore`
- `app.py`
- `best.pt`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `download_data.py`
- `LICENSE`
- `README.md`
- `requirements.txt`
- `SECURITY.md`
- `TEST_REPORT.md`
- ... and 5 more entries

## Getting Started

### Prerequisites

- Git
- Runtime dependencies for this project's stack

### Installation

```bash
uv venv
uv pip install -r requirements.txt
```

## Usage

Run the primary app with `uv run app.py`.

## Testing

Run tests with `uv run pytest` from repository root.

## Security

Please review [SECURITY.md](SECURITY.md) for reporting and handling security issues.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening issues or pull requests.

## Changelog

Ongoing changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the terms described in [LICENSE](LICENSE).

import os
import types

import download_data


def test_download_dataset_uses_env_api_key(monkeypatch):
    calls = {}

    class FakeDataset:
        location = "./TVD-2"

    class FakeVersion:
        def __init__(self, version):
            calls["version"] = version

        def download(self, fmt):
            calls["format"] = fmt
            return FakeDataset()

    class FakeProject:
        def version(self, number):
            return FakeVersion(number)

    class FakeWorkspace:
        def project(self, name):
            calls["project"] = name
            return FakeProject()

    class FakeRoboflow:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def workspace(self, name):
            calls["workspace"] = name
            return FakeWorkspace()

    monkeypatch.setattr(download_data, "Roboflow", FakeRoboflow)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "abc123")

    download_data.download_dataset()

    assert calls["api_key"] == "abc123"
    assert calls["workspace"] == "traffic-violation-detection"
    assert calls["project"] == "tvd-kp9qw"
    assert calls["version"] == 2
    assert calls["format"] == "yolov8"


def test_download_dataset_prompts_when_env_missing(monkeypatch):
    calls = {}

    class FakeDataset:
        location = "./TVD-2"

    class FakeVersion:
        def download(self, _fmt):
            return FakeDataset()

    class FakeProject:
        def version(self, _number):
            return FakeVersion()

    class FakeWorkspace:
        def project(self, _name):
            return FakeProject()

    class FakeRoboflow:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def workspace(self, _name):
            return FakeWorkspace()

    monkeypatch.setattr(download_data, "Roboflow", FakeRoboflow)
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(download_data, "sys", types.SimpleNamespace(stdin=types.SimpleNamespace(isatty=lambda: True), exit=download_data.sys.exit))
    monkeypatch.setattr("builtins.input", lambda _prompt: "typed-key")

    download_data.download_dataset()
    assert calls["api_key"] == "typed-key"


def test_download_dataset_exits_when_no_api_key(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(download_data, "sys", types.SimpleNamespace(stdin=types.SimpleNamespace(isatty=lambda: True), exit=download_data.sys.exit))
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    try:
        download_data.download_dataset()
        assert False, "Expected SystemExit when API key missing"
    except SystemExit as exc:
        assert exc.code == 1


def test_download_dataset_exits_in_non_interactive_shell_without_env_key(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(download_data, "sys", types.SimpleNamespace(stdin=types.SimpleNamespace(isatty=lambda: False), exit=download_data.sys.exit))

    try:
        download_data.download_dataset()
        assert False, "Expected SystemExit for non-interactive shell without API key"
    except SystemExit as exc:
        assert exc.code == 1

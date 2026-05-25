"""Pytest config: make project root importable and gate slow integration tests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow integration tests (e.g. USE model load)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (need --runslow to run)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _isolate_config_path(monkeypatch, tmp_path):
    """Point every backend's DEFAULT_CONFIG_PATH at a missing tmp file so tests see defaults."""
    missing = tmp_path / "absent_config.json"
    from backends import classical as classical_module
    from backends import gpt as gpt_module
    from backends import use as use_module
    monkeypatch.setattr(classical_module, "DEFAULT_CONFIG_PATH", missing)
    monkeypatch.setattr(gpt_module, "DEFAULT_CONFIG_PATH", missing)
    monkeypatch.setattr(use_module, "DEFAULT_CONFIG_PATH", missing)

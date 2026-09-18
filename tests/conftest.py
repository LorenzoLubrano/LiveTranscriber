"""Shared pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow `import app.*` when running pytest from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "hardware: needs real audio hardware on this machine"
    )
    config.addinivalue_line(
        "markers", "slow: takes more than a few seconds"
    )

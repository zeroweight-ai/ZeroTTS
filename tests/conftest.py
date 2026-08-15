"""Shared fixtures.

Tests marked ``@pytest.mark.model`` need real weights and are skipped unless
``ZEROTTS_TEST_MODEL`` points at a model directory (or a repo id you are willing
to download). Everything else runs offline in CI.

    ZEROTTS_TEST_MODEL=/path/to/model pytest
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs real weights (ZEROTTS_TEST_MODEL)")


@pytest.fixture(scope="session")
def model_path():
    path = os.environ.get("ZEROTTS_TEST_MODEL")
    if not path:
        pytest.skip("set ZEROTTS_TEST_MODEL to run tests that need weights")
    return path


@pytest.fixture(scope="session")
def tts(model_path):
    from zerotts import ZeroTTS

    return ZeroTTS.from_pretrained(model_path)

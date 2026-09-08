"""Shared pytest fixtures."""

import os
import tempfile

import pytest

# litellm setup exports the OAuth token directories from the real user's home the
# first time a provider module is imported, which happens while pytest collects.
# Claiming them here keeps provider discovery away from the developer's own tokens,
# which otherwise made the startup gate report a configured provider under a
# temporary HOME.
_OAUTH_SANDBOX = tempfile.mkdtemp(prefix="opendde-harness-tests-oauth-")
os.environ["CHATGPT_TOKEN_DIR"] = os.path.join(_OAUTH_SANDBOX, "chatgpt")
os.environ["GITHUB_COPILOT_TOKEN_DIR"] = os.path.join(_OAUTH_SANDBOX, "github_copilot")


@pytest.fixture(autouse=True)
def restore_environment():
    """Undo environment writes made by the code under test.

    Provider setup exports credentials into ``os.environ`` (``ZHIPUAI_API_KEY``
    and friends). ``monkeypatch`` only restores what a test set itself, so those
    writes leaked into later tests.
    """
    snapshot = dict(os.environ)
    yield
    if os.environ != snapshot:
        os.environ.clear()
        os.environ.update(snapshot)

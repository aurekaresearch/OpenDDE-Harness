"""The bundled provider-name snapshot must match the LiteLLM that is installed."""

import importlib.metadata as md

from opendde_harness.providers.litellm_provider_names import LITELLM_PROVIDER_NAMES


def test_snapshot_matches_installed_litellm():
    """A bump that adds or drops a provider fails here, not in `provider list`."""
    import litellm

    installed = {str(getattr(p, "value", p)) for p in litellm.provider_list}

    assert installed == set(LITELLM_PROVIDER_NAMES), (
        f"regenerate for litellm {md.version('litellm')}: "
        f"+{sorted(installed - LITELLM_PROVIDER_NAMES)} -{sorted(LITELLM_PROVIDER_NAMES - installed)}"
    )

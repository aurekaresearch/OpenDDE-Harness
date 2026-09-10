"""The session banner names the thinking level the current model will run with."""

from opendde_harness.config.schema import Config
from opendde_harness.tui_rpc.methods.session import _default_session_info


async def test_the_banner_carries_the_models_overlay_effort_over_the_global_default():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"model": "custom/deep-thinker", "provider": "custom", "reasoningEffort": "low"}},
            "providers": {"custom": {"apiKey": "k", "modelOverlay": {"deep-thinker": {"reasoningEffort": "high"}}}},
        }
    )

    info = await _default_session_info(None, config)

    assert info["reasoning_effort"] == "high"

    config.providers.custom.model_overlay = {}
    assert (await _default_session_info(None, config))["reasoning_effort"] == "low"

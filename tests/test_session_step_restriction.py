from types import SimpleNamespace

import pytest

from data.plugins.astrbot_plugin_novelai.main import NovelAIPlugin


def make_plugin() -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin._initialize_safe_defaults()
    plugin.default_steps = 31
    plugin.enable_style_preset = False
    plugin.selected_style_preset = ""
    return plugin


def test_session_step_override_is_ignored_and_removed():
    plugin = make_plugin()
    event = SimpleNamespace(unified_msg_origin="session-1")
    plugin.session_drawing_settings["session-1"] = {
        "steps": 50,
        "scale": 8.0,
    }

    settings = plugin._get_session_drawing_settings(event)

    assert settings["steps"] == 31
    assert settings["scale"] == 8.0
    assert plugin.session_drawing_settings["session-1"] == {"scale": 8.0}


def test_session_setting_api_rejects_step_override():
    plugin = make_plugin()
    event = SimpleNamespace(unified_msg_origin="session-1")

    plugin._set_session_drawing_setting(event, "steps", 50)

    assert "session-1" not in plugin.session_drawing_settings
    assert plugin._get_session_drawing_settings(event)["steps"] == 31


@pytest.mark.asyncio
async def test_step_command_is_read_only_for_users():
    plugin = make_plugin()
    plugin._check_whitelist = lambda event: True
    event = SimpleNamespace(
        unified_msg_origin="session-1",
        message_str="/画图步数 50",
        plain_result=lambda text: text,
    )

    results = [result async for result in plugin.set_drawing_steps(event)]

    assert results == ["⛔ 画图步数由管理员统一设置，用户不可调整。\n当前步数: 31"]
    assert "session-1" not in plugin.session_drawing_settings

from types import SimpleNamespace

import pytest

from data.plugins.astrbot_plugin_novelai.main import NovelAIPlugin


def make_plugin() -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin.presets = {}
    plugin.style_presets = {}
    plugin.enable_style_preset = False
    plugin.selected_style_preset = ""
    return plugin


@pytest.mark.parametrize(
    ("resolution_name", "expected_resolution"),
    [
        ("长图", "832x1216"),
        ("宽图", "1216x832"),
        ("方图", "1024x1024"),
    ],
)
def test_inline_resolution_overrides_one_request_and_is_removed_from_prompt(
    resolution_name, expected_resolution
):
    plugin = make_plugin()
    settings = {"resolution": "832x1216"}

    prompt, detected_style = plugin.apply_preset_and_style(
        f"{resolution_name} 海边日落",
        drawing_settings=settings,
    )

    assert prompt == "海边日落"
    assert detected_style is None
    assert settings["resolution"] == expected_resolution


def test_draw_command_applies_inline_resolution_before_prompt_processing():
    plugin = make_plugin()
    settings = {"resolution": "832x1216"}
    event = SimpleNamespace(message_str="/画图 宽图 海边日落")

    prompt, detected_style = plugin.extract_prompt_from_message(
        event,
        drawing_settings=settings,
    )

    assert prompt == "海边日落"
    assert detected_style is None
    assert settings["resolution"] == "1216x832"


def test_inline_resolution_supports_comma_separated_prompt_and_last_value_wins():
    plugin = make_plugin()
    settings = {"resolution": "832x1216"}

    prompt, _ = plugin.apply_preset_and_style(
        "长图, 海边, 宽图, sunset",
        drawing_settings=settings,
    )

    assert prompt == "海边, sunset"
    assert settings["resolution"] == "1216x832"


def test_resolution_word_inside_description_is_not_treated_as_a_parameter():
    plugin = make_plugin()
    settings = {"resolution": "1024x1024"}

    prompt, _ = plugin.apply_preset_and_style(
        "一张宽图风格海报",
        drawing_settings=settings,
    )

    assert prompt == "一张宽图风格海报"
    assert settings["resolution"] == "1024x1024"

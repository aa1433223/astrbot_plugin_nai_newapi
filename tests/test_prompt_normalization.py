import asyncio
from types import SimpleNamespace

import pytest

from data.plugins.astrbot_plugin_novelai.main import NAIErrors, NovelAIPlugin
from data.plugins.astrbot_plugin_novelai.novelai_payload import (
    normalize_v5_prompt,
    split_v5_prompt,
)


class FailingProvider:
    """Provider that makes an accidental prompt-enhancement call fail loudly."""

    def __init__(self):
        self.calls = 0

    async def text_chat(self, **_kwargs):
        self.calls += 1
        raise AssertionError("prompt normalization must not call an LLM")


class FakeContext:
    def __init__(self, provider=None):
        self.provider = provider

    def get_using_provider(self, umo=None):
        return self.provider


def make_plugin(provider=None) -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin.context = FakeContext(provider)
    plugin.presets = {}
    plugin.preset_categories = {}
    plugin.style_presets = {}
    plugin.character_aliases = {}
    plugin.character_database = {}
    return plugin


def normalize(plugin, prompt, model="nai-diffusion-5-full", settings=None):
    event = SimpleNamespace(unified_msg_origin="test:session")
    return asyncio.run(
        plugin.normalize_prompt_for_novelai(
            prompt,
            event,
            model,
            user_prompt=prompt,
            drawing_settings=settings,
        )
    )


def test_normalization_is_local_and_does_not_call_an_llm():
    provider = FailingProvider()
    plugin = make_plugin(provider)

    result = normalize(plugin, "A girl beside a pond")

    assert result == "A girl beside a pond"
    assert provider.calls == 0


def test_normalization_does_not_require_a_provider():
    plugin = make_plugin()
    plugin.context = SimpleNamespace()

    assert normalize(plugin, "rainy night, blue umbrella") == (
        "rainy night, blue umbrella"
    )


def test_plain_character_name_is_not_replaced_by_the_character_database():
    plugin = make_plugin()
    plugin.character_aliases = {"hsin": "hsin_(wuthering_waves)"}
    plugin.character_database = {
        "hsin_(wuthering_waves)": {"tag": "hsin_(wuthering_waves)"}
    }

    assert normalize(plugin, "hsin, blue eyes") == "hsin, blue eyes"


def test_character_processing_preserves_normal_names_without_database_lookup():
    plugin = make_plugin()

    def database_must_not_be_used():
        raise AssertionError("ordinary character names must not query the database")

    plugin._ensure_character_database_loaded = database_must_not_be_used

    assert plugin.process_character_prompt("hsin, blue eyes") == "hsin, blue eyes"


def test_original_prompt_does_not_gain_characters_artists_or_quality_tags():
    plugin = make_plugin()
    prompt = "1girl, solo, 1.25::yoneyama mai, -1:: multiple views::"

    result = normalize(plugin, prompt)

    assert result == prompt
    assert "hsin" not in result
    assert "masterpiece" not in result
    assert "very aesthetic" not in result
    assert "herex" not in result
    assert "i-19_(azur_lane)" not in result


def test_explicit_character_sections_are_retained_without_flattened_inference():
    plugin = make_plugin()

    result = normalize(plugin, "1girl, pond | girl, blue eyes")

    assert result == "1girl, solo, pond | girl, blue eyes"
    base, characters = split_v5_prompt(result)
    assert base == "1girl, solo, pond"
    assert characters == ["girl, blue eyes"]


def test_plain_prompt_is_not_split_into_character_sections():
    prompt = "cinematic lighting, girl, red hair, boy, blue coat"

    assert normalize_v5_prompt(prompt) == prompt
    assert split_v5_prompt(prompt) == (prompt, [])


def test_randomizer_bars_are_not_treated_as_character_separators():
    prompt = "1girl, ||red|blue|| hair"

    assert normalize_v5_prompt(prompt) == prompt
    assert split_v5_prompt(prompt) == (prompt, [])


def test_weight_blocks_are_preserved_without_semantic_rewriting():
    prompt = "1girl, solo, 1.25::yoneyama mai, -1:: multiple views::"

    assert normalize_v5_prompt(prompt) == prompt


def test_normalization_removes_legacy_planner_state_from_request_settings():
    plugin = make_plugin()
    settings = {
        "character_positions": [{"x": 0.2, "y": 0.8}],
        "scene_negative_prompt": "photorealistic",
        "character_negative_prompts": ["red hair"],
    }

    normalize(plugin, "1girl, pond", settings=settings)

    assert settings == {}


def test_v5_rejects_more_than_22_explicit_character_sections():
    prompt = "scene | " + " | ".join("girl" for _ in range(23))

    with pytest.raises(NAIErrors.ParameterError, match="最多支持 22 个"):
        normalize(make_plugin(), prompt)


def test_v45_rejects_more_than_6_explicit_character_sections():
    prompt = "scene | " + " | ".join("girl" for _ in range(7))

    with pytest.raises(NAIErrors.ParameterError, match="最多支持 6 个"):
        normalize(make_plugin(), prompt, model="nai-diffusion-4-5-full")


def test_explicit_character_subject_validation_does_not_infer_missing_data():
    with pytest.raises(ValueError, match="必须以 girl、boy 或 other"):
        normalize_v5_prompt("pond | blue eyes")

import json
from pathlib import Path

from data.plugins.astrbot_plugin_novelai.main import Constants, NovelAIPlugin


def make_plugin() -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin.presets = {"人物": "1girl, solo"}
    plugin.style_presets = {
        "动漫": {
            "positive_prompt": "anime style",
            "negative_prompt": "bad anatomy, extra fingers",
            "preview_image": "",
        }
    }
    plugin.enable_style_preset = False
    plugin.selected_style_preset = ""
    plugin.prompt_guidance_rescale = 0.0
    plugin.v4_5_noise_schedule = "karras"
    plugin.v4_5_variety_boost = False
    plugin.v4_5_decrisper = False
    plugin.v4_5_auto_smea = False
    plugin.v4_5_smea = False
    plugin.v4_5_smea_dyn = False
    return plugin


def test_inline_style_applies_positive_and_negative_prompts_to_one_request():
    plugin = make_plugin()
    settings = {
        "style_enabled": False,
        "style_name": "",
    }

    prompt, detected_style = plugin.apply_preset_and_style(
        "人物, 动漫, smile",
        drawing_settings=settings,
    )

    assert prompt == "1girl, solo, smile, anime style"
    assert detected_style == "动漫"
    assert settings["applied_style_name"] == "动漫"
    assert plugin.get_negative_prompt(settings) == "bad anatomy, extra fingers"

    params = plugin.get_model_parameters(
        "nai-diffusion-5-full",
        prompt,
        832,
        1216,
        28,
        7.0,
        "k_euler",
        drawing_settings=settings,
    )
    assert params["negative_prompt"] == "bad anatomy, extra fingers"
    assert (
        params["v4_negative_prompt"]["caption"]["base_caption"]
        == "bad anatomy, extra fingers"
    )


def test_space_separated_inline_style_applies_both_prompt_sides():
    plugin = make_plugin()
    settings = {"style_enabled": False, "style_name": ""}

    prompt, detected_style = plugin.apply_preset_and_style(
        "女孩 动漫",
        drawing_settings=settings,
    )

    assert prompt == "女孩, anime style"
    assert detected_style == "动漫"
    assert settings == {
        "style_enabled": False,
        "style_name": "",
        "applied_style_name": "动漫",
    }
    assert plugin.get_negative_prompt(settings) == "bad anatomy, extra fingers"


def test_session_style_applies_its_negative_prompt_when_no_inline_style_is_used():
    plugin = make_plugin()
    settings = {
        "style_enabled": True,
        "style_name": "动漫",
    }

    prompt, _ = plugin.apply_preset_and_style(
        "1girl, smile",
        drawing_settings=settings,
    )

    assert prompt == "anime style, 1girl, smile"
    assert settings["applied_style_name"] == "动漫"
    assert plugin.get_negative_prompt(settings) == "bad anatomy, extra fingers"


def test_uc_parameter_completely_overrides_request_negative_prompt():
    plugin = make_plugin()
    cleaned, strength, noise, custom_negative = plugin._parse_temp_img2img_params(
        "/画图 -s0.6 -n0.2 2girls, park | girl, red hair | girl, blue hair "
        "--uc lowres, extra characters, blurry"
    )

    assert cleaned == "/画图 2girls, park | girl, red hair | girl, blue hair"
    assert strength == 0.6
    assert noise == 0.2
    assert custom_negative == "lowres, extra characters, blurry"

    settings = {
        "scene_negative_prompt": "photorealistic",
        "character_negative_prompts": ["blue hair", "red hair"],
        "custom_negative_prompt": custom_negative,
    }
    assert plugin.get_negative_prompt(settings) == "lowres, extra characters, blurry"

    params = plugin.get_model_parameters(
        "nai-diffusion-5-full",
        "2girls, park | girl, red hair | girl, blue hair",
        832,
        1216,
        28,
        7.0,
        "k_euler",
        drawing_settings=settings,
    )
    assert params["negative_prompt"] == "lowres, extra characters, blurry"
    assert (
        params["v4_negative_prompt"]["caption"]["base_caption"]
        == "lowres, extra characters, blurry"
    )
    assert [
        item["char_caption"]
        for item in params["v4_negative_prompt"]["caption"]["char_captions"]
    ] == ["", ""]


def test_model_parameters_remove_positive_tags_from_custom_uc():
    plugin = make_plugin()
    settings = {
        "custom_negative_prompt": (
            "blurry, chromatic aberration, film grain, chromatic aberration, bad anatomy"
        )
    }

    params = plugin.get_model_parameters(
        "nai-diffusion-5-full",
        "1girl, chromatic aberration, film grain",
        832,
        1216,
        28,
        7.0,
        "k_euler",
        drawing_settings=settings,
    )

    assert params["negative_prompt"] == "blurry, bad anatomy"
    assert params["v4_negative_prompt"]["caption"]["base_caption"] == (
        "blurry, bad anatomy"
    )


def test_empty_uc_parameter_is_removed_without_matching_longer_words():
    plugin = make_plugin()

    cleaned, _, _, custom_negative = plugin._parse_temp_img2img_params(
        "/画图 1girl --uc"
    )
    untouched, _, _, false_match = plugin._parse_temp_img2img_params(
        "/画图 1girl --ucpreset"
    )

    assert cleaned == "/画图 1girl"
    assert custom_negative is None
    assert untouched == "/画图 1girl --ucpreset"
    assert false_match is None


def test_v5_parameters_submit_multi_character_prompts_as_structured_captions():
    plugin = make_plugin()
    settings = {"negative_enabled": False, "negative_name": ""}

    params = plugin.get_model_parameters(
        "nai-diffusion-5-full",
        "1girl, 1boy, station | girl, red hair | boy, black hair",
        1216,
        832,
        28,
        7.0,
        "k_euler",
        drawing_settings=settings,
    )

    assert params["params_version"] == 4
    assert params["v4_prompt"]["caption"]["base_caption"] == ("1girl, 1boy, station")
    assert [
        item["char_caption"] for item in params["v4_prompt"]["caption"]["char_captions"]
    ] == ["girl, red hair", "boy, black hair"]
    assert len(params["v4_negative_prompt"]["caption"]["char_captions"]) == 2


def test_models_do_not_append_a_built_in_positive_prompt():
    plugin = make_plugin()
    settings = {"negative_enabled": False, "negative_name": ""}

    for model_name in ("nai-diffusion-5-full", "nai-diffusion-5-curated"):
        params = plugin.get_model_parameters(
            model_name,
            "1girl, solo, beach",
            832,
            1216,
            28,
            7.0,
            "k_euler",
            drawing_settings=settings,
        )
        assert params["v4_prompt"]["caption"]["base_caption"] == ("1girl, solo, beach")

    v4_5_params = plugin.get_model_parameters(
        "nai-diffusion-4-5-full",
        "1girl, solo, beach",
        832,
        1216,
        28,
        7.0,
        "k_euler",
        drawing_settings=settings,
    )
    assert v4_5_params["v4_prompt"]["caption"]["base_caption"] == ("1girl, solo, beach")


def test_model_parameters_do_not_enable_server_side_quality_prompt_injection():
    plugin = make_plugin()
    settings = {"negative_enabled": False, "negative_name": ""}

    params = plugin.get_model_parameters(
        "nai-diffusion-5-full",
        "hsin, blue eyes",
        832,
        1216,
        28,
        7.0,
        "k_euler",
        drawing_settings=settings,
    )

    assert params["qualityToggle"] is False


def test_legacy_scene_planner_state_does_not_reach_novelai_parameters():
    plugin = make_plugin()
    positions = [{"x": 0.2, "y": 0.8}, {"x": 0.8, "y": 0.2}]
    settings = {
        "applied_style_name": "动漫",
        "scene_negative_prompt": "photorealistic",
        "character_positions": positions,
        "character_negative_prompts": ["blue hair", "red hair"],
    }

    params = plugin.get_model_parameters(
        "nai-diffusion-5-full",
        "2girls, bubbles | girl, red hair | girl, blue hair",
        832,
        1216,
        28,
        7.0,
        "k_euler",
        drawing_settings=settings,
    )

    assert params["negative_prompt"] == "bad anatomy, extra fingers"
    assert [
        item["centers"][0] for item in params["v4_prompt"]["caption"]["char_captions"]
    ] != positions
    assert [
        item["centers"][0]
        for item in params["v4_negative_prompt"]["caption"]["char_captions"]
    ] != positions
    assert [
        item["char_caption"]
        for item in params["v4_negative_prompt"]["caption"]["char_captions"]
    ] == ["", ""]


def test_multi_character_v5_removes_solo_only_negative_prompt_terms():
    plugin = make_plugin()
    plugin.style_presets["单人画风"] = {
        "positive_prompt": "anime style",
        "negative_prompt": "bad anatomy, little dolls, extra characters, blurry",
        "preview_image": "",
    }
    settings = {"applied_style_name": "单人画风"}

    params = plugin.get_model_parameters(
        "nai-diffusion-5-full",
        "2girls, park | girl, red hair | girl, blue hair",
        1216,
        832,
        28,
        6.5,
        "k_euler_ancestral",
        drawing_settings=settings,
    )

    assert params["negative_prompt"] == "bad anatomy, blurry"
    assert (
        params["v4_negative_prompt"]["caption"]["base_caption"] == "bad anatomy, blurry"
    )


def test_v4_5_advanced_parameters_apply_after_automatic_fallback():
    plugin = make_plugin()

    class ExhaustedUsageManager:
        @staticmethod
        def check_group_daily_limit(quota_scope):
            return False

    plugin.usage_manager = ExhaustedUsageManager()
    settings = {
        "model": "nai-diffusion-5-full",
        "negative_enabled": False,
        "negative_name": "",
        "cfg_rescale": 0.35,
        "v4_5_noise_schedule": "polyexponential",
        "v4_5_variety_boost": True,
        "v4_5_decrisper": True,
        "v4_5_auto_smea": False,
        "v4_5_smea": False,
        "v4_5_smea_dyn": True,
    }

    assert plugin._apply_daily_model_fallback(settings, "group:10001") is True
    assert settings["model"] == Constants.DAILY_FALLBACK_MODEL

    params = plugin.get_model_parameters(
        settings["model"],
        "1girl, solo",
        832,
        1216,
        28,
        6.0,
        "k_euler_ancestral",
        drawing_settings=settings,
    )

    assert params["params_version"] == 4
    assert params["cfg_rescale"] == 0.35
    assert params["noise_schedule"] == "polyexponential"
    assert params["dynamic_thresholding"] is True
    assert params["skip_cfg_above_sigma"] == 58
    assert params["sm"] is True
    assert params["sm_dyn"] is True


def test_v4_5_auto_smea_uses_the_official_resolution_threshold():
    plugin = make_plugin()
    settings = {
        "negative_enabled": False,
        "negative_name": "",
        "v4_5_auto_smea": True,
        "v4_5_smea_dyn": True,
    }

    square_params = plugin.get_model_parameters(
        "nai-diffusion-4-5-full",
        "1girl",
        1024,
        1024,
        28,
        6.0,
        "k_euler_ancestral",
        drawing_settings=settings,
    )
    portrait_params = plugin.get_model_parameters(
        "nai-diffusion-4-5-full",
        "1girl",
        832,
        1216,
        28,
        6.0,
        "k_euler_ancestral",
        drawing_settings=settings,
    )

    assert square_params["sm"] is True
    assert square_params["sm_dyn"] is True
    assert portrait_params["sm"] is False
    assert portrait_params["sm_dyn"] is False


def test_v4_5_settings_do_not_change_normal_v5_sampling_parameters():
    plugin = make_plugin()
    settings = {
        "negative_enabled": False,
        "negative_name": "",
        "cfg_rescale": 0.85,
        "v4_5_noise_schedule": "exponential",
        "v4_5_variety_boost": True,
        "v4_5_decrisper": True,
        "v4_5_auto_smea": True,
        "v4_5_smea": True,
        "v4_5_smea_dyn": True,
    }

    params = plugin.get_model_parameters(
        "nai-diffusion-5-full",
        "1girl",
        1024,
        1024,
        28,
        6.0,
        "k_euler_ancestral",
        drawing_settings=settings,
    )

    assert params["params_version"] == 4
    assert params["cfg_rescale"] == 0.0
    assert params["noise_schedule"] == "karras"
    assert params["dynamic_thresholding"] is False
    assert params["skip_cfg_above_sigma"] is None
    assert "sm" not in params
    assert "sm_dyn" not in params


def test_configuration_schema_exposes_all_v4_5_sampling_controls():
    schema_path = Path(__file__).parents[1] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert {
        "prompt_guidance_rescale",
        "v4_5_noise_schedule",
        "v4_5_variety_boost",
        "v4_5_decrisper",
        "v4_5_auto_smea",
        "v4_5_smea",
        "v4_5_smea_dyn",
    } <= schema.keys()

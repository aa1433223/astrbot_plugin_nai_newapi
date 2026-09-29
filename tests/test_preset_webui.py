import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from data.plugins.astrbot_plugin_novelai.preset_webui import PresetValidationError, PresetWebUIController


class DummyConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_calls = 0
        self.fail_save = False

    def save_config(self):
        self.save_calls += 1
        if self.fail_save:
            raise OSError("disk unavailable")


class DummyContext:
    def __init__(self):
        self.routes = []

    def register_web_api(self, route, handler, methods, description):
        self.routes.append((route, handler, methods, description))


def build_controller(data_dir: Path | None = None):
    presets = {
        "角色甲": "1girl, white hair",
        "制服": "school uniform",
        "海边": "beach, ocean",
        "挥手": "waving",
        "旧预设": "legacy prompt",
        "孤立预设": "orphan prompt",
    }
    categories = {
        "人物": ["角色甲"],
        "衣服": ["制服"],
        "场景": ["海边"],
        "动作": ["挥手"],
        "其他": [],
        "旧分类": ["旧预设"],
    }
    config = DummyConfig(
        presets=json.dumps(presets, ensure_ascii=False),
        preset_categories=json.dumps(categories, ensure_ascii=False),
        style_presets=json.dumps({"旧画风": "legacy style"}, ensure_ascii=False),
    )
    plugin = SimpleNamespace(
        presets=presets,
        preset_categories=categories,
        style_presets={"旧画风": "legacy style"},
        selected_style_preset="",
        session_drawing_settings={},
        config=config,
        get_data_dir=lambda _: data_dir or Path("/tmp/astrbot-novelai-tests"),
    )
    context = DummyContext()
    return PresetWebUIController(context, plugin), context, plugin


def category_rows(state, category_id):
    category = next(item for item in state["categories"] if item["id"] == category_id)
    return [item["name"] for item in category["presets"]]


def test_routes_use_the_plugin_extension_prefix():
    controller, context, _ = build_controller()
    controller.register_routes()

    assert len(context.routes) == 10
    assert context.routes[0][0] == "/astrbot_plugin_novelai/presets/state"
    assert context.routes[0][2] == ["GET"]


def test_state_maps_legacy_and_orphaned_presets_to_unclassified():
    controller, _, _ = build_controller()

    state = controller.serialize_state()

    assert category_rows(state, "character") == ["角色甲"]
    assert category_rows(state, "clothing") == ["制服"]
    assert category_rows(state, "unclassified") == ["旧预设", "孤立预设"]
    assert state["stats"] == {"total": 6, "unclassified": 2}


def test_create_and_case_insensitive_duplicate_validation():
    controller, _, plugin = build_controller()

    state = controller.create_preset(
        {"name": "雨夜", "prompt": "rainy night", "category": "scene"}
    )

    assert plugin.presets["雨夜"] == "rainy night"
    assert plugin.preset_categories["场景"] == ["海边", "雨夜"]
    assert category_rows(state, "scene") == ["海边", "雨夜"]
    assert plugin.config.save_calls == 1

    with pytest.raises(PresetValidationError, match="已存在"):
        controller.create_preset(
            {"name": "制服", "prompt": "duplicate", "category": "clothing"}
        )

    with pytest.raises(PresetValidationError, match="有效的预设分类"):
        controller.create_preset(
            {"name": "无效分类", "prompt": "prompt", "category": {"id": "scene"}}
        )


def test_update_renames_reference_and_moves_category():
    controller, _, plugin = build_controller()

    controller.update_preset(
        {
            "original_name": "角色甲",
            "name": "角色乙",
            "prompt": "1boy, black hair",
            "category": "action",
        }
    )

    assert "角色甲" not in plugin.presets
    assert plugin.presets["角色乙"] == "1boy, black hair"
    assert plugin.preset_categories["人物"] == []
    assert plugin.preset_categories["动作"] == ["挥手", "角色乙"]


def test_bulk_move_preserves_request_order_and_target_index():
    controller, _, plugin = build_controller()

    controller.move_presets(
        {
            "names": ["旧预设", "制服"],
            "target_category": "character",
            "target_index": 0,
        }
    )

    assert plugin.preset_categories["人物"] == ["旧预设", "制服", "角色甲"]
    assert plugin.preset_categories["衣服"] == []
    assert plugin.preset_categories["旧分类"] == []


def test_delete_removes_all_category_references():
    controller, _, plugin = build_controller()
    plugin.preset_categories["其他"] = ["制服"]

    controller.delete_presets({"names": ["制服"]})

    assert "制服" not in plugin.presets
    assert "制服" not in plugin.preset_categories["衣服"]
    assert "制服" not in plugin.preset_categories["其他"]


def test_failed_save_keeps_live_state_and_config_unchanged():
    controller, _, plugin = build_controller()
    original_presets_config = plugin.config["presets"]
    original_categories_config = plugin.config["preset_categories"]
    plugin.config.fail_save = True

    with pytest.raises(OSError, match="disk unavailable"):
        controller.create_preset(
            {"name": "失败预设", "prompt": "prompt", "category": "action"}
        )

    assert "失败预设" not in plugin.presets
    assert plugin.config["presets"] == original_presets_config
    assert plugin.config["preset_categories"] == original_categories_config


def test_style_state_normalizes_legacy_string_entries():
    controller, _, _ = build_controller()

    state = controller.serialize_style_state()

    assert state["styles"] == [
        {
            "name": "旧画风",
            "positive_prompt": "legacy style",
            "negative_prompt": "",
            "preview_image": "",
            "preview_url": "",
        }
    ]
    assert state["stats"] == {"total": 1, "with_preview": 0}


def test_style_preview_optimizer_outputs_bounded_webp():
    controller, _, _ = build_controller()
    source = io.BytesIO()
    Image.new("RGB", (1600, 1200), "#7c5cba").save(source, format="PNG")

    optimized = controller._optimize_style_preview(source.getvalue())

    with Image.open(io.BytesIO(optimized)) as image:
        assert image.format == "WEBP"
        assert image.size == (960, 720)

    with pytest.raises(PresetValidationError, match="无法识别"):
        controller._optimize_style_preview(b"not an image")


def test_style_crud_persists_both_prompts_and_managed_preview(tmp_path):
    controller, _, plugin = build_controller(tmp_path)
    preview_dir = tmp_path / "style_previews"
    preview_dir.mkdir()
    first_preview = "a" * 32 + ".webp"
    second_preview = "b" * 32 + ".webp"
    (preview_dir / first_preview).write_bytes(b"first")
    (preview_dir / second_preview).write_bytes(b"second")

    controller.create_style(
        {
            "name": "水彩",
            "positive_prompt": "watercolor, soft colors",
            "negative_prompt": "photorealistic, harsh shadows",
            "preview_image": first_preview,
        }
    )

    assert plugin.style_presets["水彩"] == {
        "positive_prompt": "watercolor, soft colors",
        "negative_prompt": "photorealistic, harsh shadows",
        "preview_image": first_preview,
    }
    plugin.selected_style_preset = "水彩"
    plugin.config["selected_style_preset"] = "水彩"
    plugin.session_drawing_settings["session-1"] = {
        "style_name": "水彩",
        "style_enabled": True,
    }

    controller.update_style(
        {
            "original_name": "水彩",
            "name": "透明水彩",
            "positive_prompt": "transparent watercolor",
            "negative_prompt": "oil painting",
            "preview_image": second_preview,
        }
    )

    assert "水彩" not in plugin.style_presets
    assert plugin.style_presets["透明水彩"]["negative_prompt"] == "oil painting"
    assert plugin.selected_style_preset == "透明水彩"
    assert plugin.config["selected_style_preset"] == "透明水彩"
    assert plugin.session_drawing_settings["session-1"]["style_name"] == "透明水彩"
    assert not (preview_dir / first_preview).exists()

    controller.delete_style({"name": "透明水彩"})

    assert "透明水彩" not in plugin.style_presets
    assert plugin.selected_style_preset == ""
    assert plugin.config["selected_style_preset"] == ""
    assert plugin.session_drawing_settings["session-1"] == {
        "style_name": "",
        "style_enabled": False,
    }
    assert not (preview_dir / second_preview).exists()

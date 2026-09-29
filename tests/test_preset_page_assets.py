import json
import re
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PAGE_ROOT = PLUGIN_ROOT / "pages" / "preset-manager"
PRESET_VIEW_ROOT = PAGE_ROOT / "presets"
STYLE_PAGE_ROOT = PAGE_ROOT / "style-gallery"


def test_management_shell_mounts_both_views_without_nested_iframes():
    html = (PAGE_ROOT / "index.html").read_text(encoding="utf-8")
    app_js = (PAGE_ROOT / "app.js").read_text(encoding="utf-8")
    declared_ids = set(re.findall(r'\bid="([^"]+)"', html))
    referenced_ids = set(re.findall(r'getElementById\("([^"]+)"\)', app_js))

    assert referenced_ids <= declared_ids
    assert (PAGE_ROOT / "styles.css").is_file()
    assert "http://" not in html
    assert "https://" not in html
    assert "<iframe" not in html
    assert 'loadView("presets", "./presets/index.html")' in app_js
    assert 'loadView("styles", "./style-gallery/index.html")' in app_js
    assert "namespacePanel" in app_js
    assert "initializePresetManager" in app_js
    assert "initializeStyleGallery" in app_js


def test_preset_view_has_all_referenced_dom_ids_and_local_assets():
    html = (PRESET_VIEW_ROOT / "index.html").read_text(encoding="utf-8")
    app_js = (PRESET_VIEW_ROOT / "app.js").read_text(encoding="utf-8")
    declared_ids = set(re.findall(r'\bid="([^"]+)"', html))
    referenced_ids = set(re.findall(r'byId\("([^"]+)"\)', app_js))

    assert referenced_ids <= declared_ids
    assert (PRESET_VIEW_ROOT / "styles.css").is_file()
    assert (PRESET_VIEW_ROOT / "api.js").is_file()
    assert "http://" not in html
    assert "https://" not in html


def test_raw_preset_config_fields_are_hidden_but_preserved():
    schema = json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))

    assert schema["presets"]["invisible"] is True
    assert schema["preset_categories"]["invisible"] is True
    assert schema["presets"]["default"] == "{}"
    assert schema["preset_categories"]["default"] == "{}"


def test_style_page_has_all_referenced_dom_ids_and_local_assets():
    html = (STYLE_PAGE_ROOT / "index.html").read_text(encoding="utf-8")
    app_js = (STYLE_PAGE_ROOT / "app.js").read_text(encoding="utf-8")
    declared_ids = set(re.findall(r'\bid="([^"]+)"', html))
    referenced_ids = set(re.findall(r'byId\("([^"]+)"\)', app_js))

    assert referenced_ids <= declared_ids
    assert (STYLE_PAGE_ROOT / "styles.css").is_file()
    assert (STYLE_PAGE_ROOT / "api.js").is_file()
    assert "http://" not in html
    assert "https://" not in html


def test_style_library_is_nested_in_primary_plugin_page():
    pages_root = PLUGIN_ROOT / "pages"
    discovered_page_names = sorted(
        path.name
        for path in pages_root.iterdir()
        if path.is_dir() and (path / "index.html").is_file()
    )

    assert discovered_page_names == ["preset-manager"]
    assert (PAGE_ROOT / "style-gallery" / "index.html").is_file()


def test_chat_style_gallery_template_only_exposes_names_and_previews():
    template = (PLUGIN_ROOT / "data" / "t2i_templates" / "style_gallery.html").read_text(
        encoding="utf-8"
    )

    assert "{{ item.name }}" in template
    assert "item.preview_data" in template
    assert "positive_prompt" not in template
    assert "negative_prompt" not in template
    assert "提示词" not in template


def test_chat_style_gallery_uses_local_playwright_renderer():
    main_source = (PLUGIN_ROOT / "main.py").read_text(encoding="utf-8")
    renderer_source = (PLUGIN_ROOT / "style_gallery_renderer.py").read_text(
        encoding="utf-8"
    )
    requirements = (PLUGIN_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "render_style_gallery_image(" in main_source
    assert "self.html_render(" not in main_source
    assert "async_playwright" in renderer_source
    assert "page.screenshot(" in renderer_source
    assert "playwright>=" in requirements


def test_style_config_is_hidden_and_standalone_negative_presets_are_removed():
    schema = json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))

    assert schema["style_presets"]["invisible"] is True
    assert "negative_prompts" not in schema
    assert "enable_negative_prompt" not in schema
    assert "selected_negative_prompt" not in schema

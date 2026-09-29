from types import SimpleNamespace

import pytest

from data.plugins.astrbot_plugin_novelai.main import CacheManager, NovelAIPlugin


class _Config(dict):
    def __init__(self):
        super().__init__()
        self.saved = 0

    def save_config(self):
        self.saved += 1


def make_plugin() -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin._initialize_safe_defaults()
    plugin.enable_style_preset = False
    plugin.selected_style_preset = ""
    plugin.config = _Config()
    plugin.config_manager = None
    plugin._check_whitelist = lambda _event: True
    plugin._is_user_admin = lambda _event: True
    return plugin


def test_cache_manager_honors_per_entry_ttl(monkeypatch):
    cache = CacheManager()
    now = 100.0
    monkeypatch.setattr("data.plugins.astrbot_plugin_novelai.main.time.time", lambda: now)

    cache.set("short", "value", ttl=10)
    cache.set("default", "value")
    assert cache.get("short") == "value"
    assert cache.get("default") == "value"

    now = 110.0
    assert cache.get("short", "expired") == "expired"
    assert cache.get("default") == "value"


def test_fallback_cache_honors_per_entry_ttl(monkeypatch):
    plugin = object.__new__(NovelAIPlugin)
    cache = plugin._create_fallback_cache()
    now = 100.0
    monkeypatch.setattr("data.plugins.astrbot_plugin_novelai.main.time.time", lambda: now)

    cache.set("short", "value", ttl=10)
    assert cache.get("short") == "value"
    now = 110.0
    assert cache.get("short", "expired") == "expired"


@pytest.mark.parametrize(
    ("message", "commands", "minimum", "maximum", "expected"),
    [
        ("/画图权重 7.5", ("画图权重", "NAI权重"), 1.0, 15.0, 7.5),
        ("图生图噪声 0", ("图生图噪声",), 0.0, 1.0, 0.0),
        ("/缩放 1", ("缩放",), 0.0, 1.0, 1.0),
    ],
)
def test_parse_session_float_argument_accepts_bounded_values(
    message, commands, minimum, maximum, expected
):
    plugin = make_plugin()
    event = SimpleNamespace(message_str=message)

    value_text, value, error = plugin._parse_session_float_argument(
        event,
        *commands,
        minimum=minimum,
        maximum=maximum,
    )

    assert value_text == message.split(maxsplit=1)[1]
    assert value == expected
    assert error is None


@pytest.mark.parametrize("message", ["/画图权重 nan", "/画图权重 inf", "/画图权重 -inf"])
def test_parse_session_float_argument_rejects_non_finite_values(message):
    plugin = make_plugin()
    event = SimpleNamespace(message_str=message)

    _, value, error = plugin._parse_session_float_argument(
        event,
        "画图权重",
        minimum=1.0,
        maximum=15.0,
    )

    assert value is None
    assert error == "range"


@pytest.mark.parametrize(
    ("message", "expected_error"),
    [
        ("/图生图强度 nope", "invalid"),
        ("/图生图强度 0.09", "range"),
        ("/图生图强度 0.91", "range"),
        ("/图生图强度", None),
    ],
)
def test_parse_session_float_argument_distinguishes_invalid_and_empty_values(
    message, expected_error
):
    plugin = make_plugin()
    event = SimpleNamespace(message_str=message)

    _, value, error = plugin._parse_session_float_argument(
        event,
        "图生图强度",
        minimum=0.1,
        maximum=0.9,
    )

    assert value is None
    assert error == expected_error


@pytest.mark.asyncio
async def test_session_float_command_keeps_command_specific_error_message():
    plugin = make_plugin()
    event = SimpleNamespace(
        message_str="/图生图强度 nan",
        plain_result=lambda text: text,
        unified_msg_origin="session-1",
    )

    results = [result async for result in plugin.set_img2img_strength(event)]

    assert results == ["❌ 强度值必须在0.1-0.9之间"]
    assert plugin.session_drawing_settings == {}


def test_update_audit_whitelist_persists_add_and_remove():
    plugin = make_plugin()

    added = plugin._update_audit_whitelist(
        attribute_name="audit_whitelist_groups",
        config_key="audit_whitelist_groups",
        value="123",
        noun="群",
        subject="该群",
        add=True,
    )
    duplicate = plugin._update_audit_whitelist(
        attribute_name="audit_whitelist_groups",
        config_key="audit_whitelist_groups",
        value="123",
        noun="群",
        subject="该群",
        add=True,
    )
    removed = plugin._update_audit_whitelist(
        attribute_name="audit_whitelist_groups",
        config_key="audit_whitelist_groups",
        value="123",
        noun="群",
        subject="该群",
        add=False,
    )

    assert "添加到审核白名单" in added
    assert duplicate == "❌ 群 123 已在审核白名单中"
    assert "从审核白名单中移除" in removed
    assert plugin.audit_whitelist_groups == []
    assert plugin.config["audit_whitelist_groups"] == []
    assert plugin.config.saved == 2


def test_update_bot_audit_whitelist_describes_audit_enablement():
    plugin = make_plugin()

    added = plugin._update_audit_whitelist(
        attribute_name="audit_whitelist_bots",
        config_key="audit_whitelist_bots",
        value="bot-a",
        noun="机器人",
        subject="该机器人",
        add=True,
        audit_on_add=True,
    )
    removed = plugin._update_audit_whitelist(
        attribute_name="audit_whitelist_bots",
        config_key="audit_whitelist_bots",
        value="bot-a",
        noun="机器人",
        subject="该机器人",
        add=False,
        audit_on_add=True,
    )

    assert "生成的图片将进入审核" in added
    assert "生成的图片将不再进入审核" in removed
    assert plugin.audit_whitelist_bots == []


def test_should_audit_image_requires_whitelisted_bot():
    plugin = make_plugin()
    bot_a = SimpleNamespace(get_self_id=lambda: "bot-a")
    bot_b = SimpleNamespace(get_self_id=lambda: "bot-b")

    assert plugin._should_audit_image(bot_a) is False

    plugin.audit_whitelist_bots = ["bot-a"]
    assert plugin._should_audit_image(bot_a) is True
    assert plugin._should_audit_image(bot_b) is False


def test_should_audit_image_keeps_group_exemption():
    plugin = make_plugin()
    plugin.audit_whitelist_bots = ["bot-a"]
    plugin.audit_whitelist_groups = ["group-a"]
    event = SimpleNamespace(
        get_self_id=lambda: "bot-a",
        get_group_id=lambda: "group-a",
    )

    assert plugin._should_audit_image(event) is False


def test_metadata_caption_extraction_supports_v4_and_legacy_fallbacks():
    plugin = make_plugin()

    assert plugin._extract_prompt_content(
        {
            "v4_prompt": {"caption": {"base_caption": "v4 prompt"}},
            "prompt": "legacy prompt",
        }
    ) == "v4 prompt"
    assert plugin._extract_negative_content(
        {
            "v4_negative_prompt": {"caption": {"base_caption": "v4 uc"}},
            "uc": "legacy uc",
        }
    ) == "v4 uc"
    assert plugin._extract_prompt_content({"prompt": "legacy prompt"}) == (
        "legacy prompt"
    )
    assert plugin._extract_negative_content({"uc": "legacy uc"}) == "legacy uc"

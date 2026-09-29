from types import SimpleNamespace

import pytest

from astrbot.core.provider.register import llm_tools
from data.plugins.astrbot_plugin_novelai.main import NovelAIPlugin


class _ToolEvent:
    def __init__(self, message_str: str = "请画一张图") -> None:
        self.message_str = message_str
        self.stopped = False

    def stop_event(self) -> None:
        self.stopped = True

    def plain_result(self, text: str):
        return SimpleNamespace(text=text)


def _make_plugin() -> NovelAIPlugin:
    plugin = NovelAIPlugin.__new__(NovelAIPlugin)
    plugin._check_whitelist = lambda _event: True
    return plugin


def test_generate_image_tool_is_registered_with_structured_skill_fields():
    tool = llm_tools.get_func("generate_image")

    assert tool is not None
    assert tool.parameters["properties"] == {
        "prompt": {
            "type": "string",
            "description": "用户提供或确认后的完整正向提示词，不要添加画师、角色、质量词、风格、动作、场景细节或其它未提供内容。",
        },
        "character_prompts": {
            "type": "array",
            "description": "仅填写用户明确提供或确认的独立 Character Prompt；按顺序传入，不要自行拆分或补写角色信息。",
            "items": {"type": "string"},
        },
        "negative_prompt": {
            "type": "string",
            "description": "仅传入用户明确提供或确认的 UC 排除项；没有就传空字符串，不要复制默认预设。",
        },
    }


def test_build_generate_image_tool_command_maps_skill_fields_to_draw_syntax():
    plugin = _make_plugin()

    command = plugin._build_generate_image_tool_command(
        "Prompt: 2girls, looking at another, cafe",
        [
            "Character 1: girl, black hair, maid",
            "Character 2: girl, blonde hair, casual clothes",
        ],
        "UC: nsfw, watermark",
    )

    assert command == (
        "/画图 2girls, looking at another, cafe"
        " | girl, black hair, maid"
        " | girl, blonde hair, casual clothes"
        " --uc nsfw, watermark"
    )


@pytest.mark.parametrize(
    "default_uc",
    ["", "用默认预设", "UC: 默认预设", "use default preset"],
)
def test_build_generate_image_tool_command_does_not_copy_default_uc(default_uc):
    plugin = _make_plugin()

    command = plugin._build_generate_image_tool_command(
        "1girl, solo, standing",
        [],
        default_uc,
    )

    assert command == "/画图 1girl, solo, standing"


@pytest.mark.asyncio
async def test_generate_image_tool_reuses_draw_pipeline_and_restores_event_message():
    plugin = _make_plugin()
    event = _ToolEvent("原始自然语言消息")
    captured = {}
    expected_result = SimpleNamespace(kind="image")

    async def fake_handle_text_to_image(tool_event):
        captured["message_str"] = tool_event.message_str
        yield expected_result

    plugin._handle_text_to_image = fake_handle_text_to_image

    results = [
        result
        async for result in plugin.generate_image(
            event,
            "1girl, solo, from below, standing",
            ["girl, silver hair, long hair, blue eyes, white dress"],
            "",
        )
    ]

    assert results == [expected_result]
    assert captured["message_str"] == (
        "/画图 1girl, solo, from below, standing"
        " | girl, silver hair, long hair, blue eyes, white dress"
    )
    assert event.message_str == "原始自然语言消息"
    assert event.stopped is True


@pytest.mark.asyncio
async def test_generate_image_tool_rejects_empty_prompt_without_running_pipeline():
    plugin = _make_plugin()
    event = _ToolEvent()

    async def unexpected_handler(_event):
        raise AssertionError("draw pipeline must not run")
        yield

    plugin._handle_text_to_image = unexpected_handler

    results = [
        result async for result in plugin.generate_image(event, "  ", [], "")
    ]

    assert len(results) == 1
    assert results[0].text == "❌ 无法调用画图工具: prompt 不能为空"
    assert event.stopped is True

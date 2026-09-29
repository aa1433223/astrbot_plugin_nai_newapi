import asyncio
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from data.plugins.astrbot_plugin_novelai.main import NovelAIPlugin
from data.plugins.astrbot_plugin_novelai.style_gallery_renderer import (
    render_drawing_help_image,
)


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELP_TEMPLATE = PLUGIN_ROOT / "data" / "t2i_templates" / "drawing_help.html"


def test_drawing_help_template_covers_current_user_workflows():
    template = HELP_TEMPLATE.read_text(encoding="utf-8")

    for expected in (
        "/画图",
        "/图生图",
        "/roll图",
        "/重画",
        "/预设列表",
        "/画风列表",
        "/参数设置",
        "/画图配置",
        "/重置画图参数",
        "/roll次数",
        "--uc",
    ):
        assert expected in template

    assert "very aesthetic" not in template
    assert "masterpiece" not in template
    assert "提示词规则" not in template


def test_help_renderer_outputs_non_empty_png(tmp_path):
    output_path = asyncio.run(
        render_drawing_help_image(
            HELP_TEMPLATE,
            {
                "model": "nai-diffusion-5-full",
                "resolution": "832x1216",
                "steps": 28,
                "scale": 6.5,
                "sampler": "k_euler_ancestral",
                "credential_status": "已配置 Token",
                "preset_count": 4,
                "style_count": 4,
            },
            output_dir=tmp_path,
        )
    )

    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.width == 1200
        assert image.height > 1300


def test_drawing_help_command_falls_back_to_text_when_browser_fails():
    plugin = NovelAIPlugin.__new__(NovelAIPlugin)
    plugin._check_whitelist = lambda _event: True

    async def fail_render(_event):
        raise RuntimeError("browser unavailable")

    plugin._render_drawing_help = fail_render
    event = SimpleNamespace(
        message_str="/画图帮助",
        plain_result=lambda text: text,
        chain_result=lambda value: value,
    )

    async def collect_results():
        return [result async for result in plugin.drawing_help(event)]

    results = asyncio.run(collect_results())

    assert len(results) == 1
    assert "参数:" in results[0]
    assert "普通提示词按原样处理" not in results[0]

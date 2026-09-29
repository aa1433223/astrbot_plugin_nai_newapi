import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import astrbot.api.message_components as Comp
from data.plugins.astrbot_plugin_novelai.main import NovelAIPlugin


def make_plugin() -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin.image_proxy = ""
    return plugin


@pytest.mark.asyncio
async def test_metadata_image_loader_reads_local_path(tmp_path):
    image_bytes = b"local NovelAI PNG bytes"
    image_path = tmp_path / "original.png"
    image_path.write_bytes(image_bytes)

    result = await make_plugin()._download_image_with_client(str(image_path))

    assert result == image_bytes


@pytest.mark.asyncio
async def test_metadata_image_loader_reads_file_uri(tmp_path):
    image_bytes = b"file URI NovelAI PNG bytes"
    image_path = tmp_path / "original with spaces.png"
    image_path.write_bytes(image_bytes)

    result = await make_plugin()._download_image_with_client(image_path.as_uri())

    assert result == image_bytes


@pytest.mark.asyncio
async def test_metadata_image_loader_decodes_astrbot_base64_reference():
    image_bytes = b"base64 NovelAI PNG bytes"
    image_ref = "base64://" + base64.b64encode(image_bytes).decode("ascii")

    result = await make_plugin()._download_image_with_client(image_ref)

    assert result == image_bytes


@pytest.mark.asyncio
async def test_metadata_image_loader_decodes_data_uri():
    image_bytes = b"data URI NovelAI PNG bytes"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    image_ref = f"data:image/png;base64,{encoded}"

    result = await make_plugin()._download_image_with_client(image_ref)

    assert result == image_bytes


def test_metadata_sources_include_png_file_attachments():
    plugin = make_plugin()
    components = [
        Comp.Image.fromURL("https://example.com/preview.jpg"),
        Comp.File(
            name="NovelAI_original.png",
            url="https://example.com/NovelAI_original.png",
        ),
    ]

    image_only = plugin.extract_images_from_messages(components)
    with_files = plugin.extract_images_from_messages(components, include_files=True)

    assert image_only == ["https://example.com/preview.jpg"]
    assert with_files == [
        "https://example.com/NovelAI_original.png",
        "https://example.com/preview.jpg",
    ]


def test_metadata_sources_prefer_local_file_attachment_over_url(tmp_path):
    plugin = make_plugin()
    local_path = tmp_path / "original.png"
    local_path.write_bytes(b"png")
    components = [
        Comp.File(
            name="original.png",
            file=str(local_path),
            url="https://example.com/preview.jpg",
        )
    ]

    assert plugin.extract_images_from_messages(components, include_files=True) == [
        str(local_path),
        "https://example.com/preview.jpg",
    ]


def test_metadata_sources_include_replied_png_file():
    plugin = make_plugin()
    reply = Comp.Reply(
        id="1",
        chain=[Comp.File(name="original.png", file="/tmp/original.png")],
    )
    event = SimpleNamespace(message_obj=SimpleNamespace(message=[reply]))

    result = plugin.get_reply_message_images(event, include_files=True)

    assert result == ["/tmp/original.png"]


@pytest.mark.asyncio
async def test_metadata_command_uses_latest_generated_png_without_preview():
    plugin = make_plugin()
    plugin._check_whitelist = lambda _event: True

    original = io.BytesIO()
    png_info = PngInfo()
    png_info.add_text(
        "Comment",
        json.dumps({"prompt": "cached original", "steps": 28}, ensure_ascii=False),
    )
    Image.new("RGBA", (2, 2), "white").save(
        original, format="PNG", pnginfo=png_info
    )

    event = SimpleNamespace(
        unified_msg_origin="group:1",
        message_obj=SimpleNamespace(message=[]),
    )
    event.plain_result = lambda text: text
    event.chain_result = lambda chain: chain
    plugin._cache_generated_image(event, original.getvalue())

    results = [result async for result in plugin.extract_nai_metadata(event)]

    assert len(results) == 1
    forwarded_chain = results[0]
    forwarded_nodes = next(
        component for component in forwarded_chain if isinstance(component, Comp.Nodes)
    )
    forwarded_text = "\n".join(
        part.text
        for node in forwarded_nodes.nodes
        for part in node.content
        if isinstance(part, Comp.Plain)
    )
    assert "cached original" in forwarded_text


@pytest.mark.asyncio
async def test_metadata_command_falls_back_to_cached_png_for_quoted_jpeg_preview(
    tmp_path,
):
    plugin = make_plugin()
    plugin._check_whitelist = lambda _event: True

    preview_path = tmp_path / "quoted-preview.jpg"
    preview = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(preview, format="JPEG")
    preview_path.write_bytes(preview.getvalue())

    original = io.BytesIO()
    png_info = PngInfo()
    png_info.add_text(
        "Comment",
        json.dumps({"prompt": "quoted cached original", "seed": 456}),
    )
    Image.new("RGBA", (2, 2), "white").save(
        original, format="PNG", pnginfo=png_info
    )

    event = SimpleNamespace(
        unified_msg_origin="group:quoted-preview",
        message_obj=SimpleNamespace(
            message=[
                Comp.Reply(
                    id="42",
                    chain=[Comp.Image.fromFileSystem(str(preview_path))],
                )
            ]
        ),
    )
    event.plain_result = lambda text: text
    event.chain_result = lambda chain: chain
    plugin._cache_generated_image(event, original.getvalue())

    results = [result async for result in plugin.extract_nai_metadata(event)]

    assert len(results) == 1
    forwarded_chain = results[0]
    forwarded_nodes = next(
        component for component in forwarded_chain if isinstance(component, Comp.Nodes)
    )
    forwarded_text = "\n".join(
        part.text
        for node in forwarded_nodes.nodes
        for part in node.content
        if isinstance(part, Comp.Plain)
    )
    assert "quoted cached original" in forwarded_text


@pytest.mark.asyncio
async def test_metadata_command_fetches_quoted_image_when_reply_chain_is_empty(
    tmp_path, monkeypatch
):
    plugin = make_plugin()
    plugin._check_whitelist = lambda _event: True

    original = io.BytesIO()
    png_info = PngInfo()
    png_info.add_text(
        "Comment",
        json.dumps({"prompt": "remotely fetched original"}),
    )
    Image.new("RGBA", (2, 2), "white").save(
        original, format="PNG", pnginfo=png_info
    )
    original_path = tmp_path / "remote-original.png"
    original_path.write_bytes(original.getvalue())

    async def fetch_quoted_images(_event):
        return [str(original_path)]

    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.extract_quoted_message_images",
        fetch_quoted_images,
    )

    event = SimpleNamespace(
        message_obj=SimpleNamespace(
            message=[Comp.Reply(id="message-id", chain=[])],
        )
    )
    event.plain_result = lambda text: text
    event.chain_result = lambda chain: chain

    results = [result async for result in plugin.extract_nai_metadata(event)]

    assert len(results) == 1
    forwarded_chain = results[0]
    forwarded_nodes = next(
        component for component in forwarded_chain if isinstance(component, Comp.Nodes)
    )
    forwarded_text = "\n".join(
        part.text
        for node in forwarded_nodes.nodes
        for part in node.content
        if isinstance(part, Comp.Plain)
    )
    assert "remotely fetched original" in forwarded_text


@pytest.mark.asyncio
async def test_auto_save_preserves_exact_png_bytes(tmp_path):
    plugin = make_plugin()
    plugin.enable_auto_save = True
    plugin.save_path = str(tmp_path)
    png_bytes = b"\x89PNG\r\n\x1a\nuntouched payload"

    saved_path = await plugin._save_image_locally(png_bytes, "txt2img")

    assert saved_path is not None
    assert saved_path.endswith(".png")
    assert Path(saved_path).read_bytes() == png_bytes


def test_metadata_extractor_reads_standard_png_json_metadata():
    plugin = make_plugin()
    output = io.BytesIO()
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
    png_info = PngInfo()
    png_info.add_text(
        "Comment",
        json.dumps({"prompt": "1girl", "steps": 28}, ensure_ascii=False),
    )
    image.save(output, format="PNG", pnginfo=png_info)

    with Image.open(io.BytesIO(output.getvalue())) as loaded:
        assert plugin._extract_novelai_metadata(loaded) == {
            "Comment": {"prompt": "1girl", "steps": 28},
        }


@pytest.mark.asyncio
async def test_metadata_command_tries_original_after_compressed_preview(tmp_path):
    plugin = make_plugin()
    plugin._check_whitelist = lambda _event: True

    preview = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(preview, format="JPEG")
    original = io.BytesIO()
    png_info = PngInfo()
    png_info.add_text(
        "Comment",
        json.dumps({"prompt": "1girl"}, ensure_ascii=False),
    )
    Image.new("RGBA", (2, 2), "white").save(
        original, format="PNG", pnginfo=png_info
    )
    preview_path = tmp_path / "preview.jpg"
    original_path = tmp_path / "original.png"
    preview_path.write_bytes(preview.getvalue())
    original_path.write_bytes(original.getvalue())

    event = SimpleNamespace(
        message_obj=SimpleNamespace(
            message=[
                Comp.Image.fromFileSystem(str(preview_path)),
                Comp.File(name="original.png", file=str(original_path)),
            ]
        )
    )
    event.plain_result = lambda text: text
    event.chain_result = lambda chain: chain

    results = [result async for result in plugin.extract_nai_metadata(event)]

    assert len(results) == 1
    assert not any("只支持PNG" in str(result) for result in results)
    forwarded_chain = results[0]
    forwarded_nodes = next(
        component for component in forwarded_chain if isinstance(component, Comp.Nodes)
    )
    forwarded_text = "\n".join(
        part.text
        for node in forwarded_nodes.nodes
        for part in node.content
        if isinstance(part, Comp.Plain)
    )
    assert "1girl" in forwarded_text


@pytest.mark.asyncio
async def test_metadata_command_reads_external_png_without_generated_cache(tmp_path):
    plugin = make_plugin()
    plugin._check_whitelist = lambda _event: True

    original = io.BytesIO()
    png_info = PngInfo()
    png_info.add_text(
        "Comment",
        json.dumps({"prompt": "external original", "seed": 123}, ensure_ascii=False),
    )
    Image.new("RGBA", (2, 2), "white").save(
        original, format="PNG", pnginfo=png_info
    )
    original_path = tmp_path / "external-original.png"
    original_path.write_bytes(original.getvalue())

    event = SimpleNamespace(
        unified_msg_origin="group:external",
        message_obj=SimpleNamespace(
            message=[Comp.File(name="external-original.png", file=str(original_path))]
        ),
    )
    event.plain_result = lambda text: text
    event.chain_result = lambda chain: chain

    results = [result async for result in plugin.extract_nai_metadata(event)]

    assert len(results) == 1
    forwarded_chain = results[0]
    forwarded_nodes = next(
        component for component in forwarded_chain if isinstance(component, Comp.Nodes)
    )
    forwarded_text = "\n".join(
        part.text
        for node in forwarded_nodes.nodes
        for part in node.content
        if isinstance(part, Comp.Plain)
    )
    assert "external original" in forwarded_text

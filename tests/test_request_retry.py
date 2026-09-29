import asyncio

import httpx
import pytest

from data.plugins.astrbot_plugin_novelai.main import Constants, NovelAIPlugin


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text or f"status {status_code}"


class _FakeAsyncClient:
    """Minimal ``httpx.AsyncClient`` stand-in that replays a scripted response list."""

    script: list = []
    attempts = 0

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None):
        step = _FakeAsyncClient.script[_FakeAsyncClient.attempts]
        _FakeAsyncClient.attempts += 1
        if isinstance(step, Exception):
            raise step
        return step


def _make_plugin(**overrides) -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin.default_model = Constants.DEFAULT_MODEL
    plugin.default_sampler = Constants.DEFAULT_SAMPLER
    plugin.enable_request_retry = overrides.get("enable_request_retry", True)
    plugin.request_retry_count = overrides.get("request_retry_count", 5)
    plugin.request_retry_max_delay = overrides.get("request_retry_max_delay", 0.0)
    plugin._validate_common_api_params = lambda *args, **kwargs: None

    async def _fake_client_and_headers(*args, **kwargs):
        return {}, {"Authorization": "Bearer test-token"}

    plugin._create_http_client_and_headers = _fake_client_and_headers
    plugin.get_model_parameters = lambda *args, **kwargs: {}
    plugin.uses_v4_prompt_structure = lambda model_name: False
    plugin.get_model_version_label = lambda model_name: "test"
    plugin.get_api_url = lambda *args, **kwargs: "http://test.local/generate-image"
    return plugin


def _install_client(monkeypatch, script: list) -> None:
    _FakeAsyncClient.script = list(script)
    _FakeAsyncClient.attempts = 0
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)


def _img2img_kwargs() -> dict:
    return {
        "prompt": "draw",
        "reference_image": "base64-data",
        "strength": 0.5,
        "noise": 0.2,
    }


@pytest.mark.asyncio
async def test_txt2img_retries_server_error_then_succeeds(monkeypatch):
    plugin = _make_plugin()
    _install_client(monkeypatch, [_FakeResponse(500), _FakeResponse(200, b"image")])

    image_data, meta_info = await plugin._call_novelai_txt2img(prompt="draw")

    assert image_data == b"image"
    assert meta_info == {"success": True, "retries": 1}
    assert _FakeAsyncClient.attempts == 2


@pytest.mark.asyncio
async def test_txt2img_retries_network_error_then_succeeds(monkeypatch):
    plugin = _make_plugin()
    _install_client(
        monkeypatch,
        [httpx.ConnectError("connection refused"), _FakeResponse(200, b"image")],
    )

    image_data, _meta_info = await plugin._call_novelai_txt2img(prompt="draw")

    assert image_data == b"image"
    assert _FakeAsyncClient.attempts == 2


@pytest.mark.parametrize("status_code", [400, 403, 404, 429, 500, 503])
@pytest.mark.asyncio
async def test_txt2img_retries_every_error_status(monkeypatch, status_code):
    plugin = _make_plugin()
    _install_client(
        monkeypatch, [_FakeResponse(status_code), _FakeResponse(200, b"image")]
    )

    image_data, meta_info = await plugin._call_novelai_txt2img(prompt="draw")

    assert image_data == b"image"
    assert meta_info == {"success": True, "retries": 1}
    assert _FakeAsyncClient.attempts == 2


@pytest.mark.asyncio
async def test_txt2img_reports_error_after_exhausting_default_retries(monkeypatch):
    plugin = _make_plugin()
    _install_client(monkeypatch, [_FakeResponse(503) for _ in range(6)])

    image_data, meta_info = await plugin._call_novelai_txt2img(prompt="draw")

    assert image_data is None
    assert meta_info["error_type"] == "server"
    # 首次请求 + 默认 5 次重试
    assert _FakeAsyncClient.attempts == 6


@pytest.mark.asyncio
async def test_txt2img_auth_error_does_not_retry(monkeypatch):
    plugin = _make_plugin(request_retry_count=1)
    _install_client(monkeypatch, [_FakeResponse(401), _FakeResponse(401)])

    image_data, meta_info = await plugin._call_novelai_txt2img(prompt="draw")

    assert image_data is None
    assert meta_info["error_type"] == "auth"
    assert _FakeAsyncClient.attempts == 1


@pytest.mark.asyncio
async def test_txt2img_retry_can_be_disabled(monkeypatch):
    plugin = _make_plugin(enable_request_retry=False)
    _install_client(monkeypatch, [_FakeResponse(500), _FakeResponse(200, b"image")])

    image_data, meta_info = await plugin._call_novelai_txt2img(prompt="draw")

    assert image_data is None
    assert meta_info["error_type"] == "server"
    assert _FakeAsyncClient.attempts == 1


@pytest.mark.asyncio
async def test_retry_count_zero_fails_immediately(monkeypatch):
    plugin = _make_plugin(request_retry_count=0)
    _install_client(monkeypatch, [_FakeResponse(503), _FakeResponse(200, b"image")])

    image_data, meta_info = await plugin._call_novelai_txt2img(prompt="draw")

    assert image_data is None
    assert meta_info["error_type"] == "server"
    assert _FakeAsyncClient.attempts == 1


@pytest.mark.asyncio
async def test_retry_delay_backs_off_exponentially_within_the_configured_cap(
    monkeypatch,
):
    plugin = _make_plugin(request_retry_count=3, request_retry_max_delay=3.0)
    _install_client(
        monkeypatch,
        [
            _FakeResponse(500),
            _FakeResponse(500),
            _FakeResponse(500),
            _FakeResponse(200, b"image"),
        ],
    )
    delays: list[float] = []

    async def _capture_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _capture_sleep)

    image_data, _meta_info = await plugin._call_novelai_txt2img(prompt="draw")

    assert image_data == b"image"
    assert len(delays) == 3
    assert all(0 < delay <= 3.0 for delay in delays)


@pytest.mark.asyncio
async def test_img2img_retries_server_error_then_succeeds(monkeypatch):
    plugin = _make_plugin()
    _install_client(monkeypatch, [_FakeResponse(502), _FakeResponse(200, b"image")])

    image_data, meta_info = await plugin._call_novelai_img2img(**_img2img_kwargs())

    assert image_data == b"image"
    assert meta_info == {"success": True, "retries": 1}
    assert _FakeAsyncClient.attempts == 2


@pytest.mark.asyncio
async def test_img2img_retries_quota_error_then_succeeds(monkeypatch):
    plugin = _make_plugin()
    _install_client(monkeypatch, [_FakeResponse(429), _FakeResponse(200, b"image")])

    image_data, _meta_info = await plugin._call_novelai_img2img(**_img2img_kwargs())

    assert image_data == b"image"
    assert _FakeAsyncClient.attempts == 2


@pytest.mark.asyncio
async def test_img2img_reports_error_after_exhausting_default_retries(monkeypatch):
    plugin = _make_plugin()
    _install_client(monkeypatch, [_FakeResponse(500) for _ in range(6)])

    image_data, meta_info = await plugin._call_novelai_img2img(**_img2img_kwargs())

    assert image_data is None
    assert meta_info["error_type"] == "server"
    assert _FakeAsyncClient.attempts == 6

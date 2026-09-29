from types import SimpleNamespace

import httpx
import pytest

from data.plugins.astrbot_plugin_novelai.main import Constants, NovelAIPlugin


def make_plugin(**overrides) -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin._initialize_safe_defaults()
    plugin._check_whitelist = lambda _event: True
    plugin._check_admin_permission = lambda _event, _name: True
    for key, value in overrides.items():
        setattr(plugin, key, value)
    return plugin


def _make_client(payload, calls, raises=None):
    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class _Client:
        def __init__(self, **kwargs):
            calls.setdefault("configs", []).append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, url, headers=None):
            calls["url"] = url
            calls["headers"] = headers
            if raises is not None:
                raise raises
            return _Response()

    return _Client


def test_newapi_base_url_strips_known_endpoint_suffixes():
    assert (
        NovelAIPlugin._newapi_base_url("http://localhost:3000/")
        == "http://localhost:3000"
    )
    assert (
        NovelAIPlugin._newapi_base_url("https://api.example.com/v1")
        == "https://api.example.com"
    )
    assert (
        NovelAIPlugin._newapi_base_url("https://api.example.com/v1/images/generations")
        == "https://api.example.com"
    )


def test_format_usd_uses_newapi_quota_scale():
    assert NovelAIPlugin._format_usd(Constants.NEWAPI_QUOTA_PER_USD) == "$1.0000"
    assert NovelAIPlugin._format_usd(0) == "$0.0000"
    assert NovelAIPlugin._format_usd(None) == "$0.0000"
    assert NovelAIPlugin._format_usd("bad") == "$0.0000"


@pytest.mark.asyncio
async def test_fetch_newapi_balance_targets_user_self(monkeypatch):
    plugin = make_plugin()
    calls = {}
    payload = {
        "data": {
            "username": "alice",
            "quota": 1000000,
            "used_quota": 250000,
            "request_count": 7,
        }
    }
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.httpx.AsyncClient",
        _make_client(payload, calls),
    )

    data = await plugin._fetch_newapi_balance("http://localhost:3000/", "sk-demo")

    assert data["quota"] == 1000000
    assert calls["url"] == "http://localhost:3000/api/user/self"
    assert calls["headers"]["Authorization"] == "Bearer sk-demo"
    assert calls["configs"][0]["timeout"] == Constants.NEWAPI_BALANCE_TIMEOUT


@pytest.mark.asyncio
async def test_fetch_newapi_balance_rejects_unexpected_shape(monkeypatch):
    plugin = make_plugin()
    calls = {}
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.httpx.AsyncClient",
        _make_client({"data": "not-a-dict"}, calls),
    )

    with pytest.raises(ValueError):
        await plugin._fetch_newapi_balance("http://localhost:3000", "sk-demo")


@pytest.mark.asyncio
async def test_fetch_newapi_daily_usage_is_best_effort(monkeypatch):
    plugin = make_plugin()
    calls = {}
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.httpx.AsyncClient",
        _make_client({}, calls, raises=httpx.ConnectError("boom")),
    )

    assert await plugin._fetch_newapi_daily_usage("http://localhost:3000", "sk") is None


def test_format_newapi_balance_lines_renders_account_and_usd():
    plugin = make_plugin()
    text = plugin._format_newapi_balance_lines(
        "http://localhost:3000/",
        {
            "display_name": "@alice",
            "quota": 1500000,
            "used_quota": 500000,
            "request_count": "12",
        },
    )

    assert text.splitlines() == [
        "💰 New API 余额",
        "站点：http://localhost:3000",
        "账户：alice",
        "剩余额度：$3.0000",
        "历史消耗：$1.0000",
        "总请求数：12",
    ]


@pytest.mark.asyncio
async def test_show_newapi_balance_reports_unconfigured():
    plugin = make_plugin(newapi_base_url="", newapi_api_key="")
    event = SimpleNamespace(plain_result=lambda text: text)

    results = [result async for result in plugin.show_newapi_balance(event)]

    assert "未配置 New API" in results[0]


@pytest.mark.asyncio
async def test_show_newapi_balance_requires_admin():
    plugin = make_plugin(newapi_base_url="http://localhost:3000", newapi_api_key="sk")
    plugin._check_admin_permission = lambda _event, _name: False
    event = SimpleNamespace(plain_result=lambda text: text)

    results = [result async for result in plugin.show_newapi_balance(event)]

    assert results == ["❌ 仅管理员可使用此命令"]


@pytest.mark.asyncio
async def test_show_newapi_balance_reports_remaining_and_recent_usage():
    plugin = make_plugin(
        newapi_base_url="http://localhost:3000", newapi_api_key="sk-demo"
    )

    async def fake_balance(base_url, api_key):
        return {
            "username": "alice",
            "quota": 1000000,
            "used_quota": 250000,
            "request_count": 7,
        }

    async def fake_recent(base_url, api_key):
        return (50000.0, 3)

    plugin._fetch_newapi_balance = fake_balance
    plugin._fetch_newapi_daily_usage = fake_recent
    event = SimpleNamespace(plain_result=lambda text: text)

    results = [result async for result in plugin.show_newapi_balance(event)]

    text = results[0]
    assert "💰 New API 余额" in text
    assert "剩余额度：$2.0000" in text
    assert "历史消耗：$0.5000" in text
    assert "总请求数：7" in text
    assert "近 24 小时消耗：$0.1000 · 请求 3 次" in text


@pytest.mark.asyncio
async def test_show_newapi_balance_surfaces_query_error():
    plugin = make_plugin(
        newapi_base_url="http://localhost:3000", newapi_api_key="sk-demo"
    )

    async def boom(base_url, api_key):
        raise httpx.ConnectError("boom")

    plugin._fetch_newapi_balance = boom
    event = SimpleNamespace(plain_result=lambda text: text)

    results = [result async for result in plugin.show_newapi_balance(event)]

    assert "❌ 查询 New API 余额失败：网络请求失败" in results[0]

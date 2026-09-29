from datetime import datetime
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


def test_collect_quota_accounts_lists_main_first_and_deduplicates():
    plugin = make_plugin(
        api_token="pst-main",
        account_tokens=[
            {"name": "小号A", "api_token": "pst-a", "enable": True},
            {"name": "小号B", "api_token": "pst-b", "enable": False},
            {"name": "重复主账户", "api_token": "pst-main", "enable": True},
            {"name": "", "api_token": "pst-c"},
            {"name": "空Token", "api_token": "   "},
        ],
    )

    assert plugin._collect_quota_accounts() == [
        (Constants.ACCOUNT_QUOTA_MAIN_LABEL, "pst-main"),
        ("小号A", "pst-a"),
        ("账户3", "pst-c"),
    ]


def test_collect_quota_accounts_ignores_malformed_entries():
    plugin = make_plugin(api_token="", account_tokens="pst-x")

    assert plugin._collect_quota_accounts() == []

    plugin.account_tokens = ["pst-x", None, {"api_token": "pst-y", "enable": "false"}]

    assert plugin._collect_quota_accounts() == []

    plugin.account_tokens = [{"name": "备用", "api_token": "pst-y", "enable": "true"}]

    assert plugin._collect_quota_accounts() == [("备用", "pst-y")]


def test_normalize_account_tokens_copies_dictionary_entries():
    entries = [{"name": "A", "api_token": "pst-a"}]

    normalized = NovelAIPlugin._normalize_account_tokens(entries)

    assert normalized == entries
    assert normalized[0] is not entries[0]
    assert NovelAIPlugin._normalize_account_tokens(None) == []
    assert NovelAIPlugin._normalize_account_tokens("pst-a") == []


@pytest.mark.asyncio
async def test_fetch_account_quota_targets_image_host_with_persistent_token(monkeypatch):
    plugin = make_plugin()
    calls = {}

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"subscription": {"tier": 3}}

    class _Client:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, url, headers=None):
            calls["url"] = url
            calls["headers"] = headers
            return _Response()

    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.httpx.AsyncClient", _Client
    )

    payload = await plugin._fetch_account_quota("pst-demo")

    assert payload == {"subscription": {"tier": 3}}
    assert calls["url"] == "https://image.novelai.net/user/data"
    assert calls["headers"]["Authorization"] == "Bearer pst-demo"
    assert calls["headers"]["x-correlation-id"]
    assert calls["config"]["timeout"] == Constants.ACCOUNT_QUOTA_TIMEOUT


def test_format_account_quota_lines_summarizes_subscription(monkeypatch):
    plugin = make_plugin()
    expires_at = 1789926023
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.time.time",
        lambda: float(expires_at - 4 * 86400 - 3600),
    )

    lines = plugin._format_account_quota_lines(
        1,
        "主账户",
        {
            "subscription": {
                "tier": 3,
                "active": True,
                "expiresAt": expires_at,
                "usage": {
                    "percent": 13,
                    "isNegative": False,
                    "timeUntilNextPercent": 7888,
                },
                "trainingStepsLeft": {
                    "fixedTrainingStepsLeft": 10,
                    "purchasedTrainingSteps": 3,
                },
            },
        },
    )

    expected_expiry = datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d")
    assert lines[0] == "1. 主账户 · Opus"
    # 官网前端用 round(17.3 * percent) 把剩余百分比换算成可生成图片数。
    assert "剩余额度 13%（约 225 张）" in lines[1]
    assert "Anlas 13（订阅 10 + 自购 3）" in lines[1]
    assert "每天恢复 11.0%" in lines[1]
    assert f"到期 {expected_expiry}（剩 4 天）" in lines[1]


def test_format_account_quota_lines_marks_expired_and_grace_period(monkeypatch):
    plugin = make_plugin()
    expires_at = 1789926023
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.time.time",
        lambda: float(expires_at + 3600),
    )

    lines = plugin._format_account_quota_lines(
        1,
        "旧订阅",
        {
            "subscription": {
                "tier": 3,
                "active": True,
                "expiresAt": expires_at,
                "isGracePeriod": True,
            }
        },
    )

    expected_expiry = datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d")
    assert f"已于 {expected_expiry} 到期" in lines[1]
    assert "宽限期" in lines[1]


def test_format_account_quota_lines_marks_inactive_and_overspent_accounts():
    plugin = make_plugin()

    lines = plugin._format_account_quota_lines(
        2,
        "备用",
        {
            "subscription": {
                "tier": 1,
                "active": False,
                "usage": {"percent": 0, "isNegative": True},
            },
        },
    )

    assert lines[0] == "2. 备用 · Tablet（未激活）"
    assert lines[1].strip() == "剩余额度 0%（已超额，出图改扣 Anlas）"


def test_format_account_quota_lines_handles_missing_details():
    plugin = make_plugin()

    assert plugin._format_account_quota_lines(1, "未知", {"subscription": {}}) == [
        "1. 未知 · 未知档位"
    ]


def test_format_account_quota_lines_reports_http_failure():
    plugin = make_plugin()
    request = httpx.Request("GET", "https://image.novelai.net/user/data")
    error = httpx.HTTPStatusError(
        "unauthorized", request=request, response=httpx.Response(401, request=request)
    )

    assert plugin._format_account_quota_lines(2, "小号A", error) == [
        "2. 小号A · 查询失败：Token 无效或不可用于额度查询（HTTP 401）"
    ]


@pytest.mark.parametrize(
    ("tier", "expected"),
    [(0, "无订阅"), (1, "Tablet"), (2, "Scroll"), (3, "Opus"), (9, "Tier 9")],
)
def test_format_subscription_tier_maps_known_plans(tier, expected):
    assert NovelAIPlugin._format_subscription_tier(tier) == expected

    assert NovelAIPlugin._format_subscription_tier(None) == "未知档位"


@pytest.mark.parametrize(
    ("training_steps", "expected"),
    [
        ({"fixedTrainingStepsLeft": 10, "purchasedTrainingSteps": 0}, "Anlas 10（订阅 10 + 自购 0）"),
        ({"fixedTrainingStepsLeft": 0, "purchasedTrainingSteps": 500}, "Anlas 500（订阅 0 + 自购 500）"),
        ({"fixedTrainingStepsLeft": 12.7}, "Anlas 12（订阅 12 + 自购 0）"),
        (42, "Anlas 42"),
        ({}, ""),
        (None, ""),
        ("unknown", ""),
    ],
)
def test_format_anlas_balance_reads_subscription_wallet(training_steps, expected):
    assert NovelAIPlugin._format_anlas_balance(training_steps) == expected


@pytest.mark.asyncio
async def test_show_account_quota_reports_every_configured_account():
    plugin = make_plugin(
        api_token="pst-main",
        account_tokens=[{"name": "小号A", "api_token": "pst-a", "enable": True}],
    )

    async def fake_fetch(token):
        if token == "pst-a":
            raise httpx.ConnectError("boom")
        return {
            "subscription": {"tier": 3, "active": True, "usage": {"percent": 13}},
        }

    plugin._fetch_account_quota = fake_fetch
    event = SimpleNamespace(plain_result=lambda text: text)

    results = [result async for result in plugin.show_account_quota(event)]

    assert results[0].splitlines()[0] == "📊 NovelAI 账户额度（2 个）"
    assert "1. 主账户 · Opus" in results[0]
    assert "剩余额度 13%" in results[0]
    assert "2. 小号A · 查询失败：网络请求失败" in results[0]


@pytest.mark.asyncio
async def test_show_account_quota_requires_a_configured_token():
    plugin = make_plugin()
    event = SimpleNamespace(plain_result=lambda text: text)

    results = [result async for result in plugin.show_account_quota(event)]

    assert "未配置任何 NovelAI Token" in results[0]


@pytest.mark.asyncio
async def test_show_account_quota_respects_admin_restriction():
    plugin = make_plugin(api_token="pst-main")
    plugin._check_admin_permission = lambda _event, _name: False
    event = SimpleNamespace(plain_result=lambda text: text)

    results = [result async for result in plugin.show_account_quota(event)]

    assert results == ["❌ 仅管理员可使用此命令"]


class _QuotaResponse:
    def __init__(self, payload=None, status_code=200, request=None):
        self._payload = payload or {"subscription": {"tier": 3}}
        self.status_code = status_code
        self._request = request or httpx.Request(
            "GET", "https://image.novelai.net/user/data"
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom", request=self._request, response=self
            )

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_fetch_account_quota_retries_transient_network_errors(monkeypatch):
    plugin = make_plugin()
    plugin.enable_request_retry = True
    plugin.request_retry_count = 5
    plugin.request_retry_max_delay = 0.0
    attempts = []

    class _FlakyClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, url, headers=None):
            attempts.append(url)
            if len(attempts) == 1:
                raise httpx.ConnectError("proxy hiccup")
            return _QuotaResponse()

    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.httpx.AsyncClient", _FlakyClient
    )

    payload = await plugin._fetch_account_quota("pst-demo")

    assert payload == {"subscription": {"tier": 3}}
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_fetch_account_quota_gives_up_after_max_attempts(monkeypatch):
    plugin = make_plugin()
    plugin.enable_request_retry = True
    plugin.request_retry_count = 9
    plugin.request_retry_max_delay = 0.0
    attempts = []

    class _BrokenClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, url, headers=None):
            attempts.append(url)
            raise httpx.ConnectTimeout("proxy timeout")

    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.httpx.AsyncClient", _BrokenClient
    )

    with pytest.raises(httpx.HTTPError):
        await plugin._fetch_account_quota("pst-demo")

    assert len(attempts) == Constants.ACCOUNT_QUOTA_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_fetch_account_quota_does_not_retry_rejected_tokens(monkeypatch):
    plugin = make_plugin()
    plugin.enable_request_retry = True
    plugin.request_retry_count = 5
    attempts = []

    class _RejectingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, url, headers=None):
            attempts.append(url)
            return _QuotaResponse(status_code=401)

    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_novelai.main.httpx.AsyncClient",
        _RejectingClient,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await plugin._fetch_account_quota("pst-demo")

    assert len(attempts) == 1

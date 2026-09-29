import io
from datetime import datetime, timedelta

from PIL import Image, ImageDraw

from data.plugins.astrbot_plugin_novelai.main import (
    Constants,
    NovelAIPlugin,
    UsageManager,
)
from data.plugins.astrbot_plugin_novelai.statistics_card import (
    CARD_HEIGHT,
    CARD_WIDTH,
    COLORS,
    render_statistics_card,
)


def _avatar_bytes(color: str) -> bytes:
    image = Image.new("RGB", (96, 96), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted(
        (luminance(foreground), luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_statistics_text_colors_meet_wcag_aa_contrast():
    for text_color in ("ink", "muted", "faint", "accent"):
        for surface_color in ("background", "surface", "surface_soft"):
            assert _contrast_ratio(COLORS[text_color], COLORS[surface_color]) >= 4.5


def test_global_statistics_contains_sorted_user_rankings():
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    manager = object.__new__(UsageManager)
    manager.usage_data = {
        "10001": {today: 4, yesterday: 8},
        "10002": {today: 6, yesterday: 1},
        "10003": {today: 0, yesterday: 20},
    }
    manager.queue = []
    manager.processing_request = None

    statistics = manager.get_global_statistics()

    assert statistics["today_total"] == 10
    assert statistics["total_usage"] == 39
    assert statistics["quota_scope"] is None
    assert statistics["daily_quota_limit"] == Constants.GROUP_DAILY_IMAGE_LIMIT
    assert statistics["daily_quota_used"] == 0
    assert statistics["daily_quota_remaining"] == Constants.GROUP_DAILY_IMAGE_LIMIT
    assert not statistics["fallback_active"]
    assert statistics["fallback_model"] == "nai-diffusion-4-5-full"
    assert statistics["fallback_usage_today"] == 0
    assert [item["user_id"] for item in statistics["user_rankings"]] == [
        "10003",
        "10001",
        "10002",
    ]
    assert statistics["user_rankings"][1]["today"] == 4
    assert [item["user_id"] for item in statistics["daily_user_rankings"]] == [
        "10002",
        "10001",
    ]


def test_group_daily_limit_counts_each_scope_separately_and_resets_by_date():
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    manager = object.__new__(UsageManager)
    manager.max_group_daily_usage = Constants.GROUP_DAILY_IMAGE_LIMIT
    manager.group_usage_data = {
        "group:10001": {today: 4, yesterday: 500},
        "group:10002": {today: 2},
    }
    manager.usage_data = {}
    manager.queue = []
    manager.processing_request = None

    assert manager.max_group_daily_usage == 5
    assert manager.get_group_daily_usage("group:10001") == 4
    assert manager.get_group_daily_remaining("group:10001") == 1
    assert manager.check_group_daily_limit("group:10001")
    assert not manager.check_group_daily_limit("group:10001", 2)

    # Another group still owns its own untouched quota.
    assert manager.get_group_daily_usage("group:10002") == 2
    assert manager.get_group_daily_remaining("group:10002") == 3
    assert manager.check_group_daily_limit("group:10002", 3)
    assert manager.get_group_daily_usage("group:99999") == 0

    manager.group_usage_data["group:10001"][today] = 5

    assert manager.get_group_daily_remaining("group:10001") == 0
    assert not manager.check_group_daily_limit("group:10001")

    statistics = manager.get_global_statistics("group:10001")

    assert statistics["quota_scope"] == "group:10001"
    assert statistics["daily_quota_used"] == 5
    assert statistics["daily_quota_remaining"] == 0
    assert statistics["fallback_active"]
    assert statistics["fallback_usage_today"] == 0

    manager.group_usage_data["group:10001"][today] = 6
    statistics = manager.get_global_statistics("group:10001")

    assert statistics["fallback_usage_today"] == 1


def test_group_daily_limit_accepts_configured_value():
    today = datetime.now().strftime("%Y-%m-%d")
    manager = object.__new__(UsageManager)
    manager.max_group_daily_usage = 12
    manager.group_usage_data = {"group:10001": {today: 11}}
    manager.usage_data = {}
    manager.queue = []
    manager.processing_request = None

    assert manager.get_group_daily_remaining("group:10001") == 1
    assert manager.check_group_daily_limit("group:10001")
    assert not manager.check_group_daily_limit("group:10001", 2)

    manager.max_group_daily_usage = 0

    assert manager.get_group_daily_remaining("group:10001") == 0
    assert not manager.check_group_daily_limit("group:10001")
    assert not manager.get_global_statistics("group:10001")["fallback_active"]


def test_daily_quota_routes_requests_to_v45_full_after_limit():
    today = datetime.now().strftime("%Y-%m-%d")
    manager = object.__new__(UsageManager)
    manager.max_group_daily_usage = Constants.GROUP_DAILY_IMAGE_LIMIT
    manager.group_usage_data = {"group:10001": {today: 4}}
    plugin = object.__new__(NovelAIPlugin)
    plugin.usage_manager = manager
    settings = {"model": "nai-diffusion-5-full"}

    assert not plugin._apply_daily_model_fallback(settings, "group:10001")
    assert settings["model"] == "nai-diffusion-5-full"

    manager.group_usage_data["group:10001"][today] = 5

    assert plugin._apply_daily_model_fallback(settings, "group:10001")
    assert settings["model"] == Constants.DAILY_FALLBACK_MODEL
    assert not plugin._apply_daily_model_fallback(settings, "group:10001")

    # An exhausted group does not degrade other groups.
    other_settings = {"model": "nai-diffusion-5-full"}
    assert not plugin._apply_daily_model_fallback(other_settings, "group:10002")
    assert other_settings["model"] == "nai-diffusion-5-full"


def test_statistics_card_renders_complete_dashboard_image():
    recent_days = [
        {
            "date": (datetime(2026, 8, 28) - timedelta(days=index)).strftime(
                "%Y-%m-%d"
            ),
            "count": count,
        }
        for index, count in enumerate([38, 52, 31, 64, 46, 27, 41])
    ]
    statistics = {
        "today_total": 38,
        "total_usage": 1248,
        "active_users_today": 17,
        "queue_length": 2,
        "processing_request": True,
        "recent_days": recent_days,
    }
    account = {
        "available": True,
        "subscription_anlas": 8377,
        "paid_anlas": 320,
        "total_anlas": 8697,
        "tier_name": "Opus",
        "usage": {
            "percent": 63,
            "isNegative": False,
            "timeUntilNextPercent": 7920,
        },
    }
    leaderboard = [
        {
            "user_id": str(10000 + index),
            "nickname": name,
            "avatar": _avatar_bytes(color),
            "today": today,
            "total": total,
        }
        for index, (name, color, today, total) in enumerate(
            [
                ("青岚", "#75978B", 8, 284),
                ("柚木", "#C9A77B", 5, 233),
                ("北岛", "#7D8EA3", 3, 197),
                ("阿槐", "#9A7F87", 2, 156),
                ("沉舟", "#A1A681", 1, 128),
            ]
        )
    ]

    png_bytes = render_statistics_card(
        statistics,
        account,
        leaderboard,
        generated_at=datetime(2026, 8, 28, 21, 30),
    )

    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.format == "PNG"
        assert image.size == (CARD_WIDTH, CARD_HEIGHT)
        assert image.mode == "RGB"


def test_statistics_card_shows_local_activity_pulse(monkeypatch):
    rendered_texts = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(draw, position, text, *args, **kwargs):
        rendered_texts.append(str(text))
        return original_text(draw, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)

    render_statistics_card(
        {
            "today_total": 4,
            "total_usage": 4,
            "active_users_today": 1,
            "queue_length": 0,
            "processing_request": False,
            "recent_days": [
                {"date": "2026-08-28", "count": 4},
                {"date": "2026-08-27", "count": 0},
            ],
        },
        {},
        [],
        generated_at=datetime(2026, 8, 28, 21, 30),
    )

    assert "本周节奏" in rendered_texts
    assert "日均出图" in rendered_texts
    assert "最高一天" in rendered_texts
    assert "活跃天数" in rendered_texts
    assert "账户额度" not in rendered_texts
    assert not any("Anlas" in text for text in rendered_texts)


def test_statistics_card_handles_empty_activity_and_ranking():
    png_bytes = render_statistics_card(
        {
            "today_total": 0,
            "total_usage": 0,
            "active_users_today": 0,
            "queue_length": 0,
            "processing_request": False,
            "recent_days": [],
        },
        {},
        [],
        generated_at=datetime(2026, 8, 28, 21, 30),
    )

    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.size == (CARD_WIDTH, CARD_HEIGHT)


def test_statistics_card_shows_daily_fallback_quota_data(monkeypatch):
    rendered_texts = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(draw, position, text, *args, **kwargs):
        rendered_texts.append(str(text))
        return original_text(draw, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)

    render_statistics_card(
        {
            "today_total": 107,
            "total_usage": 107,
            "active_users_today": 3,
            "daily_quota_limit": 100,
            "daily_quota_used": 100,
            "daily_quota_remaining": 0,
            "fallback_active": True,
            "fallback_model": "nai-diffusion-4-5-full",
            "fallback_usage_today": 7,
            "queue_length": 0,
            "processing_request": False,
            "recent_days": [],
        },
        {"available": False, "error": "额度读取失败"},
        [],
        generated_at=datetime(2026, 8, 28, 21, 30),
    )

    assert "每日免费额度" not in rendered_texts
    assert "100 / 100" not in rendered_texts
    assert "剩余 0 次" not in rendered_texts
    assert "已进入超额路由" not in rendered_texts
    assert "V4.5 Full" not in rendered_texts
    assert "已生效 · 超额出图 7 次" not in rendered_texts


def test_statistics_card_shows_daily_ranking(monkeypatch):
    rendered_texts = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(draw, position, text, *args, **kwargs):
        rendered_texts.append(str(text))
        return original_text(draw, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)

    render_statistics_card(
        {"recent_days": []},
        {"available": False},
        [{"user_id": "10001", "nickname": "总榜用户", "total": 20, "today": 1}],
        datetime(2026, 8, 28, 21, 30),
        [{"user_id": "10002", "nickname": "今日用户", "total": 3, "today": 3}],
    )

    assert "今日画图排行" in rendered_texts
    assert "今日用户" in rendered_texts
    assert "总榜用户" in rendered_texts


def test_statistics_card_renders_ten_rows_per_leaderboard(monkeypatch):
    rendered_texts = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(draw, position, text, *args, **kwargs):
        rendered_texts.append(str(text))
        return original_text(draw, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    users = [
        {"user_id": str(index), "nickname": f"用户{index}", "total": index, "today": index}
        for index in range(1, 11)
    ]

    render_statistics_card(
        {"recent_days": []},
        {"available": False},
        users,
        datetime(2026, 8, 28, 21, 30),
        users,
    )

    assert "用户10" in rendered_texts
    assert rendered_texts.count("10") >= 2


def test_statistics_card_does_not_render_removed_quota_content(monkeypatch):
    rendered_texts = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(draw, position, text, *args, **kwargs):
        rendered_texts.append(str(text))
        return original_text(draw, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)

    render_statistics_card(
        {
            "today_total": 0,
            "total_usage": 0,
            "active_users_today": 0,
            "queue_length": 0,
            "processing_request": False,
            "recent_days": [],
        },
        {
            "available": True,
            "subscription_anlas": 8377,
            "paid_anlas": 320,
            "total_anlas": 8697,
            "tier_name": "Opus",
            "usage": {"percent": 0, "isNegative": True},
        },
        [],
        generated_at=datetime(2026, 8, 28, 21, 30),
    )

    assert "本周节奏" in rendered_texts
    assert "账户额度" not in rendered_texts
    assert not any("Anlas" in text or "Opus" in text for text in rendered_texts)

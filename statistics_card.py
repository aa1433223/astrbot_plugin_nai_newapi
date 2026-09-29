"""Render the NovelAI usage summary as a single image card."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

CARD_WIDTH = 1200
CARD_HEIGHT = 1460
DEFAULT_DAILY_QUOTA = 100

COLORS = {
    "background": "#EAF1F4",
    "surface": "#F8FBFC",
    "surface_soft": "#DCE8EE",
    "ink": "#14232D",
    "muted": "#465B66",
    "faint": "#526B77",
    "line": "#C7D8DF",
    "accent": "#176B87",
    "accent_soft": "#DDEFF4",
    "accent_gradient_end": "#36A7A1",
    "gold": "#A66B00",
    "silver": "#526B77",
    "bronze": "#955018",
    "white": "#FFFFFF",
}

REGULAR_FONT_PATHS = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
)
BOLD_FONT_PATHS = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
)


def _font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a CJK-capable font available in common AstrBot deployments."""
    paths = BOLD_FONT_PATHS if bold else REGULAR_FONT_PATHS
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _truncate_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    """Shorten text to fit within the supplied width."""
    value = str(text or "")
    if draw.textlength(value, font=font) <= max_width:
        return value
    ellipsis = "…"
    while value and draw.textlength(value + ellipsis, font=font) > max_width:
        value = value[:-1]
    return value + ellipsis


def _avatar_image(avatar_bytes: bytes | None, size: int) -> Image.Image | None:
    """Decode and crop avatar bytes to a circle-ready square."""
    if not avatar_bytes:
        return None
    try:
        with Image.open(io.BytesIO(avatar_bytes)) as source:
            return ImageOps.fit(
                source.convert("RGB"),
                (size, size),
                method=Image.Resampling.LANCZOS,
            )
    except Exception:
        return None


def _draw_acrylic_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int = 24,
    border_color: str = "#FFFFFF",
    fill_color: str = "#FAF8FC",
) -> None:
    """Draw an iOS/ACG glassmorphism card with delicate border and inner soft fill."""
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill_color,
        outline=COLORS["line"],
        width=1,
    )
    # Inner specular highlight rim
    inner_box = (box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1)
    draw.rounded_rectangle(
        inner_box,
        radius=max(1, radius - 1),
        outline=border_color,
        width=1,
    )


def render_statistics_card(
    statistics: dict[str, Any],
    account: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    generated_at: datetime | None = None,
    daily_leaderboard: list[dict[str, Any]] | None = None,
) -> bytes:
    """Render local drawing activity and leaderboards into PNG bytes.

    ``account`` remains in the signature for compatibility with older callers;
    account quota is intentionally no longer fetched or displayed.
    """
    now = generated_at or datetime.now()

    # A quiet system-panel background with a subtle cool tint.
    canvas = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), COLORS["background"])
    bg_draw = ImageDraw.Draw(canvas)

    # Soft vertical tint keeps the translucent panels readable in chat.
    for y in range(CARD_HEIGHT):
        ratio = y / CARD_HEIGHT
        r = int(239 + (226 - 239) * ratio)
        g = int(246 + (237 - 246) * ratio)
        b = int(248 + (242 - 248) * ratio)
        bg_draw.line([(0, y), (CARD_WIDTH, y)], fill=(r, g, b))

    draw = ImageDraw.Draw(canvas)

    title_font = _font(48, bold=True)
    heading_font = _font(25, bold=True)
    summary_metric_font = _font(42, bold=True)
    metric_small_font = _font(28, bold=True)
    body_font = _font(20)
    body_bold_font = _font(20, bold=True)
    small_font = _font(16)
    small_bold_font = _font(16, bold=True)
    tiny_font = _font(14)
    tiny_bold_font = _font(14, bold=True)

    margin = 56

    # Header section
    badge_text = "NOVELAI / 使用概览"
    draw.text((margin, 38), badge_text, font=tiny_bold_font, fill=COLORS["accent"])
    draw.text((margin, 58), "画图统计", font=title_font, fill=COLORS["ink"])
    draw.text(
        (margin, 122),
        "最近 30 天的出图情况，今天也在稳定产出",
        font=small_font,
        fill=COLORS["muted"],
    )

    # Queue Status Pill (iOS dynamic island / status capsule style)
    queue_length = int(statistics.get("queue_length", 0) or 0)
    is_processing = bool(statistics.get("processing_request"))
    if is_processing:
        status_text = f"正在出图 · 前面还有 {queue_length} 位" if queue_length else "正在出图"
        pill_fill = COLORS["accent_soft"]
        pill_border = COLORS["accent"]
    else:
        status_text = "现在空闲"
        pill_fill = COLORS["surface"]
        pill_border = COLORS["line"]
    
    status_width = int(draw.textlength(status_text, font=small_bold_font)) + 36
    status_box = (
        CARD_WIDTH - margin - status_width,
        64,
        CARD_WIDTH - margin,
        108,
    )
    draw.rounded_rectangle(
        status_box,
        radius=22,
        fill=pill_fill,
        outline=pill_border,
        width=1,
    )
    draw.text(
        (status_box[0] + 18, status_box[1] + 12),
        status_text,
        font=small_bold_font,
        fill=COLORS["accent"],
    )

    # Top Panels: Left = Overview + Trend, Right = Local activity pulse
    overview_box = (margin, 160, 716, 560)
    activity_box = (738, 160, CARD_WIDTH - margin, 560)
    
    _draw_acrylic_card(draw, overview_box, radius=24, border_color="#FFFFFF", fill_color=COLORS["surface"])
    _draw_acrylic_card(draw, activity_box, radius=24, border_color="#FFFFFF", fill_color=COLORS["surface"])

    # Left Panel: Metrics Row
    summary_items = (
        ("今日产出", int(statistics.get("today_total", 0) or 0), "张"),
        ("近30天", int(statistics.get("total_usage", 0) or 0), "张"),
        ("活跃画师", int(statistics.get("active_users_today", 0) or 0), "位"),
    )
    summary_x_positions = (90, 290, 490)
    for (label, value, unit), x in zip(summary_items, summary_x_positions):
        draw.text((x, 186), label, font=small_font, fill=COLORS["muted"])
        value_text = str(value)
        draw.text(
            (x - 2, 218),
            value_text,
            font=summary_metric_font,
            fill=COLORS["ink"],
        )
        value_width = draw.textlength(value_text, font=summary_metric_font)
        draw.text(
            (x + value_width + 4, 240),
            unit,
            font=small_bold_font,
            fill=COLORS["accent"],
        )

    draw.line((266, 192, 266, 275), fill=COLORS["line"], width=1)
    draw.line((466, 192, 466, 275), fill=COLORS["line"], width=1)
    draw.line((86, 298, 686, 298), fill=COLORS["line"], width=1)

    # Trend Chart: 最近7天
    draw.text((90, 318), "最近 7 天产出走势", font=small_bold_font, fill=COLORS["ink"])

    recent_days = list(reversed(statistics.get("recent_days", [])[:7]))
    max_count = max([int(item.get("count", 0) or 0) for item in recent_days] or [1])
    chart_left = 92
    chart_bottom = 505
    chart_height = 110
    column_gap = 18
    column_width = 66
    for index, day in enumerate(recent_days):
        count = int(day.get("count", 0) or 0)
        x = chart_left + index * (column_width + column_gap)
        bar_height = 4 if count <= 0 else max(10, int(chart_height * count / max_count))
        
        # Column bar background slot
        draw.rounded_rectangle(
            (x, chart_bottom - chart_height, x + column_width, chart_bottom),
            radius=8,
            fill="#F1ECF8",
        )
        # Active column bar
        if count > 0:
            draw.rounded_rectangle(
                (x, chart_bottom - bar_height, x + column_width, chart_bottom),
                radius=8,
                fill=COLORS["accent"],
            )
        count_text = str(count)
        count_width = draw.textlength(count_text, font=tiny_bold_font)
        draw.text(
            (x + (column_width - count_width) / 2, chart_bottom - bar_height - 22),
            count_text,
            font=tiny_bold_font,
            fill=COLORS["accent"] if count else COLORS["muted"],
        )
        date_text = str(day.get("date", ""))[-5:].replace("-", "/")
        date_width = draw.textlength(date_text, font=tiny_font)
        draw.text(
            (x + (column_width - date_width) / 2, chart_bottom + 10),
            date_text,
            font=tiny_font,
            fill=COLORS["faint"],
        )

    # Right Panel: Local activity pulse
    draw.text((772, 186), "本周节奏", font=heading_font, fill=COLORS["ink"])
    recent_counts = [
        int(item.get("count", 0) or 0)
        for item in statistics.get("recent_days", [])[:7]
    ]
    average_count = round(sum(recent_counts) / len(recent_counts)) if recent_counts else 0
    peak_count = max(recent_counts or [0])
    active_days = sum(1 for count in recent_counts if count > 0)
    peak_day = next(
        (
            str(item.get("date", ""))[-5:].replace("-", "/")
            for item in statistics.get("recent_days", [])[:7]
            if int(item.get("count", 0) or 0) == peak_count and peak_count > 0
        ),
        "暂无",
    )
    activity_items = (
        ("日均出图", f"{average_count:,} 张"),
        ("最高一天", f"{peak_count:,} 张 · {peak_day}"),
        ("活跃天数", f"{active_days} / {len(recent_counts) or 7} 天"),
    )
    for index, (label, value) in enumerate(activity_items):
        y = 242 + index * 72
        draw.text((772, y), label, font=small_font, fill=COLORS["muted"])
        draw.text((772, y + 27), value, font=body_bold_font, fill=COLORS["ink"])
        if index < len(activity_items) - 1:
            draw.line((772, y + 64, CARD_WIDTH - margin - 30, y + 64), fill=COLORS["line"], width=1)

    queue_length = int(statistics.get("queue_length", 0) or 0)
    is_processing = bool(statistics.get("processing_request"))
    current_status = (
        f"出图中 · 前面还有 {queue_length} 位"
        if is_processing and queue_length
        else "出图中"
        if is_processing
        else "现在空闲"
    )
    draw.text((772, 458), "当前状态", font=small_font, fill=COLORS["muted"])
    draw.text((772, 485), current_status, font=body_bold_font, fill=COLORS["accent"])

    # Leaderboard Section (Top 10 users, 2 columns)
    daily_leaderboard = daily_leaderboard or []
    leaderboard_box = (margin, 582, CARD_WIDTH - margin, 1410)
    _draw_acrylic_card(draw, leaderboard_box, radius=24, border_color="#FFFFFF", fill_color=COLORS["surface"])

    # Column Titles
    draw.text((88, 608), "近 30 天排行", font=heading_font, fill=COLORS["ink"])
    draw.text((88, 646), "按成功出图次数排列", font=small_font, fill=COLORS["muted"])
    draw.text((628, 608), "今日画图排行", font=heading_font, fill=COLORS["ink"])
    draw.text((628, 646), "今日画图排行，按成功出图次数排列", font=small_font, fill=COLORS["muted"])

    # Divider between columns
    draw.line((594, 608, 594, 1388), fill=COLORS["line"], width=1)

    output_label = "出图"
    output_label_width = draw.textlength(output_label, font=small_font)
    draw.text(
        (566 - output_label_width, 622),
        output_label,
        font=small_font,
        fill=COLORS["faint"],
    )
    draw.text(
        (CARD_WIDTH - margin - 30 - output_label_width, 622),
        output_label,
        font=small_font,
        fill=COLORS["faint"],
    )

    row_top = 680
    row_height = 72
    rank_colors = {
        1: ("#EF4444", "#FEE2E2"),  # 1st Gold/Red
        2: ("#F59E0B", "#FEF3C7"),  # 2nd Orange/Gold
        3: ("#8B5CF6", "#EDE9FE"),  # 3rd Violet
    }

    for column, ranking in enumerate((leaderboard, daily_leaderboard)):
        column_left = 88 if column == 0 else 628
        column_right = 566 if column == 0 else CARD_WIDTH - margin - 30
        top_value = max(
            [
                int(item.get("total" if column == 0 else "today", 0) or 0)
                for item in ranking[:10]
            ]
            or [1]
        )
        for index, user in enumerate(ranking[:10]):
            y = row_top + index * row_height
            if index > 0:
                draw.line(
                    (column_left, y, column_right, y), fill="#EFEBF6", width=1
                )

            # Rank badge
            rank_num = index + 1
            rank_text = f"{rank_num:02d}"
            
            if rank_num in rank_colors:
                badge_fg, badge_bg = rank_colors[rank_num]
                draw.rounded_rectangle(
                    (column_left, y + 22, column_left + 36, y + 54),
                    radius=8,
                    fill=badge_bg,
                )
                draw.text(
                    (column_left + 8, y + 28),
                    rank_text,
                    font=small_bold_font,
                    fill=badge_fg,
                )
            else:
                draw.text(
                    (column_left + 6, y + 28),
                    rank_text,
                    font=small_bold_font,
                    fill=COLORS["faint"],
                )

            # Avatar
            avatar_size = 46
            avatar_x = column_left + 48
            avatar_y = y + 16
            avatar = _avatar_image(user.get("avatar"), avatar_size)
            mask = Image.new("L", (avatar_size, avatar_size), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, avatar_size, avatar_size), fill=255)
            if avatar is not None:
                canvas.paste(avatar, (avatar_x, avatar_y), mask)
            else:
                draw.ellipse(
                    (avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size),
                    fill=COLORS["accent_soft"],
                )
                initial = str(user.get("nickname") or user.get("user_id") or "?")[:1]
                initial_font = _font(20, bold=True)
                initial_box = draw.textbbox((0, 0), initial, font=initial_font)
                draw.text(
                    (
                        avatar_x + (avatar_size - initial_box[2] + initial_box[0]) / 2,
                        avatar_y + (avatar_size - initial_box[3] + initial_box[1]) / 2 - 2,
                    ),
                    initial,
                    font=initial_font,
                    fill=COLORS["accent"],
                )

            # Nickname & QQ
            nickname = _truncate_text(
                draw,
                str(user.get("nickname") or user.get("user_id") or "未知画师"),
                body_bold_font,
                160,
            )
            draw.text((column_left + 106, y + 14), nickname, font=body_bold_font, fill=COLORS["ink"])
            draw.text(
                (column_left + 106, y + 44),
                f"ID {user.get('user_id', '')}",
                font=tiny_font,
                fill=COLORS["muted"],
            )

            # Usage Count & Progress Bar
            value_key = "total" if column == 0 else "today"
            value = int(user.get(value_key, 0) or 0)
            bar_width = max(6, int(88 * value / top_value))
            draw.rounded_rectangle(
                (column_right - 170, y + 46, column_right - 170 + bar_width, y + 52),
                radius=3,
                fill=COLORS["accent"],
            )
            value_text = f"{value:,}"
            value_width = draw.textlength(value_text, font=metric_small_font)
            draw.text(
                (column_right - value_width, y + 18),
                value_text,
                font=metric_small_font,
                fill=COLORS["ink"],
            )

    if not leaderboard and not daily_leaderboard:
        empty_text = "还没有出图记录"
        empty_width = draw.textlength(empty_text, font=body_font)
        draw.text(
            ((CARD_WIDTH - empty_width) / 2, 1100),
            empty_text,
            font=body_font,
            fill=COLORS["muted"],
        )
    elif not daily_leaderboard:
        empty_text = "今天还没有出图"
        empty_width = draw.textlength(empty_text, font=body_font)
        draw.text(
            (628 + (450 - empty_width) / 2, 1100),
            empty_text,
            font=body_font,
            fill=COLORS["muted"],
        )

    # Footer
    footer_text = f"数据截至 {now:%Y-%m-%d %H:%M}"
    footer_width = draw.textlength(footer_text, font=tiny_font)
    draw.text(
        (CARD_WIDTH - margin - footer_width, 1428),
        footer_text,
        font=tiny_font,
        fill=COLORS["faint"],
    )

    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()

"""Shared data helpers for NovelAI style presets."""

from __future__ import annotations

from typing import Any

STYLE_FIELDS = ("positive_prompt", "negative_prompt", "preview_image")


def normalize_style_entry(value: Any) -> dict[str, str]:
    """Normalize a legacy or structured style preset.

    Args:
        value: Legacy positive-prompt string or structured style mapping.

    Returns:
        Style mapping with stable positive, negative, and preview fields.
    """
    if isinstance(value, str):
        return {
            "positive_prompt": value.strip(),
            "negative_prompt": "",
            "preview_image": "",
        }
    if not isinstance(value, dict):
        return dict.fromkeys(STYLE_FIELDS, "")
    return {
        "positive_prompt": str(
            value.get("positive_prompt", value.get("prompt", "")) or ""
        ).strip(),
        "negative_prompt": str(value.get("negative_prompt", "") or "").strip(),
        "preview_image": str(value.get("preview_image", "") or "").strip(),
    }


def normalize_style_presets(value: Any) -> dict[str, dict[str, str]]:
    """Normalize a complete style preset mapping.

    Args:
        value: Parsed style preset configuration.

    Returns:
        Valid named styles in source order.
    """
    if not isinstance(value, dict):
        return {}
    styles: dict[str, dict[str, str]] = {}
    for raw_name, raw_entry in value.items():
        name = str(raw_name or "").strip()
        entry = normalize_style_entry(raw_entry)
        if name and entry["positive_prompt"]:
            styles[name] = entry
    return styles

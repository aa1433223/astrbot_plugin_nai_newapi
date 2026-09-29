"""AstrBot Plugin Page APIs for managing NovelAI prompt presets."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from astrbot.api import logger
from astrbot.api.star import Context
from astrbot.api.web import error_response, json_response, request

try:
    from astrbot.core.star.star_tools import StarTools
except ImportError:  # pragma: no cover - supports older AstrBot releases
    StarTools = None

try:
    from .style_presets import normalize_style_entry, normalize_style_presets
except ImportError:  # pragma: no cover - supports direct module tests
    from style_presets import normalize_style_entry, normalize_style_presets

PLUGIN_NAME = "astrbot_plugin_novelai"
_UNSET = object()


class PresetValidationError(ValueError):
    """Represent a user-correctable preset management error."""


class PresetWebUIController:
    """Expose authenticated preset management APIs to an AstrBot Plugin Page."""

    CATEGORY_DEFINITIONS = (
        {
            "id": "unclassified",
            "storage_key": "其他",
            "label": "待分类",
            "description": "尚未归入四个正式分类的预设",
        },
        {
            "id": "character",
            "storage_key": "人物",
            "label": "角色预设",
            "description": "人物、角色外观与身份相关预设",
        },
        {
            "id": "clothing",
            "storage_key": "衣服",
            "label": "衣服预设",
            "description": "服装、鞋帽与配饰相关预设",
        },
        {
            "id": "scene",
            "storage_key": "场景",
            "label": "场景预设",
            "description": "背景、环境与地点相关预设",
        },
        {
            "id": "action",
            "storage_key": "动作",
            "label": "动作预设",
            "description": "姿势、动作与镜头行为相关预设",
        },
    )
    FORMAL_STORAGE_KEYS = ("人物", "衣服", "场景", "动作")
    MAX_NAME_LENGTH = 100
    MAX_PROMPT_LENGTH = 20000
    MAX_PREVIEW_BYTES = 6 * 1024 * 1024
    MAX_PREVIEW_PIXELS = 25_000_000
    PREVIEW_EDGE = 960

    def __init__(self, context: Context, plugin: Any):
        """Initialize the controller.

        Args:
            context: Active AstrBot plugin context.
            plugin: NovelAI plugin instance that owns preset state and config.
        """
        self.context = context
        self.plugin = plugin
        self._write_lock = asyncio.Lock()

    def register_routes(self) -> None:
        """Register all dashboard-authenticated preset management routes."""
        routes = (
            ("/presets/state", self.page_state, ["GET"], "Load preset manager state"),
            ("/presets/create", self.page_create, ["POST"], "Create a preset"),
            ("/presets/update", self.page_update, ["POST"], "Update a preset"),
            ("/presets/move", self.page_move, ["POST"], "Move prompt presets"),
            ("/presets/delete", self.page_delete, ["POST"], "Delete prompt presets"),
            ("/styles/state", self.page_style_state, ["GET"], "Load style library"),
            ("/styles/create", self.page_style_create, ["POST"], "Create a style"),
            ("/styles/update", self.page_style_update, ["POST"], "Update a style"),
            ("/styles/delete", self.page_style_delete, ["POST"], "Delete a style"),
            (
                "/styles/upload",
                self.page_style_upload,
                ["POST"],
                "Upload a style preview",
            ),
        )
        for path, handler, methods, description in routes:
            self.context.register_web_api(
                f"/{PLUGIN_NAME}{path}",
                handler,
                methods,
                description,
            )

    def serialize_state(self) -> dict[str, Any]:
        """Build the complete preset manager payload from current plugin state.

        Returns:
            Categories with ordered preset rows and aggregate statistics.
        """
        presets = self.plugin.presets
        categories = self.plugin.preset_categories
        assigned_names: set[str] = set()
        categorized_names: dict[str, list[str]] = {}

        for definition in self.CATEGORY_DEFINITIONS[1:]:
            storage_key = definition["storage_key"]
            names = []
            raw_names = categories.get(storage_key, [])
            if isinstance(raw_names, list):
                for name in raw_names:
                    if (
                        isinstance(name, str)
                        and name in presets
                        and name not in assigned_names
                    ):
                        names.append(name)
                        assigned_names.add(name)
            categorized_names[definition["id"]] = names

        waiting_names: list[str] = []
        waiting_category_keys = ["其他"]
        waiting_category_keys.extend(
            key
            for key in categories
            if key not in self.FORMAL_STORAGE_KEYS and key != "其他"
        )
        for category_key in waiting_category_keys:
            raw_names = categories.get(category_key, [])
            if not isinstance(raw_names, list):
                continue
            for name in raw_names:
                if (
                    isinstance(name, str)
                    and name in presets
                    and name not in assigned_names
                ):
                    waiting_names.append(name)
                    assigned_names.add(name)
        for name in presets:
            if name not in assigned_names:
                waiting_names.append(name)
                assigned_names.add(name)
        categorized_names["unclassified"] = waiting_names

        serialized_categories = []
        for definition in self.CATEGORY_DEFINITIONS:
            rows = [
                {"name": name, "prompt": str(presets.get(name, ""))}
                for name in categorized_names[definition["id"]]
            ]
            serialized_categories.append(
                {
                    "id": definition["id"],
                    "label": definition["label"],
                    "description": definition["description"],
                    "count": len(rows),
                    "presets": rows,
                }
            )

        return {
            "categories": serialized_categories,
            "stats": {
                "total": len(presets),
                "unclassified": len(waiting_names),
            },
        }

    def create_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create and persist one prompt preset.

        Args:
            payload: Name, prompt, and UI category identifier.

        Returns:
            Refreshed preset manager state.

        Raises:
            PresetValidationError: If the payload is invalid or the name exists.
        """
        name = self._validate_name(payload.get("name"))
        prompt = self._validate_prompt(payload.get("prompt"))
        storage_key = self._resolve_category(payload.get("category"))
        self._ensure_unique_name(name)

        presets = dict(self.plugin.presets)
        categories = self._copy_categories()
        presets[name] = prompt
        categories.setdefault(storage_key, []).append(name)
        self._persist(presets, categories)
        return self.serialize_state()

    def update_preset(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update a prompt preset and keep category references consistent.

        Args:
            payload: Original name, replacement fields, and target category.

        Returns:
            Refreshed preset manager state.

        Raises:
            PresetValidationError: If the preset or replacement values are invalid.
        """
        original_name = self._validate_existing_name(payload.get("original_name"))
        name = self._validate_name(payload.get("name"))
        prompt = self._validate_prompt(payload.get("prompt"))
        storage_key = self._resolve_category(payload.get("category"))
        self._ensure_unique_name(name, excluding=original_name)

        presets: dict[str, Any] = {}
        for existing_name, existing_prompt in self.plugin.presets.items():
            if existing_name == original_name:
                presets[name] = prompt
            else:
                presets[existing_name] = existing_prompt

        categories = self._copy_categories()
        current_key = None
        current_index = None
        for category_key, names in categories.items():
            if original_name in names and current_key is None:
                current_key = category_key
                current_index = names.index(original_name)
            categories[category_key] = [
                item for item in names if item != original_name and item != name
            ]

        target_names = categories.setdefault(storage_key, [])
        if current_key == storage_key and current_index is not None:
            target_names.insert(min(current_index, len(target_names)), name)
        else:
            target_names.append(name)
        self._persist(presets, categories)
        return self.serialize_state()

    def move_presets(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Move one or more presets to a category and optional order position.

        Args:
            payload: Ordered preset names, target category, and optional target index.

        Returns:
            Refreshed preset manager state.

        Raises:
            PresetValidationError: If names, category, or target index are invalid.
        """
        names = self._validate_existing_names(payload.get("names"))
        storage_key = self._resolve_category(payload.get("target_category"))
        raw_index = payload.get("target_index")
        if raw_index is None:
            target_index = None
        elif isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise PresetValidationError("目标位置必须是整数。")
        elif raw_index < 0:
            raise PresetValidationError("目标位置不能小于 0。")
        else:
            target_index = raw_index

        categories = self._copy_categories()
        moving = set(names)
        for category_key, category_names in categories.items():
            categories[category_key] = [
                name for name in category_names if name not in moving
            ]
        target_names = categories.setdefault(storage_key, [])
        insert_at = (
            len(target_names)
            if target_index is None
            else min(target_index, len(target_names))
        )
        target_names[insert_at:insert_at] = names
        self._persist(dict(self.plugin.presets), categories)
        return self.serialize_state()

    def delete_presets(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Delete one or more presets and all category references.

        Args:
            payload: Preset names to remove.

        Returns:
            Refreshed preset manager state.

        Raises:
            PresetValidationError: If the names list is invalid.
        """
        names = self._validate_existing_names(payload.get("names"))
        deleting = set(names)
        presets = {
            name: prompt
            for name, prompt in self.plugin.presets.items()
            if name not in deleting
        }
        categories = self._copy_categories()
        for category_key, category_names in categories.items():
            categories[category_key] = [
                name for name in category_names if name not in deleting
            ]
        self._persist(presets, categories)
        return self.serialize_state()

    async def page_state(self):
        """Return the current preset manager state to the Plugin Page."""
        try:
            return json_response(self.serialize_state())
        except Exception:
            logger.exception("Failed to load NovelAI preset manager state")
            return error_response(
                "预设列表加载失败，请查看 AstrBot 日志。", status_code=500
            )

    async def page_create(self):
        """Handle a create request from the Plugin Page."""
        return await self._handle_write(self.create_preset)

    async def page_update(self):
        """Handle an update request from the Plugin Page."""
        return await self._handle_write(self.update_preset)

    async def page_move(self):
        """Handle a move request from the Plugin Page."""
        return await self._handle_write(self.move_presets)

    async def page_delete(self):
        """Handle a delete request from the Plugin Page."""
        return await self._handle_write(self.delete_presets)

    def serialize_style_state(self) -> dict[str, Any]:
        """Build style cards and inline their optimized preview thumbnails.

        Returns:
            Ordered style rows and aggregate statistics for the style page.
        """
        styles = normalize_style_presets(self.plugin.style_presets)
        rows = []
        for name, entry in styles.items():
            preview_image = entry["preview_image"]
            preview_url = ""
            if preview_image:
                image_path = self._style_image_path(preview_image)
                if image_path and image_path.is_file():
                    try:
                        preview_url = "data:image/webp;base64," + base64.b64encode(
                            image_path.read_bytes()
                        ).decode("ascii")
                    except OSError:
                        logger.warning(
                            "Failed to read style preview for %s: %s",
                            name,
                            image_path,
                        )
            rows.append({"name": name, **entry, "preview_url": preview_url})
        return {
            "styles": rows,
            "stats": {
                "total": len(rows),
                "with_preview": sum(bool(row["preview_url"]) for row in rows),
            },
        }

    def create_style(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create and persist a structured style preset.

        Args:
            payload: Style name, prompt fields, and uploaded preview filename.

        Returns:
            Refreshed style library state.

        Raises:
            PresetValidationError: If required style fields are invalid.
        """
        name = self._validate_style_name(payload.get("name"))
        styles = normalize_style_presets(self.plugin.style_presets)
        self._ensure_unique_style_name(styles, name)
        styles[name] = self._validate_style_entry(payload, require_preview=True)
        self._persist_styles(styles)
        return self.serialize_style_state()

    def update_style(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update a style preset while preserving its order.

        Args:
            payload: Original style name and complete replacement fields.

        Returns:
            Refreshed style library state.

        Raises:
            PresetValidationError: If the style or replacement fields are invalid.
        """
        styles = normalize_style_presets(self.plugin.style_presets)
        original_name = self._validate_style_name(payload.get("original_name"))
        if original_name not in styles:
            raise PresetValidationError(f"画风“{original_name}”不存在或已被删除。")
        name = self._validate_style_name(payload.get("name"))
        self._ensure_unique_style_name(styles, name, excluding=original_name)
        current_entry = styles[original_name]
        entry_payload = dict(payload)
        if "preview_image" not in entry_payload:
            entry_payload["preview_image"] = current_entry["preview_image"]
        replacement = self._validate_style_entry(entry_payload, require_preview=True)

        updated = {}
        for existing_name, existing_entry in styles.items():
            updated[name if existing_name == original_name else existing_name] = (
                replacement if existing_name == original_name else existing_entry
            )
        selected_style = (
            name if self.plugin.selected_style_preset == original_name else _UNSET
        )
        self._persist_styles(updated, selected_style)
        old_preview = current_entry["preview_image"]
        if old_preview and old_preview != replacement["preview_image"]:
            self._delete_style_image(old_preview)
        for settings in self.plugin.session_drawing_settings.values():
            if settings.get("style_name") == original_name:
                settings["style_name"] = name
        return self.serialize_style_state()

    def delete_style(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Delete a style and its managed preview asset.

        Args:
            payload: Exact style name to remove.

        Returns:
            Refreshed style library state.

        Raises:
            PresetValidationError: If the style does not exist.
        """
        styles = normalize_style_presets(self.plugin.style_presets)
        name = self._validate_style_name(payload.get("name"))
        if name not in styles:
            raise PresetValidationError(f"画风“{name}”不存在或已被删除。")
        preview_image = styles.pop(name)["preview_image"]
        selected_style = "" if self.plugin.selected_style_preset == name else _UNSET
        self._persist_styles(styles, selected_style)
        self._delete_style_image(preview_image)

        for settings in self.plugin.session_drawing_settings.values():
            if settings.get("style_name") == name:
                settings["style_name"] = ""
                settings["style_enabled"] = False
        return self.serialize_style_state()

    async def page_style_state(self):
        """Return the current style library state to the Plugin Page."""
        try:
            return json_response(self.serialize_style_state())
        except Exception:
            logger.exception("Failed to load NovelAI style library state")
            return error_response(
                "画风库加载失败，请查看 AstrBot 日志。", status_code=500
            )

    async def page_style_create(self):
        """Handle a style creation request from the Plugin Page."""
        return await self._handle_write(self.create_style, "画风保存失败")

    async def page_style_update(self):
        """Handle a style update request from the Plugin Page."""
        return await self._handle_write(self.update_style, "画风保存失败")

    async def page_style_delete(self):
        """Handle a style deletion request from the Plugin Page."""
        return await self._handle_write(self.delete_style, "画风删除失败")

    async def page_style_upload(self):
        """Validate, resize, and persist one style preview image."""
        try:
            files = await request.files()
            upload = files.get("file")
            if upload is None:
                raise PresetValidationError("请选择一张示例图。")
            if upload.content_length and upload.content_length > self.MAX_PREVIEW_BYTES:
                raise PresetValidationError("示例图不能超过 6 MB。")
            image_bytes = await upload.read(self.MAX_PREVIEW_BYTES + 1)
            if len(image_bytes) > self.MAX_PREVIEW_BYTES:
                raise PresetValidationError("示例图不能超过 6 MB。")
            optimized_image = self._optimize_style_preview(image_bytes)

            filename = f"{uuid4().hex}.webp"
            image_dir = self._style_image_dir()
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / filename
            image_path.write_bytes(optimized_image)
            preview_url = "data:image/webp;base64," + base64.b64encode(
                optimized_image
            ).decode("ascii")
            return json_response(
                {"preview_image": filename, "preview_url": preview_url}
            )
        except PresetValidationError as exc:
            return error_response(str(exc), status_code=400)
        except Exception:
            logger.exception("NovelAI style preview upload failed")
            return error_response(
                "示例图上传失败，请查看 AstrBot 日志。", status_code=500
            )

    def _optimize_style_preview(self, image_bytes: bytes) -> bytes:
        """Validate and encode an uploaded preview as a bounded WebP image.

        Args:
            image_bytes: Original PNG, JPEG, or WebP upload.

        Returns:
            Optimized WebP image bytes.

        Raises:
            PresetValidationError: If the upload is unsafe or cannot be decoded.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                if source.width * source.height > self.MAX_PREVIEW_PIXELS:
                    raise PresetValidationError(
                        "示例图像素尺寸过大，请缩小后重新上传。"
                    )
                source.verify()
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = ImageOps.exif_transpose(source)
                image.thumbnail((self.PREVIEW_EDGE, self.PREVIEW_EDGE))
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert(
                        "RGBA" if "transparency" in image.info else "RGB"
                    )
                output = io.BytesIO()
                image.save(output, format="WEBP", quality=84, method=6)
                return output.getvalue()
        except PresetValidationError:
            raise
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as exc:
            raise PresetValidationError(
                "无法识别该图片，请使用 PNG、JPEG 或 WebP。"
            ) from exc

    async def _handle_write(
        self,
        operation: Callable[[dict[str, Any]], dict[str, Any]],
        error_message: str = "预设保存失败",
    ):
        """Parse and serialize one mutating Plugin Page request.

        Args:
            operation: Validated synchronous mutation to execute under the write lock.

        Returns:
            JSON success or error response for the bridge.
        """
        try:
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise PresetValidationError("请求内容必须是 JSON 对象。")
            async with self._write_lock:
                return json_response(operation(payload))
        except PresetValidationError as exc:
            return error_response(str(exc), status_code=400)
        except Exception:
            logger.exception("NovelAI preset manager write request failed")
            return error_response(
                f"{error_message}，请查看 AstrBot 日志。", status_code=500
            )

    def _persist_styles(
        self,
        styles: dict[str, dict[str, str]],
        selected_style: Any = _UNSET,
    ) -> None:
        """Persist a complete structured style mapping atomically.

        Args:
            styles: Complete replacement style mapping.
            selected_style: Optional replacement for the global style selection.

        Raises:
            Exception: If AstrBot cannot persist the configuration file.
        """
        previous_styles = self.plugin.config.get("style_presets")
        previous_selected_style = self.plugin.config.get("selected_style_preset")
        self.plugin.config["style_presets"] = json.dumps(styles, ensure_ascii=False)
        if selected_style is not _UNSET:
            self.plugin.config["selected_style_preset"] = selected_style
        try:
            self.plugin.config.save_config()
        except Exception:
            self.plugin.config["style_presets"] = previous_styles
            if selected_style is not _UNSET:
                self.plugin.config["selected_style_preset"] = previous_selected_style
            raise
        self.plugin.style_presets = styles
        if selected_style is not _UNSET:
            self.plugin.selected_style_preset = selected_style

    def _validate_style_name(self, value: Any) -> str:
        """Validate and normalize a style name.

        Args:
            value: Candidate style name.

        Returns:
            Trimmed style name.

        Raises:
            PresetValidationError: If the name is empty or too long.
        """
        if not isinstance(value, str) or not value.strip():
            raise PresetValidationError("画风名称不能为空。")
        name = value.strip()
        if len(name) > self.MAX_NAME_LENGTH:
            raise PresetValidationError(
                f"画风名称不能超过 {self.MAX_NAME_LENGTH} 个字符。"
            )
        return name

    def _validate_style_entry(
        self, value: Any, *, require_preview: bool
    ) -> dict[str, str]:
        """Validate a complete structured style entry.

        Args:
            value: Candidate style mapping.
            require_preview: Whether a managed preview filename is mandatory.

        Returns:
            Normalized style entry.

        Raises:
            PresetValidationError: If a prompt or preview field is invalid.
        """
        entry = normalize_style_entry(value)
        if not entry["positive_prompt"]:
            raise PresetValidationError("正面提示词不能为空。")
        if not entry["negative_prompt"]:
            raise PresetValidationError("负面提示词不能为空。")
        for field in ("positive_prompt", "negative_prompt"):
            if len(entry[field]) > self.MAX_PROMPT_LENGTH:
                raise PresetValidationError("提示词不能超过 20,000 个字符。")
        preview_image = entry["preview_image"]
        if require_preview and not preview_image:
            raise PresetValidationError("请为画风上传一张示例图。")
        if preview_image:
            image_path = self._style_image_path(preview_image)
            if image_path is None:
                raise PresetValidationError("示例图引用无效，请重新上传。")
            if not image_path.is_file():
                raise PresetValidationError("示例图文件不存在，请重新上传。")
        return entry

    @staticmethod
    def _ensure_unique_style_name(
        styles: dict[str, dict[str, str]], name: str, excluding: str | None = None
    ) -> None:
        """Reject case-insensitive duplicate style names.

        Args:
            styles: Existing normalized style mapping.
            name: Candidate style name.
            excluding: Existing style being renamed, if any.

        Raises:
            PresetValidationError: If another style uses the same folded name.
        """
        folded_name = name.casefold()
        for existing_name in styles:
            if existing_name != excluding and existing_name.casefold() == folded_name:
                raise PresetValidationError(f"画风“{name}”已存在。")

    def _style_image_dir(self) -> Path:
        """Return the plugin-owned style preview directory.

        Returns:
            Persistent preview directory under AstrBot plugin data.
        """
        # ``get_data_dir`` belongs to AstrBot's ``StarTools`` helper, not to
        # individual ``Star`` plugin instances.  Keep the plugin method as a
        # compatibility hook for the lightweight test doubles and older
        # integrations that provided one themselves.
        plugin_get_data_dir = getattr(self.plugin, "get_data_dir", None)
        if callable(plugin_get_data_dir):
            return Path(plugin_get_data_dir(PLUGIN_NAME)) / "style_previews"

        if StarTools is None:
            raise RuntimeError("AstrBot does not provide a plugin data directory API")
        return Path(StarTools.get_data_dir(PLUGIN_NAME)) / "style_previews"

    def _style_image_path(self, filename: str) -> Path | None:
        """Resolve a managed preview filename without accepting path traversal.

        Args:
            filename: Stored preview asset identifier.

        Returns:
            Resolved asset path, or None when the identifier is unsafe.
        """
        candidate = Path(str(filename or ""))
        if (
            not candidate.name
            or candidate.name != str(filename)
            or candidate.suffix.lower() != ".webp"
            or len(candidate.stem) != 32
            or any(character not in "0123456789abcdef" for character in candidate.stem)
        ):
            return None
        return self._style_image_dir() / candidate.name

    def _delete_style_image(self, filename: str) -> None:
        """Remove an exact managed preview after its style no longer references it.

        Args:
            filename: Stored preview asset identifier.
        """
        image_path = self._style_image_path(filename)
        if not image_path:
            return
        if any(
            entry["preview_image"] == filename
            for entry in normalize_style_presets(self.plugin.style_presets).values()
        ):
            return
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete unused style preview: %s", image_path)

    def _copy_categories(self) -> dict[str, list[str]]:
        """Copy category lists while tolerating malformed legacy values.

        Returns:
            Independent category mapping safe for mutation.
        """
        categories: dict[str, list[str]] = {}
        for category, names in self.plugin.preset_categories.items():
            categories[str(category)] = list(names) if isinstance(names, list) else []
        for storage_key in (*self.FORMAL_STORAGE_KEYS, "其他"):
            categories.setdefault(storage_key, [])
        return categories

    def _persist(
        self, presets: dict[str, Any], categories: dict[str, list[str]]
    ) -> None:
        """Atomically save both preset structures before publishing them in memory.

        Args:
            presets: Complete replacement preset mapping.
            categories: Complete replacement category mapping.

        Raises:
            Exception: If AstrBot cannot persist the configuration file.
        """
        previous_presets = self.plugin.config.get("presets")
        previous_categories = self.plugin.config.get("preset_categories")
        self.plugin.config["presets"] = json.dumps(presets, ensure_ascii=False)
        self.plugin.config["preset_categories"] = json.dumps(
            categories, ensure_ascii=False
        )
        try:
            self.plugin.config.save_config()
        except Exception:
            self.plugin.config["presets"] = previous_presets
            self.plugin.config["preset_categories"] = previous_categories
            raise
        self.plugin.presets = presets
        self.plugin.preset_categories = categories

    def _resolve_category(self, value: Any) -> str:
        """Resolve a UI identifier or legacy storage key to a storage category.

        Args:
            value: UI category identifier or accepted storage key.

        Returns:
            Existing storage key used by the drawing commands.

        Raises:
            PresetValidationError: If the category is unsupported.
        """
        if not isinstance(value, str):
            raise PresetValidationError("请选择有效的预设分类。")
        for definition in self.CATEGORY_DEFINITIONS:
            if value in {definition["id"], definition["storage_key"]}:
                return definition["storage_key"]
        raise PresetValidationError("请选择有效的预设分类。")

    def _validate_name(self, value: Any) -> str:
        """Validate and normalize a preset name.

        Args:
            value: Candidate preset name.

        Returns:
            Trimmed name.

        Raises:
            PresetValidationError: If the name is missing or too long.
        """
        if not isinstance(value, str) or not value.strip():
            raise PresetValidationError("预设名称不能为空。")
        name = value.strip()
        if len(name) > self.MAX_NAME_LENGTH:
            raise PresetValidationError(
                f"预设名称不能超过 {self.MAX_NAME_LENGTH} 个字符。"
            )
        return name

    def _validate_prompt(self, value: Any) -> str:
        """Validate and normalize prompt content.

        Args:
            value: Candidate prompt content.

        Returns:
            Trimmed prompt content.

        Raises:
            PresetValidationError: If the prompt is missing or too long.
        """
        if not isinstance(value, str) or not value.strip():
            raise PresetValidationError("提示词内容不能为空。")
        prompt = value.strip()
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise PresetValidationError(
                f"提示词不能超过 {self.MAX_PROMPT_LENGTH} 个字符。"
            )
        return prompt

    def _ensure_unique_name(self, name: str, excluding: str | None = None) -> None:
        """Reject names that would be ambiguous during case-insensitive expansion.

        Args:
            name: Candidate normalized name.
            excluding: Existing name being renamed, if any.

        Raises:
            PresetValidationError: If another preset uses the same folded name.
        """
        folded_name = name.casefold()
        for existing_name in self.plugin.presets:
            if existing_name == excluding:
                continue
            if existing_name.casefold() == folded_name:
                raise PresetValidationError(f"预设“{name}”已存在。")

    def _validate_existing_name(self, value: Any) -> str:
        """Validate one exact preset reference.

        Args:
            value: Candidate preset name.

        Returns:
            Existing exact preset name.

        Raises:
            PresetValidationError: If the preset does not exist.
        """
        name = self._validate_name(value)
        if name not in self.plugin.presets:
            raise PresetValidationError(f"预设“{name}”不存在或已被删除。")
        return name

    def _validate_existing_names(self, value: Any) -> list[str]:
        """Validate a non-empty ordered list of exact preset references.

        Args:
            value: Candidate list of preset names.

        Returns:
            De-duplicated names in request order.

        Raises:
            PresetValidationError: If the list or any referenced preset is invalid.
        """
        if not isinstance(value, list) or not value:
            raise PresetValidationError("请至少选择一个预设。")
        names: list[str] = []
        seen: set[str] = set()
        for item in value:
            name = self._validate_existing_name(item)
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names

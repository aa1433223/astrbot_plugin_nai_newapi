import asyncio
import base64
import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import httpx
import numpy as np
from PIL import Image as PILImage

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, llm_tool, logger

# 导入AstrBot相关模块
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except Exception:  # pragma: no cover - supports older AstrBot releases
    get_astrbot_data_path = None

try:
    from astrbot.core.star.star_tools import StarTools
except Exception:  # pragma: no cover - supports older AstrBot releases
    StarTools = None

try:
    from astrbot.core.utils.quoted_message import extract_quoted_message_images
except Exception:  # pragma: no cover - supports older AstrBot releases
    extract_quoted_message_images = None

# 导入衣服管理器
from .clothing_manager import ClothingManager
from .novelai_payload import (
    build_v4_negative_prompt,
    build_v4_prompt,
    deduplicate_negative_prompt,
    normalize_v5_prompt,
    remove_negative_prompt_tags,
    remove_positive_duplicates_from_negative,
)
from .preset_webui import PresetWebUIController
from .statistics_card import render_statistics_card
from .style_presets import normalize_style_entry, normalize_style_presets
from .style_gallery_renderer import (
    render_drawing_help_image,
    render_style_gallery_image,
)


STYLE_GALLERY_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "data" / "t2i_templates" / "style_gallery.html"
)
DRAWING_HELP_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "data" / "t2i_templates" / "drawing_help.html"
)


PLUGIN_NAME = "astrbot_plugin_novelai"
_LEGACY_DATA_DIR = Path(__file__).resolve().parent / "data"


def resolve_plugin_data_dir() -> Path:
    """Return the writable directory used for runtime plugin state.

    Prefers AstrBot's per-plugin data directory (``data/plugin_data/<name>``)
    so state does not depend on the process working directory, and falls back
    to the plugin-local ``data`` folder for older AstrBot releases or tests.

    Returns:
        Absolute path to the runtime data directory.
    """

    if StarTools is not None:
        try:
            return Path(StarTools.get_data_dir(PLUGIN_NAME))
        except Exception as exc:  # pragma: no cover - older AstrBot releases
            logger.warning("无法使用 AstrBot 插件数据目录，改用本地目录: %s", exc)

    if get_astrbot_data_path is not None:
        try:
            return Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
        except Exception as exc:  # pragma: no cover - older AstrBot releases
            logger.warning("无法定位 AstrBot 数据目录，改用本地目录: %s", exc)

    return _LEGACY_DATA_DIR


def _migrate_legacy_state_file(target: Path) -> None:
    """Copy a pre-1.5.1 state file from the legacy ``data`` folder if needed.

    Older releases wrote runtime JSON state relative to the working directory.
    When the new target file is still missing, reuse the legacy copy so quotas
    and the last-draw cache survive the upgrade.

    Args:
        target: Newly resolved state file path.
    """

    if target.exists():
        return

    legacy = _LEGACY_DATA_DIR / target.name
    if not legacy.exists() or legacy.resolve() == target.resolve():
        return

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, target)
        logger.info("已将旧数据迁移到插件数据目录: %s", target)
    except Exception as exc:
        logger.warning("迁移旧数据文件失败 %s: %s", target, exc)


class CacheManager:
    """统一缓存管理器"""

    def __init__(self):
        self.config_cache = {}
        self.cache_timeout = 3600  # 1小时过期

    def _is_expired(self, timestamp: float, ttl: float = None) -> bool:
        """检查缓存是否过期"""
        timeout = self.cache_timeout if ttl is None else max(0.0, float(ttl))
        return time.time() - timestamp >= timeout

    # 通用缓存接口（向后兼容）
    def get(self, key: str, default=None):
        """获取通用缓存值"""
        if key in self.config_cache:
            result, timestamp, ttl = self.config_cache[key]
            if not self._is_expired(timestamp, ttl):
                return result
            # 清理过期缓存
            del self.config_cache[key]
        return default

    def set(self, key: str, value, ttl: float = None):
        """设置通用缓存值"""
        self.config_cache[key] = (value, time.time(), ttl)

    def delete(self, key: str):
        """删除缓存"""
        self.config_cache.pop(key, None)

    def clear(self):
        """清空所有缓存"""
        self.config_cache.clear()


# 自定义异常类
class NovelAIError(Exception):
    """NovelAI插件基础异常类"""

    pass


class ImageProcessingError(NovelAIError):
    """图片处理异常"""

    pass


class MetadataExtractionError(NovelAIError):
    """元数据提取异常"""

    pass


class APIError(NovelAIError):
    """API调用异常"""

    pass


class ValidationError(NovelAIError):
    """验证异常"""

    pass


# 性能监控装饰器
def performance_monitor(func_name: str = None):
    """性能监控装饰器"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                if execution_time > 1.0:  # 超过1秒记录警告
                    logger.warning(
                        f"{func_name or func.__name__} 执行时间较长: {execution_time:.2f}s"
                    )
                else:
                    logger.debug(
                        f"{func_name or func.__name__} 执行时间: {execution_time:.2f}s"
                    )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"{func_name or func.__name__} 执行失败 (耗时: {execution_time:.2f}s): {e}"
                )
                raise

        return wrapper

    return decorator


# 常量定义
class Constants:
    # API URLs
    NOVELAI_API_BASE = "https://image.novelai.net"
    BAIDU_OAUTH_API = "https://aip.baidubce.com/oauth/2.0/token"
    BAIDU_AUDIT_API = (
        "https://aip.baidubce.com/rest/2.0/solution/v1/img_censor/v2/user_defined"
    )
    # /账户额度：持久 Token 只在 image/text 主机上被接受，api.novelai.net 会拒绝
    ACCOUNT_QUOTA_ENDPOINT = "/user/data"
    ACCOUNT_QUOTA_TIMEOUT = 15.0
    # 额度查询是交互式命令，重试次数上限低于画图请求
    ACCOUNT_QUOTA_MAX_ATTEMPTS = 3
    # 官网前端用 round(17.3 * percent) 估算剩余可生成图片数，见 Opus 额度条
    ACCOUNT_QUOTA_IMAGES_PER_PERCENT = 17.3
    ACCOUNT_QUOTA_MAIN_LABEL = "主账户"
    ACCOUNT_TIER_LABELS = {
        0: "无订阅",
        1: "Tablet",
        2: "Scroll",
        3: "Opus",
    }

    # 默认配置
    DEFAULT_MODEL = "nai-diffusion-5-full"
    DEFAULT_RESOLUTION = "832x1216"
    DEFAULT_STEPS = 28
    DEFAULT_SCALE = 6.5
    DEFAULT_SAMPLER = "k_euler_ancestral"
    DEFAULT_COOLDOWN = 10
    GROUP_DAILY_IMAGE_LIMIT = 5
    # Retry policy for failed image API requests.
    RETRY_BACKOFF_FACTOR = 2.0
    DEFAULT_RETRY_COUNT = 5
    DEFAULT_RETRY_MAX_DELAY = 10.0
    DAILY_FALLBACK_MODEL = "nai-diffusion-4-5-full"
    PERSONAL_KEY_MODEL = "nai-diffusion-5-full"
    PERSONAL_V5_FALLBACK_FLAG = "_personal_v5_access_fallback"
    PERSONAL_V5_FALLBACK_NOTICE = "⚠️ 个人 Key 无 V5 权限，已自动改用 V4.5 Full。"
    V4_5_NOISE_SCHEDULES = {
        "karras",
        "exponential",
        "polyexponential",
    }
    V4_5_CFG_DELAY_SIGMA = 58
    V4_5_AUTO_SMEA_MIN_PIXELS = 1024 * 1024
    IMAGE_MODELS = {
        "v5完整": "nai-diffusion-5-full",
        "v5精选": "nai-diffusion-5-curated",
        "v4.5完整": "nai-diffusion-4-5-full",
        "v4.5精选": "nai-diffusion-4-5-curated",
    }
    IMAGE_SAMPLERS = {
        "euler": "k_euler",
        "euler_a": "k_euler_ancestral",
        "dpmpp_2s_a": "k_dpmpp_2s_ancestral",
        "dpmpp_sde": "k_dpmpp_sde",
        "dpmpp_2m": "k_dpmpp_2m",
        "dpmpp_2m_sde": "k_dpmpp_2m_sde",
        "dpmpp_3m_sde": "k_dpmpp_3m_sde",
    }
    IMAGE_RESOLUTIONS = {
        "长图": "832x1216",
        "宽图": "1216x832",
        "方图": "1024x1024",
    }

    # 审核相关
    AUDIT_CONCLUSION_STRICT = 1
    AUDIT_CONCLUSION_MODERATE = 2
    AUDIT_CONCLUSION_LOOSE = 3

    # 文件路径
    DEFAULT_SAVE_PATH = "novelai_images"
    DEFAULT_AUDIT_FAILED_PATH = "novelai_audit_failed"

    # 性能限制
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB 最大图片大小
    MAX_IMAGE_DIMENSION = 4096  # 最大图片尺寸
    LARGE_IMAGE_THRESHOLD = 5 * 1024 * 1024  # 5MB 大图片阈值

    # 预设分类
    DEFAULT_CATEGORIES = {
        "人物": {"description": "角色、人物相关的预设", "presets": []},
        "衣服": {"description": "服装、配饰相关的预设", "presets": []},
        "动作": {"description": "姿势、动作相关的预设", "presets": []},
        "场景": {"description": "背景、环境相关的预设", "presets": []},
        "风格": {"description": "画风、风格相关的预设", "presets": []},
        "其他": {"description": "其他类型的预设", "presets": []},
    }


class HttpClientManager:
    """HTTP客户端管理器"""

    def __init__(self):
        self._clients = {}
        self._default_timeout = 30.0

    def get_client(
        self,
        client_type: str = "default",
        timeout: float = None,
        use_proxy: bool = False,
        proxy_url: str = None,
    ) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        client_key = (
            f"{client_type}_{use_proxy}_{proxy_url}" if use_proxy else client_type
        )

        if client_key not in self._clients:
            timeout_value = timeout or self._default_timeout

            client_config = {
                "timeout": timeout_value,
                "limits": httpx.Limits(
                    max_keepalive_connections=3,
                    max_connections=5,
                    keepalive_expiry=30.0,
                ),
            }

            # httpx >= 0.23 使用单数 proxy 参数
            if (
                use_proxy
                and proxy_url
                and proxy_url.startswith(("http://", "https://", "socks5://"))
            ):
                client_config["proxy"] = proxy_url

            self._clients[client_key] = httpx.AsyncClient(**client_config)
        return self._clients[client_key]

    async def close_all(self):
        """关闭所有客户端"""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


class ConfigManager:
    """配置管理器"""

    def __init__(self, config: AstrBotConfig):
        self.config = config
        self._config_cache = {}

    def get_with_default(
        self, key: str, default_value: Any, value_type: type = None
    ) -> Any:
        """获取配置值，带默认值和类型验证"""
        if key in self._config_cache:
            return self._config_cache[key]

        value = self.config.get(key, default_value)
        if value_type:
            value = Utils.validate_config_value(value, value_type, default_value)

        self._config_cache[key] = value
        return value

    def update_cache(self, key: str, value: Any):
        """更新缓存"""
        self._config_cache[key] = value

    def clear_cache(self):
        """清空配置缓存"""
        self._config_cache.clear()

    def get_api_config(self) -> Dict[str, Any]:
        """获取API相关配置"""
        return {
            "api_backend": self.get_with_default("api_backend", "novelai"),
            "api_token": self.get_with_default("api_token", ""),
            "newapi_base_url": self.get_with_default(
                "newapi_base_url", "http://localhost:3000"
            ),
            "newapi_api_key": self.get_with_default("newapi_api_key", ""),
            "api_proxy": self.get_with_default("api_proxy", ""),
            "image_proxy": self.get_with_default("image_proxy", ""),
        }

    def get_baidu_audit_config(self) -> Dict[str, Any]:
        """获取百度审核配置"""
        return {
            "api_key": self.get_with_default("baidu_audit_api_key", ""),
            "secret_key": self.get_with_default("baidu_audit_secret_key", ""),
            "enable": self.get_with_default("enable_image_audit", False, bool),
            "save_path": self.get_with_default(
                "audit_failed_save_path", Constants.DEFAULT_AUDIT_FAILED_PATH
            ),
            "conclusion_type": self.get_with_default(
                "audit_conclusion_type", Constants.AUDIT_CONCLUSION_MODERATE, int
            ),
            "strategy_id": self.get_with_default("audit_strategy_id", ""),
            "whitelist_groups": self.get_with_default(
                "audit_whitelist_groups", [], list
            ),
            "whitelist_bots": self.get_with_default("audit_whitelist_bots", [], list),
        }


class Utils:
    """工具方法类"""

    @staticmethod
    def safe_json_loads(json_str: Any, default_value: Any = None) -> Any:
        """安全的JSON解析，兼容字符串和原生对象"""
        if json_str is None:
            return default_value if default_value is not None else {}

        if isinstance(json_str, (dict, list)):
            return json_str

        if isinstance(json_str, (bytes, bytearray)):
            try:
                json_str = json_str.decode("utf-8")
            except Exception as e:
                logger.warning(f"JSON字节解码失败: {e}")
                return default_value if default_value is not None else {}

        if isinstance(json_str, str):
            if not json_str.strip():
                return default_value if default_value is not None else {}
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"JSON解析失败: {e}")
                return default_value if default_value is not None else {}

        logger.warning(f"JSON解析失败: 不支持的类型 {type(json_str).__name__}")
        return default_value if default_value is not None else {}

    @staticmethod
    def safe_json_dumps(obj: Any, default_value: str = "{}") -> str:
        """安全的JSON序列化"""
        try:
            return json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning(f"JSON序列化失败: {e}")
            return default_value

    @staticmethod
    def validate_config_value(value: Any, value_type: type, default_value: Any) -> Any:
        """验证配置值类型"""
        try:
            if value_type == bool:
                return bool(value)
            elif value_type == int:
                return int(value)
            elif value_type == float:
                return float(value)
            else:
                return value
        except (ValueError, TypeError):
            return default_value

    @staticmethod
    def create_directory_if_not_exists(path: str) -> bool:
        """创建目录（如果不存在）"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"创建目录失败 {path}: {e}")
            return False

    @staticmethod
    def generate_cache_key(*args) -> str:
        """生成缓存键"""
        return hashlib.md5(str(args).encode()).hexdigest()

    @staticmethod
    def is_valid_image_data(data: bytes) -> bool:
        """验证图片数据是否有效"""
        img = None
        try:
            img = PILImage.open(io.BytesIO(data))
            img.verify()
            return True
        except Exception:
            return False
        finally:
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f}{size_names[i]}"


class ApiResponseHandler:
    """API响应处理器"""

    @staticmethod
    def handle_http_response(
        response: httpx.Response, api_name: str = "API"
    ) -> Dict[str, Any]:
        """统一处理HTTP响应"""
        result = {
            "success": False,
            "data": None,
            "error": None,
            "status_code": response.status_code,
        }

        try:
            if response.status_code == 200:
                # 尝试解析JSON
                try:
                    result["data"] = response.json()
                    result["success"] = True
                except Exception:
                    # 如果不是JSON，返回原始内容
                    result["data"] = response.content
                    result["success"] = True
            else:
                result["error"] = f"{api_name}请求失败: HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if "error_msg" in error_data:
                        result["error"] += f" - {error_data['error_msg']}"
                except Exception:
                    result["error"] += f" - {response.text[:200]}"

        except Exception as e:
            result["error"] = f"{api_name}响应处理异常: {e}"

        return result

    @staticmethod
    def extract_error_message(
        response_data: Dict[str, Any], default_msg: str = "未知错误"
    ) -> str:
        """从响应数据中提取错误信息"""
        if not response_data.get("success", False):
            return response_data.get("error", default_msg)

        # 检查业务层面的错误
        data = response_data.get("data", {})
        if isinstance(data, dict):
            if "error_code" in data:
                error_code = data.get("error_code")
                error_msg = data.get("error_msg", default_msg)
                return f"错误码{error_code}: {error_msg}"

        return default_msg


# 添加错误类型定义（命名空间类，用于组织各类错误）
class NAIErrors:
    """NovelAI错误类型命名空间（避免与顶级NovelAIError基类冲突）"""

    class NetworkError(Exception):
        """网络连接错误"""

        def __init__(self, message="网络连接错误", original_error=None):
            self.message = message
            self.original_error = original_error
            super().__init__(self.message)

        def __str__(self):
            # 简化网络错误信息，避免显示冗长的技术细节
            if "客户端异常" in self.message and self.original_error:
                # 简化客户端异常信息
                error_str = str(self.original_error)
                if "Internal Server Error" in error_str:
                    return "服务器内部错误，请稍后重试"
                elif "timeout" in error_str.lower():
                    return "网络连接超时，请检查网络连接"
                elif "connection" in error_str.lower():
                    return "网络连接失败，请检查网络连接"
                else:
                    return "网络连接错误，请检查网络连接"
            return self.message

    class APIError(Exception):
        """API调用错误"""

        def __init__(self, status_code=None, response_text=None, message=None):
            self.status_code = status_code
            self.response_text = response_text
            self.message = message or f"API错误 (状态码: {status_code})"
            super().__init__(self.message)

        def __str__(self):
            if self.response_text:
                # 尝试解析JSON响应
                try:
                    error_json = json.loads(self.response_text)
                    if "error" in error_json:
                        error_msg = error_json["error"]
                        # 简化常见的错误信息
                        if "Internal Server Error" in error_msg:
                            return "服务器内部错误，请稍后重试"
                        return f"API错误: {error_msg}"
                    if "message" in error_json:
                        error_msg = error_json["message"]
                        # 简化常见的错误信息
                        if "Internal Server Error" in error_msg:
                            return "服务器内部错误，请稍后重试"
                        return f"API错误: {error_msg}"
                except (json.JSONDecodeError, KeyError):
                    pass

                # 处理非JSON响应，简化错误信息
                if "Internal Server Error" in self.response_text:
                    return "服务器内部错误，请稍后重试"
                elif "Bad Gateway" in self.response_text:
                    return "服务器网关错误，请稍后重试"
                elif "Service Unavailable" in self.response_text:
                    return "服务暂时不可用，请稍后重试"
                elif "Gateway Timeout" in self.response_text:
                    return "服务器响应超时，请稍后重试"
                else:
                    # 如果响应文本太长，截断它
                    if len(self.response_text) > 50:
                        return f"API错误: {self.response_text[:50]}..."
                    return f"API错误: {self.response_text}"
            return self.message

    class AuthenticationError(APIError):
        """认证错误"""

        def __init__(self, status_code=401, response_text=None):
            super().__init__(
                status_code, response_text, "API认证失败，请检查Token是否正确"
            )

    class QuotaExceededError(APIError):
        """配额超限错误"""

        def __init__(self, status_code=429, response_text=None):
            super().__init__(status_code, response_text, "API使用次数超限，请稍后再试")

    class ServerError(APIError):
        """服务器错误"""

        def __init__(self, status_code=None, response_text=None):
            # 根据状态码提供更友好的错误信息
            if status_code == 500:
                message = "服务器内部错误，请稍后重试"
            elif status_code == 502:
                message = "服务器网关错误，请稍后重试"
            elif status_code == 503:
                message = "服务暂时不可用，请稍后重试"
            elif status_code == 504:
                message = "服务器响应超时，请稍后重试"
            else:
                message = f"服务器错误 ({status_code})，请稍后重试"
            super().__init__(status_code, response_text, message)

    class ConfigurationError(Exception):
        """配置错误"""

        def __init__(self, message="配置错误"):
            self.message = message
            super().__init__(self.message)

    class ImageProcessingError(Exception):
        """图片处理错误"""

        def __init__(self, message="图片处理错误", original_error=None):
            self.message = message
            self.original_error = original_error
            super().__init__(self.message)

        def __str__(self):
            if self.original_error:
                return f"{self.message}: {self.original_error}"
            return self.message

    class ParameterError(Exception):
        """参数错误"""

        def __init__(self, message="参数错误"):
            self.message = message
            super().__init__(self.message)


class UsageManager:
    """使用量和排队管理器"""

    def __init__(
        self,
        max_daily: int = 50,
        cooldown: int = 10,
        max_group_daily: int = Constants.GROUP_DAILY_IMAGE_LIMIT,
        data_dir: Optional[Path] = None,
    ):
        self.max_daily_usage = max_daily
        self.max_group_daily_usage = max_group_daily
        self.cooldown_seconds = cooldown
        self.usage_data_file = str(Path(data_dir or _LEGACY_DATA_DIR) / "usage_data.json")
        self.group_usage_file = str(
            Path(data_dir or _LEGACY_DATA_DIR) / "group_usage_data.json"
        )
        _migrate_legacy_state_file(Path(self.usage_data_file))
        _migrate_legacy_state_file(Path(self.group_usage_file))
        self.usage_data: Dict[str, Dict[str, int]] = self._load_usage_data()
        self.group_usage_data: Dict[str, Dict[str, int]] = self._load_group_usage_data()
        self.last_use: Dict[str, datetime] = {}
        self.processing_request: Optional[str] = None  # 当前正在处理的请求ID
        self.queue: List[Dict[str, Any]] = []  # 排队列表，存储请求信息
        self.queue_lock = asyncio.Lock()  # 队列锁
        self.request_events: Dict[str, asyncio.Event] = {}  # 请求等待事件
        self.request_counter = 0  # 请求计数器，用于生成唯一ID

    def generate_request_id(self, user_id: str) -> str:
        """生成唯一的请求ID"""
        self.request_counter += 1
        return f"{user_id}_{self.request_counter}_{int(time.time() * 1000)}"

    def check_daily_limit(self, user_id: str) -> bool:
        """检查用户是否达到每日使用限制"""
        if self.max_daily_usage <= 0:  # 0 表示无限制
            return True

        today = datetime.now().strftime("%Y-%m-%d")
        user_usage = self.usage_data.get(user_id, {})
        return user_usage.get(today, 0) < self.max_daily_usage

    def get_group_daily_usage(self, quota_scope: str) -> int:
        """Return the images charged to one group or session during the current day.

        Args:
            quota_scope: Group or session identifier owning the free quota.

        Returns:
            Successful generations charged to that scope today.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return self.group_usage_data.get(quota_scope, {}).get(today, 0)

    def get_group_daily_remaining(self, quota_scope: str) -> int:
        """Return the remaining free quota of one group or session today.

        Args:
            quota_scope: Group or session identifier owning the free quota.

        Returns:
            Number of images that this scope may still generate before midnight.
        """
        return max(
            0,
            self.max_group_daily_usage - self.get_group_daily_usage(quota_scope),
        )

    def check_group_daily_limit(self, quota_scope: str, image_count: int = 1) -> bool:
        """Check whether one group's daily free quota can cover a request.

        Args:
            quota_scope: Group or session identifier owning the free quota.
            image_count: Number of images the request intends to generate.

        Returns:
            Whether the requested images fit within that scope's remaining quota.
        """
        return max(0, image_count) <= self.get_group_daily_remaining(quota_scope)

    def check_cooldown(self, user_id: str) -> bool:
        """检查用户是否在冷却期内"""
        last_time = self.last_use.get(user_id)
        if not last_time:
            return True
        return datetime.now() - last_time > timedelta(seconds=self.cooldown_seconds)

    async def add_to_queue(self, user_id: str) -> tuple[str, bool, int, int]:
        """
        添加请求到队列
        返回: (请求ID, 是否可以立即处理, 队列位置, 预计等待时间)
        """
        request_id = self.generate_request_id(user_id)

        async with self.queue_lock:
            # 如果没有正在处理的请求且队列为空，可以立即处理
            if not self.processing_request and not self.queue:
                self.processing_request = request_id
                logger.info(f"请求 {request_id} (用户 {user_id}) 开始处理（无需排队）")
                return request_id, True, 0, 0

            # 否则加入队列
            request_info = {
                "request_id": request_id,
                "user_id": user_id,
                "timestamp": datetime.now(),
            }
            self.queue.append(request_info)

            # 为请求创建等待事件
            self.request_events[request_id] = asyncio.Event()

            queue_position = len(self.queue)
            # 预计时间 = 当前处理 + 队列前面的请求数 * 8秒
            estimated_time = 8 + (queue_position - 1) * 8

            logger.info(
                f"请求 {request_id} (用户 {user_id}) 加入队列，位置: {queue_position}"
            )
            return request_id, False, queue_position, estimated_time

    async def wait_for_turn(self, request_id: str, timeout: float = 300.0) -> bool:
        """等待轮到该请求。返回 True 表示成功轮到，False 表示超时被丢弃"""
        if request_id not in self.request_events:
            return True
        try:
            await asyncio.wait_for(
                self.request_events[request_id].wait(),
                timeout=timeout,
            )
            del self.request_events[request_id]
            return True
        except asyncio.TimeoutError:
            # 超时：从队列中移除该请求，释放槽位
            await self.cancel_request(request_id)
            return False

    async def cancel_request(self, request_id: str):
        """从队列中取消一个等待中的请求"""
        async with self.queue_lock:
            self.queue = [r for r in self.queue if r["request_id"] != request_id]
            self.request_events.pop(request_id, None)
        logger.info(f"请求 {request_id} 已从队列中取消")

    async def start_next_in_queue(self) -> Optional[str]:
        """开始处理队列中的下一个请求"""
        async with self.queue_lock:
            if self.queue:
                next_request = self.queue.pop(0)
                request_id = next_request["request_id"]
                user_id = next_request["user_id"]
                self.processing_request = request_id

                # 触发请求的等待事件
                if request_id in self.request_events:
                    self.request_events[request_id].set()

                logger.info(f"开始处理队列中的请求: {request_id} (用户 {user_id})")
                return request_id

            self.processing_request = None
            return None

    def finish_processing(self, request_id: str):
        """完成处理请求"""
        if self.processing_request == request_id:
            self.processing_request = None
            logger.info(f"请求 {request_id} 处理完成")

    def _load_usage_data(self) -> Dict[str, Dict[str, int]]:
        """加载使用数据"""
        try:
            if os.path.exists(self.usage_data_file):
                with open(self.usage_data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"已加载使用数据，包含 {len(data)} 个用户的记录")
                    return data
        except Exception as e:
            logger.error(f"加载使用数据失败: {e}")
        return {}

    def _save_usage_data(self):
        """保存使用数据"""
        try:
            os.makedirs(os.path.dirname(self.usage_data_file) or ".", exist_ok=True)
            with open(self.usage_data_file, "w", encoding="utf-8") as f:
                json.dump(self.usage_data, f, ensure_ascii=False, indent=2)
            logger.debug("使用数据已保存")
        except Exception as e:
            logger.error(f"保存使用数据失败: {e}")

    def _load_group_usage_data(self) -> Dict[str, Dict[str, int]]:
        """Load the per-group daily free quota counters."""
        try:
            if os.path.exists(self.group_usage_file):
                with open(self.group_usage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {
                            str(scope): {
                                str(date): int(count)
                                for date, count in (daily or {}).items()
                            }
                            for scope, daily in data.items()
                        }
        except Exception as e:
            logger.error(f"加载群额度数据失败: {e}")
        return {}

    def _save_group_usage_data(self):
        """Persist the per-group daily free quota counters."""
        try:
            os.makedirs(os.path.dirname(self.group_usage_file) or ".", exist_ok=True)
            with open(self.group_usage_file, "w", encoding="utf-8") as f:
                json.dump(self.group_usage_data, f, ensure_ascii=False, indent=2)
            logger.debug("群额度数据已保存")
        except Exception as e:
            logger.error(f"保存群额度数据失败: {e}")

    def record_usage(self, user_id: str, quota_scope: Optional[str] = None):
        """记录一次成功生成。

        Args:
            user_id: 发送者标识，用于出图排行统计。
            quota_scope: 本次出图使用共享 NovelAI Token 时所计入的群/会话额度作用域；
                个人密钥或群授权出图传入 ``None``，不占用每日免费额度。
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if user_id not in self.usage_data:
            self.usage_data[user_id] = {}
        self.usage_data[user_id][today] = self.usage_data[user_id].get(today, 0) + 1
        self.last_use[user_id] = datetime.now()
        self._save_usage_data()  # 保存到文件
        if quota_scope:
            scope_usage = self.group_usage_data.setdefault(quota_scope, {})
            scope_usage[today] = scope_usage.get(today, 0) + 1
            self._save_group_usage_data()

    def get_usage_info(self, user_id: str) -> Dict[str, Any]:
        """获取用户使用信息"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_usage = self.usage_data.get(user_id, {}).get(today, 0)

        last_time = self.last_use.get(user_id)
        cooldown_remaining = 0
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            cooldown_remaining = max(0, self.cooldown_seconds - elapsed)

        # 计算用户在队列中的请求数
        user_requests_in_queue = sum(
            1 for req in self.queue if req["user_id"] == user_id
        )

        return {
            "today_usage": today_usage,
            "max_daily": self.max_daily_usage,
            "cooldown_remaining": cooldown_remaining,
            "user_requests_in_queue": user_requests_in_queue,
            "queue_length": len(self.queue),
        }

    def get_global_statistics(
        self, quota_scope: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取全局使用统计

        Args:
            quota_scope: 需要展示免费额度的群/会话作用域；为 ``None`` 时仅返回全局统计。
        """
        today = datetime.now().strftime("%Y-%m-%d")
        quota_limit = getattr(
            self,
            "max_group_daily_usage",
            Constants.GROUP_DAILY_IMAGE_LIMIT,
        )
        scope_usage = self.get_group_daily_usage(quota_scope) if quota_scope else 0

        # 统计今日总使用次数
        today_total = 0
        total_users = len(self.usage_data)
        active_users_today = 0

        # 统计历史总使用次数
        total_usage = 0
        user_rankings = []

        for user_id, user_data in self.usage_data.items():
            # 今日使用次数
            today_usage = user_data.get(today, 0)
            today_total += today_usage
            if today_usage > 0:
                active_users_today += 1

            # 历史总使用次数
            user_total = sum(user_data.values())
            total_usage += user_total
            user_rankings.append(
                {
                    "user_id": user_id,
                    "today": today_usage,
                    "total": user_total,
                }
            )

        user_rankings.sort(
            key=lambda item: (item["total"], item["today"], item["user_id"]),
            reverse=True,
        )
        daily_user_rankings = sorted(
            (item for item in user_rankings if item["today"] > 0),
            key=lambda item: (item["today"], item["total"], item["user_id"]),
            reverse=True,
        )

        # 获取最近7天的使用数据
        recent_days = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            day_total = 0
            for user_data in self.usage_data.values():
                day_total += user_data.get(date, 0)
            recent_days.append({"date": date, "count": day_total})

        return {
            "today_total": today_total,
            "total_usage": total_usage,
            "total_users": total_users,
            "active_users_today": active_users_today,
            "quota_scope": quota_scope,
            "daily_quota_limit": quota_limit,
            "daily_quota_used": min(scope_usage, quota_limit),
            "daily_quota_remaining": max(0, quota_limit - scope_usage),
            "fallback_active": quota_limit > 0 and scope_usage >= quota_limit,
            "fallback_model": Constants.DAILY_FALLBACK_MODEL,
            "fallback_usage_today": max(0, scope_usage - quota_limit),
            "queue_length": len(self.queue),
            "processing_request": self.processing_request is not None,
            "recent_days": recent_days,
            "user_rankings": user_rankings,
            "daily_user_rankings": daily_user_rankings,
        }

    def cleanup_old_data(self, days_to_keep: int = 30):
        """清理过期的使用数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")

            cleaned_count = 0
            for user_id in list(self.usage_data.keys()):
                user_data = self.usage_data[user_id]
                # 删除过期的日期记录
                for date in list(user_data.keys()):
                    if date < cutoff_str:
                        del user_data[date]
                        cleaned_count += 1

                # 如果用户没有任何记录了，删除用户
                if not user_data:
                    del self.usage_data[user_id]

            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 条过期使用记录")
                self._save_usage_data()

            cleaned_scopes = 0
            for scope in list(self.group_usage_data.keys()):
                scope_usage = self.group_usage_data[scope]
                for date in list(scope_usage.keys()):
                    if date < cutoff_str:
                        del scope_usage[date]
                        cleaned_scopes += 1

                if not scope_usage:
                    del self.group_usage_data[scope]

            if cleaned_scopes > 0:
                logger.info(f"清理了 {cleaned_scopes} 条过期群额度记录")
                self._save_group_usage_data()

        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")


@register(
    "astrbot_plugin_novelai",
    "xxuanzQAQ",
    "NovelAI图像生成插件，支持文生图和图生图功能",
    "1.5.2",
    "https://github.com/xxuanzQAQ/astrbot_plugin_novelai",
)
class NovelAIPlugin(Star):
    """NovelAI图像生成插件"""

    # 配置项模式在 _conf_schema.json 文件中定义

    def _initialize_safe_defaults(self):
        """初始化所有关键属性为安全默认值，确保即使初始化失败也不会出现 AttributeError"""
        # 基础配置
        self.api_backend = "novelai"
        self.api_token = ""
        self.account_tokens: List[dict] = []
        self.newapi_base_url = "http://localhost:3000"
        self.newapi_api_key = ""
        self.ai_video_user_data_path = None
        self.enable_v5 = True
        self.default_model = Constants.DEFAULT_MODEL
        self.default_resolution = Constants.DEFAULT_RESOLUTION
        self.default_steps = Constants.DEFAULT_STEPS
        self.default_scale = Constants.DEFAULT_SCALE
        self.default_sampler = Constants.DEFAULT_SAMPLER
        self.cooldown_seconds = Constants.DEFAULT_COOLDOWN
        self.group_daily_image_limit = Constants.GROUP_DAILY_IMAGE_LIMIT
        self.enable_request_retry = True
        self.request_retry_count = Constants.DEFAULT_RETRY_COUNT
        self.request_retry_max_delay = Constants.DEFAULT_RETRY_MAX_DELAY
        self.img2img_strength = 0.5
        self.img2img_noise = 0.2
        self.prompt_guidance_rescale = 0.0
        self.v4_5_noise_schedule = "karras"
        self.v4_5_variety_boost = False
        self.v4_5_decrisper = False
        self.v4_5_auto_smea = False
        self.v4_5_smea = False
        self.v4_5_smea_dyn = False
        # 审核配置
        self.baidu_audit_api_key = ""
        self.baidu_audit_secret_key = ""
        self.image_audit_enabled = False
        self.audit_failed_save_path = "novelai_audit_failed"
        self.audit_conclusion_type = 1
        self.audit_strategy_id = ""
        self.audit_whitelist_groups = []
        self.audit_whitelist_bots = []

        # Roll图配置
        self.roll_image_count = 5

        # 日志配置
        self.roll_log_level = 3
        self.draw_log_level = 3

        # 撤回配置
        self.auto_recall_enabled = False
        self.auto_recall_delay = 30

        # 代理配置默认值
        self.api_proxy = ""
        self.novelai_proxy = ""

        # 数据结构
        self.last_draw_commands = {}
        # Keep the latest API response bytes available for /看参数. Chat
        # platforms may transcode the preview image before it reaches us.
        self.latest_generated_images = {}
        self.character_database = {}
        self.character_aliases = {}
        self.presets = {}
        self.preset_categories = {}
        self.style_presets = {}
        self.session_drawing_settings = {}

        # 其他配置
        self.whitelist_groups = []
        self.admin_only_commands = set()

    def _create_fallback_cache(self):
        """创建简单的fallback缓存实现"""

        class SimpleCacheManager:
            def __init__(self):
                self._cache = {}

            def get(self, key, default=None):
                item = self._cache.get(key)
                if item is None:
                    return default
                value, timestamp, ttl = item
                timeout = 3600 if ttl is None else max(0.0, float(ttl))
                if time.time() - timestamp >= timeout:
                    self._cache.pop(key, None)
                    return default
                return value

            def set(self, key, value, ttl=None):
                self._cache[key] = (value, time.time(), ttl)

            def delete(self, key):
                self._cache.pop(key, None)

            def clear(self):
                self._cache.clear()

        return SimpleCacheManager()

    def _get_session_identifier(self, event) -> str:
        """Build a stable identifier for the current chat session.

        Args:
            event: AstrBot message event.

        Returns:
            Session identifier isolated by the unified message origin.
        """
        unified_origin = getattr(event, "unified_msg_origin", None)
        if unified_origin:
            return str(unified_origin)
        platform_name = getattr(event, "get_platform_name", lambda: "unknown")()
        sender_id = getattr(event, "get_sender_id", lambda: "unknown")()
        return f"{platform_name}:{sender_id}"

    def _cache_generated_image(self, event, image_data: bytes) -> None:
        """Cache untouched generated bytes for direct metadata extraction."""
        if not image_data:
            return
        session_id = self._get_session_identifier(event)
        cache = getattr(self, "latest_generated_images", None)
        if cache is None:
            cache = self.latest_generated_images = {}
        cache[session_id] = {
            "image_data": image_data,
            "timestamp": time.time(),
        }

    def _get_cached_generated_image(self, event) -> Optional[bytes]:
        """Return the latest untouched generated image for this chat session."""
        cached = getattr(self, "latest_generated_images", {}).get(
            self._get_session_identifier(event)
        )
        if not cached:
            return None
        # Avoid using an unrelated image after a long-lived session has been
        # idle. The cache is intentionally in-memory and is not persisted.
        if time.time() - cached.get("timestamp", 0) > 30 * 60:
            return None
        return cached.get("image_data")

    def _get_session_drawing_settings(self, event) -> dict:
        """Return effective drawing settings for the current session.

        Args:
            event: AstrBot message event.

        Returns:
            Global defaults merged with current-session overrides.
        """
        settings = {
            "model": self.default_model,
            "resolution": self.default_resolution,
            "steps": self.default_steps,
            "scale": self.default_scale,
            "sampler": self.default_sampler,
            "cfg_rescale": self.prompt_guidance_rescale,
            "v4_5_noise_schedule": self.v4_5_noise_schedule,
            "v4_5_variety_boost": self.v4_5_variety_boost,
            "v4_5_decrisper": self.v4_5_decrisper,
            "v4_5_auto_smea": self.v4_5_auto_smea,
            "v4_5_smea": self.v4_5_smea,
            "v4_5_smea_dyn": self.v4_5_smea_dyn,
            "img2img_strength": self.img2img_strength,
            "img2img_noise": self.img2img_noise,
            "roll_count": self.roll_image_count,
            "style_enabled": self.enable_style_preset,
            "style_name": self.selected_style_preset,
        }
        session_id = self._get_session_identifier(event)
        session_overrides = self.session_drawing_settings.get(session_id, {})
        if "steps" in session_overrides:
            session_overrides.pop("steps", None)
            if not session_overrides:
                self.session_drawing_settings.pop(session_id, None)
        settings.update(session_overrides)
        if not getattr(self, "enable_v5", True) and str(
            settings.get("model") or ""
        ).lower().startswith("nai-diffusion-5"):
            settings["model"] = Constants.DAILY_FALLBACK_MODEL
        return settings

    def _set_session_drawing_setting(self, event, name: str, value) -> None:
        """Store one drawing override for the current session.

        Args:
            event: AstrBot message event.
            name: Drawing setting name.
            value: Session-specific setting value.
        """
        if name == "steps":
            logger.warning("已拒绝会话级画图步数覆盖")
            return

        session_id = self._get_session_identifier(event)
        self.session_drawing_settings.setdefault(session_id, {})[name] = value

    def _reset_session_drawing_settings(self, event) -> bool:
        """Remove all drawing overrides for the current session.

        Args:
            event: AstrBot message event.

        Returns:
            Whether the session previously had overrides.
        """
        session_id = self._get_session_identifier(event)
        return self.session_drawing_settings.pop(session_id, None) is not None

    def _get_command_argument(self, event, *command_names: str) -> str:
        """Extract the argument following one of the provided commands.

        Args:
            event: AstrBot message event.
            *command_names: Accepted command names without a slash prefix.

        Returns:
            Command argument text, or the original text when AstrBot stripped the command.
        """
        message_text = event.message_str.strip()
        for command_name in command_names:
            for prefix in (f"/{command_name}", command_name):
                if message_text.startswith(prefix):
                    return message_text[len(prefix) :].strip()
        return message_text

    def _parse_session_float_argument(
        self,
        event,
        *command_names: str,
        minimum: float,
        maximum: float,
    ) -> Tuple[str, Optional[float], Optional[str]]:
        """Parse a bounded finite float used by a session setting command.

        Returns the original argument, the parsed value, and an error kind. An
        empty argument has no error so callers can show the current setting.
        ``nan`` and infinities are rejected explicitly because comparisons alone
        do not reliably reject them.
        """
        value_text = self._get_command_argument(event, *command_names)
        if not value_text:
            return value_text, None, None

        try:
            value = float(value_text)
        except (TypeError, ValueError):
            return value_text, None, "invalid"

        if not math.isfinite(value) or not minimum <= value <= maximum:
            return value_text, None, "range"
        return value_text, value, None

    def __init__(self, context: Context, config: AstrBotConfig):  # 正确的参数
        super().__init__(context)
        self.context = context
        self.config = config  # AstrBotConfig 对象，由 AstrBot 自动传入

        # 首先初始化所有关键属性为安全默认值
        self._initialize_safe_defaults()

        # 初始化管理器
        try:
            self.cache = CacheManager()
            self.http_manager = HttpClientManager()
            self.config_manager = ConfigManager(config)
        except Exception as e:
            logger.error(f"初始化管理器失败: {e}")
            # 使用简单的默认实现
            self.cache = self._create_fallback_cache()
            self.http_manager = None
            self.config_manager = None

        # 从配置读取基础参数 - 使用常量作为默认值
        try:
            configured_backend = (
                str(self.config.get("api_backend", "novelai") or "novelai")
                .strip()
                .lower()
            )
            self.api_backend = (
                configured_backend
                if configured_backend in {"novelai", "newapi"}
                else "novelai"
            )
            self.api_token = self.config.get("api_token", "")
            self.account_tokens = self._normalize_account_tokens(
                self.config.get("account_tokens", [])
            )
            self.newapi_base_url = str(
                self.config.get("newapi_base_url", "http://localhost:3000")
                or "http://localhost:3000"
            ).strip()
            self.newapi_api_key = str(
                self.config.get("newapi_api_key", "") or ""
            ).strip()
            self.enable_v5 = bool(self.config.get("enable_v5", True))
            self.default_model = self.config.get(
                "default_model", Constants.DEFAULT_MODEL
            )
            self.default_resolution = self.config.get(
                "default_resolution", Constants.DEFAULT_RESOLUTION
            )
            self.default_steps = int(
                self.config.get("default_steps", Constants.DEFAULT_STEPS)
            )
            self.default_scale = float(
                self.config.get("default_scale", Constants.DEFAULT_SCALE)
            )
            self.default_sampler = self.config.get(
                "default_sampler", Constants.DEFAULT_SAMPLER
            )
            self.cooldown_seconds = int(
                self.config.get("cooldown_seconds", Constants.DEFAULT_COOLDOWN)
            )
            self.group_daily_image_limit = int(
                self.config.get(
                    "group_daily_image_limit", Constants.GROUP_DAILY_IMAGE_LIMIT
                )
            )
            self.enable_request_retry = bool(
                self.config.get("enable_request_retry", True)
            )
            self.request_retry_count = int(
                self.config.get("request_retry_count", Constants.DEFAULT_RETRY_COUNT)
            )
            self.request_retry_max_delay = float(
                self.config.get(
                    "request_retry_max_delay", Constants.DEFAULT_RETRY_MAX_DELAY
                )
            )
            self.img2img_strength = float(self.config.get("img2img_strength", 0.5))
            self.img2img_noise = float(self.config.get("img2img_noise", 0.2))
            self.prompt_guidance_rescale = float(
                self.config.get("prompt_guidance_rescale", 0.0)
            )
            self.v4_5_noise_schedule = str(
                self.config.get("v4_5_noise_schedule", "karras") or "karras"
            ).lower()
            self.v4_5_variety_boost = bool(self.config.get("v4_5_variety_boost", False))
            self.v4_5_decrisper = bool(self.config.get("v4_5_decrisper", False))
            self.v4_5_auto_smea = bool(self.config.get("v4_5_auto_smea", False))
            self.v4_5_smea = bool(self.config.get("v4_5_smea", False))
            self.v4_5_smea_dyn = bool(self.config.get("v4_5_smea_dyn", False))
        except Exception as e:
            logger.error(f"加载基础配置失败: {e}")
            # 保持默认值

        # 白名单配置
        self.whitelist_groups = self._load_whitelist()

        # 代理配置
        self.image_proxy = self.config.get("image_proxy", "")
        self.api_proxy = self.config.get("api_proxy", "")
        self.novelai_proxy = self.config.get("novelai_proxy", "")  # 画图API专用HTTP代理

        # 预设画风配置
        self.enable_style_preset = bool(self.config.get("enable_style_preset", False))
        self.selected_style_preset = self.config.get("selected_style_preset", "")

        # 日志等级配置
        self.roll_log_level = int(self.config.get("roll_log_level", 3))
        self.draw_log_level = int(self.config.get("draw_log_level", 3))

        # 自动撤回配置（从配置文件读取，覆盖 _initialize_safe_defaults 中的默认值）
        self.auto_recall_enabled = bool(
            self.config.get("enable_auto_recall", False)
        )
        try:
            self.auto_recall_delay = int(self.config.get("auto_recall_delay", 30))
        except (ValueError, TypeError):
            self.auto_recall_delay = 30

        # 管理员命令配置
        self.admin_only_commands = self._load_admin_commands()

        # 自动保存配置
        self.enable_auto_save = bool(self.config.get("enable_auto_save", False))
        self.save_path = self.config.get("save_path", Constants.DEFAULT_SAVE_PATH)

        # 显示参数配置
        self.show_params = bool(self.config.get("show_params", True))

        # 创建保存目录
        if self.enable_auto_save:
            if Utils.create_directory_if_not_exists(self.save_path):
                logger.info(f"自动保存已启用，保存路径: {self.save_path}")
            else:
                logger.error(f"无法创建保存目录: {self.save_path}")
                self.enable_auto_save = False

        # 预设配置 - 从JSON字符串解析
        self._load_presets()
        self._load_preset_categories()
        self._load_style_presets()

        # Register the authenticated AstrBot Plugin Page APIs after preset state loads.
        self.preset_webui = PresetWebUIController(context, self)
        self.preset_webui.register_routes()

        # Runtime state lives in AstrBot's per-plugin data directory.
        self._plugin_data_dir = resolve_plugin_data_dir()

        # Keep the per-user quota unlimited while enforcing one daily image cap per group.
        self.usage_manager = UsageManager(
            0,
            self.cooldown_seconds,
            self.group_daily_image_limit,
            self._plugin_data_dir,
        )

        # 清理过期数据（保留30天）
        self.usage_manager.cleanup_old_data(30)

        # 初始化衣服管理器
        clothing_db_path = os.path.join(
            os.path.dirname(__file__), "clothing_database.txt"
        )
        self.clothing_manager = ClothingManager(clothing_db_path)

        # 完成其余初始化
        self._complete_initialization()

    def _complete_initialization(self):
        """完成其余的初始化工作"""
        try:
            # 角色数据库 - 改为懒加载，提升启动速度
            self._character_db_loaded = False
            self.character_database = {}
            self.character_aliases = {}

            # 添加最后一次画图命令的记录 - 持久化存储
            data_dir = getattr(self, "_plugin_data_dir", None) or resolve_plugin_data_dir()
            self._plugin_data_dir = data_dir
            self.last_draw_commands_file = str(data_dir / "last_draw_commands.json")
            _migrate_legacy_state_file(Path(self.last_draw_commands_file))
            self.last_draw_commands = self._load_last_draw_commands()

            # 使用配置管理器加载配置 - 添加异常处理
            if self.config_manager:
                try:
                    audit_config = self.config_manager.get_baidu_audit_config()
                    self.baidu_audit_api_key = audit_config["api_key"]
                    self.baidu_audit_secret_key = audit_config["secret_key"]
                    self.image_audit_enabled = audit_config["enable"]
                    self.audit_failed_save_path = audit_config["save_path"]
                    self.audit_conclusion_type = audit_config["conclusion_type"]
                    self.audit_strategy_id = audit_config["strategy_id"]
                    self.audit_whitelist_groups = self._normalize_whitelist_items(
                        audit_config["whitelist_groups"]
                    )
                    self.audit_whitelist_bots = self._normalize_whitelist_items(
                        audit_config["whitelist_bots"]
                    )
                except Exception as e:
                    logger.error(f"加载审核配置失败: {e}")

            self._ensure_audit_failed_path()
            self._log_startup_status()

        except Exception as e:
            logger.error(f"完成初始化失败: {e}")

    # 日志等级控制方法
    def log_roll(self, message: str, level: int = 3):
        """Roll图日志输出
        level: 1=最重要, 2=重要, 3=一般, 4=详细, 5=调试
        只有当配置的日志等级 >= 消息等级时才输出
        """
        if self.roll_log_level >= level:
            logger.info(message)

    def log_draw(self, message: str, level: int = 3):
        """画图日志输出
        level: 1=最重要, 2=重要, 3=一般, 4=详细, 5=调试
        只有当配置的日志等级 >= 消息等级时才输出
        """
        if self.draw_log_level >= level:
            logger.info(message)

    def log_error(self, message: str):
        """错误日志始终输出"""
        logger.error(message)

    def log_warning(self, message: str):
        """警告日志始终输出"""
        logger.warning(message)

    @staticmethod
    def _normalize_whitelist_id(value: Any) -> str:
        return str(value or "").strip()

    @classmethod
    def _normalize_whitelist_items(cls, values: Any) -> List[str]:
        if not values:
            return []
        if isinstance(values, str):
            values = [item.strip() for item in values.split(",")]
        return [
            item for value in values if (item := cls._normalize_whitelist_id(value))
        ]

    @staticmethod
    def _current_group_id(event) -> str:
        try:
            if hasattr(event, "get_group_id"):
                group_id = event.get_group_id()
                if group_id:
                    return str(group_id)
            if hasattr(event, "group_id") and event.group_id:
                return str(event.group_id)
            if hasattr(event, "message_obj") and hasattr(event.message_obj, "group_id"):
                group_id = event.message_obj.group_id
                return str(group_id) if group_id else ""
        except Exception:
            pass
        return ""

    @classmethod
    def _resolve_quota_scope(cls, event, user_id: str = "") -> str:
        """Resolve which chat owns the daily free NovelAI image quota.

        Args:
            event: Current message event, or ``None`` for event-less requests.
            user_id: Current sender identifier, used as the last fallback.

        Returns:
            A stable scope key so every group keeps its own daily quota instead
            of sharing one global pool.
        """
        group_id = cls._current_group_id(event) if event is not None else ""
        if group_id and group_id != "0":
            return f"group:{group_id}"

        origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if origin:
            return f"session:{origin}"
        if user_id:
            return f"user:{user_id}"
        return "global"

    @staticmethod
    def _current_bot_id(event) -> str:
        try:
            if hasattr(event, "get_self_id"):
                self_id = event.get_self_id()
                if self_id:
                    return str(self_id)
            if hasattr(event, "message_obj") and hasattr(event.message_obj, "self_id"):
                self_id = event.message_obj.self_id
                return str(self_id) if self_id else ""
        except Exception:
            pass
        return ""

    def _save_list_config(self, key: str, values: List[str]):
        normalized = self._normalize_whitelist_items(values)
        self.config[key] = normalized
        if self.config_manager:
            self.config_manager.update_cache(key, normalized)
        self.config.save_config()

    def _update_audit_whitelist(
        self,
        *,
        attribute_name: str,
        config_key: str,
        value: str,
        noun: str,
        subject: str,
        add: bool,
        audit_on_add: bool = False,
    ) -> str:
        """Add or remove one normalized audit whitelist entry."""
        values = getattr(self, attribute_name)
        if add:
            if value in values:
                return f"❌ {noun} {value} 已在审核白名单中"
            values.append(value)
            self._save_list_config(config_key, values)
            effect = (
                f"{subject}生成的图片将进入审核"
                if audit_on_add
                else f"{subject}的图片将跳过审核直接发送"
            )
            return (
                f"✅ 已将{noun} {value} 添加到审核白名单，当前实例已生效\n\n"
                f"💡 {effect}"
            )

        if value not in values:
            return f"❌ {noun} {value} 不在审核白名单中"
        values.remove(value)
        self._save_list_config(config_key, values)
        effect = (
            f"{subject}生成的图片将不再进入审核"
            if audit_on_add
            else f"{subject}的图片将重新进行审核"
        )
        return (
            f"✅ 已将{noun} {value} 从审核白名单中移除\n\n"
            f"💡 {effect}"
        )

    def _ensure_audit_failed_path(self):
        """创建百度审核失败图片保存目录。"""
        # 创建审核失败图片保存目录
        if self.image_audit_enabled:
            if Utils.create_directory_if_not_exists(self.audit_failed_save_path):
                logger.info(
                    f"图片审核已启用，审核失败图片保存路径: {self.audit_failed_save_path}"
                )
                logger.info(
                    f"审核严格程度: {self.audit_conclusion_type} (1=严格, 2=中等, 3=宽松)"
                )
            else:
                logger.error(
                    f"无法创建审核失败图片保存目录: {self.audit_failed_save_path}"
                )
                self.image_audit_enabled = False

    def _log_startup_status(self):
        """输出插件启动状态。"""
        # 验证配置
        if not str(self.api_token or "").strip():
            logger.warning("未配置 NovelAI Token，每日免费额度不可用")

        if not self.default_model:
            logger.info("NovelAI插件已加载，模型名称未设置，请在管理面板配置")
        else:
            logger.info(f"NovelAI插件已加载，模型: {self.default_model}")

        if self.image_proxy:
            logger.info(f"图片代理: {self.image_proxy}")
        logger.info("每日免费额度后端: NovelAI")
        logger.info(f"个人密钥超额后端: new-api ({self.newapi_base_url})")
        if self.api_proxy:
            logger.info(f"API反代地址: {self.api_proxy}")
        if self.novelai_proxy:
            logger.info(f"画图HTTP代理: {self.novelai_proxy}")

        if self.whitelist_groups:
            logger.info(f"白名单群组: {self.whitelist_groups}")
        else:
            logger.info("未设置群组白名单，所有群组均可使用")

        logger.info(
            f"已加载 {len(self.presets)} 个预设: {', '.join(self.presets.keys())}"
        )
        logger.info(
            f"已加载 {len(self.preset_categories)} 个预设分类: {', '.join(self.preset_categories.keys())}"
        )
        logger.info(
            f"已加载 {len(self.style_presets)} 个预设画风: {', '.join(self.style_presets.keys())}"
        )
        if self.enable_style_preset and self.selected_style_preset:
            logger.info(f"当前启用预设画风: {self.selected_style_preset}")

        if self.admin_only_commands:
            logger.info(f"管理员限制命令: {', '.join(self.admin_only_commands)}")

        # 自动撤回功能状态
        if self.auto_recall_enabled:
            logger.info(f"自动撤回: 已启用，延迟 {self.auto_recall_delay} 秒")
        else:
            logger.info("自动撤回: 已禁用")

    async def _send_image_with_auto_recall(self, event, chain):
        """发送图片并启动自动撤回"""
        try:
            # 如果启用了自动撤回功能且平台支持，尝试直接使用API发送
            if (
                self.auto_recall_enabled
                and self.auto_recall_delay > 0
                and event.get_platform_name() == "aiocqhttp"
            ):
                try:
                    # AiocqhttpMessageEvent 的 bot 属性是 CQHttp 实例，直接调用 call_action
                    if hasattr(event, "bot"):
                        bot = event.bot

                        # 将消息链转换为 OneBot 消息段列表
                        # Image 组件的图片数据在 .file 属性中，格式为 "base64://xxxx" 或 URL/文件路径
                        message_content = []
                        image_ok = True
                        for component in chain:
                            class_name = component.__class__.__name__

                            if class_name == "Plain":
                                # 跳过空白文本
                                if (
                                    hasattr(component, "text")
                                    and component.text.strip()
                                ):
                                    message_content.append(
                                        {
                                            "type": "text",
                                            "data": {"text": component.text},
                                        }
                                    )
                            elif class_name == "At":
                                if hasattr(component, "qq") and component.qq:
                                    message_content.append(
                                        {
                                            "type": "at",
                                            "data": {"qq": str(component.qq)},
                                        }
                                    )
                            elif class_name == "Image":
                                # Image.fromBytes() 生成的对象：file = "base64://xxxx"
                                # Image.fromURL()   生成的对象：file = "http://..."
                                # Image.fromFileSystem() 生成的对象：file = "file:///..."
                                # 优先用 file 属性，其次用 url 属性
                                file_val = getattr(component, "file", "") or ""
                                url_val = getattr(component, "url", "") or ""
                                if file_val:
                                    message_content.append(
                                        {"type": "image", "data": {"file": file_val}}
                                    )
                                    logger.debug(
                                        f"自动撤回: 图片组件转换成功，file前缀: {file_val[:30]}"
                                    )
                                elif url_val:
                                    message_content.append(
                                        {"type": "image", "data": {"file": url_val}}
                                    )
                                else:
                                    logger.warning(
                                        "自动撤回: 图片组件无有效 file/url，回退到普通发送"
                                    )
                                    image_ok = False
                                    break
                            # 其他组件类型（Reply 等）忽略

                        # 检查消息列表是否有效（至少含一个图片）
                        has_image = any(
                            s.get("type") == "image" for s in message_content
                        )
                        if not image_ok or not has_image:
                            logger.warning("自动撤回: 消息中无有效图片，回退到普通发送")
                            yield event.chain_result(chain)
                            return

                        # 使用 bot.call_action 直接发送，获取 message_id
                        group_id = event.get_group_id()
                        sender_id = event.get_sender_id()
                        send_result = None

                        if group_id and str(group_id).isdigit():
                            send_result = await bot.call_action(
                                "send_group_msg",
                                group_id=int(group_id),
                                message=message_content,
                            )
                        elif sender_id and str(sender_id).isdigit():
                            send_result = await bot.call_action(
                                "send_private_msg",
                                user_id=int(sender_id),
                                message=message_content,
                            )

                        # 解析 message_id 并启动撤回任务
                        # OneBot 返回格式: {"message_id": 12345} 或 {"data": {"message_id": 12345}}
                        message_id = None
                        if isinstance(send_result, dict):
                            message_id = send_result.get("message_id") or (
                                send_result.get("data") or {}
                            ).get("message_id")

                        if message_id:
                            logger.info(
                                f"自动撤回: 图片发送成功，将在 {self.auto_recall_delay} 秒后撤回 (ID: {message_id})"
                            )
                            asyncio.create_task(
                                self._auto_recall_single_message(
                                    event, self.auto_recall_delay, message_id
                                )
                            )
                            return
                        else:
                            logger.warning(
                                f"自动撤回: 无法获取消息ID，回退到普通发送。API返回: {send_result}"
                            )

                except Exception as api_error:
                    logger.error(
                        f"自动撤回: 直接API发送失败，回退到普通发送: {api_error}",
                        exc_info=True,
                    )

            # 普通发送方式（未启用自动撤回 / API发送失败时回退）
            yield event.chain_result(chain)

        except Exception as e:
            logger.error(f"发送图片失败: {e}")
            raise

    def _convert_chain_to_cq_code(self, chain):
        """将消息链转换为CQ码格式"""
        try:
            message_parts = []

            for component in chain:
                if hasattr(component, "__class__"):
                    class_name = component.__class__.__name__

                    if "Plain" in class_name:
                        # 纯文本组件
                        if hasattr(component, "text"):
                            message_parts.append(component.text)
                    elif "At" in class_name:
                        # @某人组件
                        if hasattr(component, "qq"):
                            message_parts.append(f"[CQ:at,qq={component.qq}]")
                    elif "Image" in class_name:
                        # 图片组件
                        try:
                            # 调试：打印组件的所有属性
                            logger.debug(f"图片组件属性: {dir(component)}")

                            # 尝试获取图片数据
                            image_data = None

                            # 尝试多种可能的属性名
                            for attr_name in [
                                "_data",
                                "data",
                                "_bytes",
                                "bytes",
                                "file_data",
                                "content",
                            ]:
                                if hasattr(component, attr_name):
                                    attr_value = getattr(component, attr_name)
                                    if attr_value and isinstance(attr_value, bytes):
                                        image_data = attr_value
                                        logger.debug(
                                            f"找到图片数据，属性名: {attr_name}, 大小: {len(image_data)} bytes"
                                        )
                                        break

                            if image_data:
                                # 将图片数据转换为base64
                                image_base64 = base64.b64encode(image_data).decode(
                                    "utf-8"
                                )
                                message_parts.append(
                                    f"[CQ:image,file=base64://{image_base64}]"
                                )
                                logger.info(
                                    f"成功转换图片组件，数据大小: {len(image_data)} bytes"
                                )
                            elif hasattr(component, "url") and component.url:
                                message_parts.append(f"[CQ:image,file={component.url}]")
                                logger.info(f"使用图片URL: {component.url}")
                            elif hasattr(component, "file") and component.file:
                                message_parts.append(
                                    f"[CQ:image,file={component.file}]"
                                )
                                logger.info(f"使用图片文件路径: {component.file}")
                            else:
                                logger.warning(
                                    f"图片组件缺少数据，可用属性: {[attr for attr in dir(component) if not attr.startswith('_')]}"
                                )

                        except Exception as img_error:
                            logger.error(f"处理图片组件失败: {img_error}")

            result = "".join(message_parts)
            logger.debug(f"消息链转换为CQ码完成，长度: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"转换消息链为CQ码失败: {e}")
            return "图片生成完成"

    async def _auto_recall_message_by_id(
        self, event, delay_seconds: int, message_id_list: list
    ):
        """根据消息ID列表自动撤回消息"""
        try:
            # 等待指定的延迟时间
            await asyncio.sleep(delay_seconds)

            # 检查平台类型
            platform_name = event.get_platform_name()
            if platform_name != "aiocqhttp":
                logger.info(f"平台 {platform_name} 不支持自动撤回功能")
                return

            # 使用 event.bot.call_action 撤回消息（AiocqhttpMessageEvent 的正确调用方式）
            if hasattr(event, "bot"):
                bot = event.bot

                success_count = 0
                for message_id in message_id_list:
                    try:
                        ret = await bot.call_action("delete_msg", message_id=message_id)
                        logger.info(
                            f"自动撤回消息成功: 消息ID {message_id}, 结果: {ret}"
                        )
                        success_count += 1
                    except Exception as recall_error:
                        logger.debug(f"撤回消息ID {message_id} 失败: {recall_error}")
                        continue

                if success_count > 0:
                    logger.info(f"自动撤回完成: 成功撤回 {success_count} 条消息")
                else:
                    logger.warning(
                        f"自动撤回失败: 尝试撤回的所有消息ID都失败: {message_id_list}"
                    )

        except Exception as e:
            logger.error(f"自动撤回功能失败: {e}")

    async def _auto_recall_single_message(
        self, event, delay_seconds: int, message_id: int
    ):
        """撤回单个消息"""
        try:
            # 等待指定的延迟时间
            await asyncio.sleep(delay_seconds)

            # 检查平台类型
            platform_name = event.get_platform_name()
            if platform_name != "aiocqhttp":
                logger.info(f"平台 {platform_name} 不支持自动撤回功能")
                return

            # 使用 event.bot.call_action 撤回消息（AiocqhttpMessageEvent 的正确调用方式）
            if hasattr(event, "bot"):
                bot = event.bot

                try:
                    result = await bot.call_action("delete_msg", message_id=message_id)
                    logger.info(f"自动撤回: 图片消息已撤回 (ID: {message_id})")
                    logger.debug(f"撤回API返回结果: {result}")
                except Exception as recall_error:
                    error_msg = str(recall_error)
                    logger.error(f"自动撤回: 撤回失败 (ID: {message_id}) - {error_msg}")

                    # 分析常见的撤回失败原因
                    if "权限" in error_msg or "permission" in error_msg.lower():
                        logger.warning(
                            "撤回失败可能原因: 机器人缺少撤回消息权限，请检查群管理员设置"
                        )
                    elif "时间" in error_msg or "time" in error_msg.lower():
                        logger.warning(
                            "撤回失败可能原因: 消息发送时间过长，超过撤回时间限制"
                        )
                    elif "不存在" in error_msg or "not found" in error_msg.lower():
                        logger.warning("撤回失败可能原因: 消息不存在或已被撤回")
                    else:
                        logger.warning(
                            "撤回失败，建议检查: 1.机器人权限 2.撤回时间设置 3.NCQQ连接状态"
                        )

        except Exception as e:
            logger.error(f"自动撤回功能失败: {e}")

    async def _send_status_message_with_auto_recall(
        self, event, message: str, delay: int = 10
    ):
        """发送状态消息（暂不支持撤回，避免触发AI对话）"""
        try:
            # 直接使用普通发送方式
            # 经过多次尝试，API直接发送状态消息都会触发AI对话
            # 无论使用什么格式（纯文本、消息链、特殊标记等）
            logger.debug(
                f"发送状态消息: {message} (延迟参数: {delay}秒，API发送会触发AI对话，功能已禁用)"
            )
            yield event.plain_result(message)

        except Exception as e:
            logger.error(f"发送状态消息失败: {e}")
            yield event.plain_result(message)

    def process_character_prompt(self, prompt):
        """Preserve character text and handle only explicit random tokens.

        Character database lookup is intentionally excluded from image generation:
        a name such as ``hsin`` must not be replaced by any canonical tag.
        """
        if not prompt:
            return prompt

        # 处理随机人物（从人物预设中随机选择）
        if "随机人物" in prompt:
            random_character = self._get_random_character_preset()
            if random_character:
                prompt = prompt.replace("随机人物", random_character)
                logger.info(f"🎲 随机选择人物预设: {random_character}")
            else:
                logger.warning("⚠️ 没有可用的人物预设，随机人物功能无效")
                prompt = prompt.replace("随机人物", "")

        # 处理随机角色（从Danbooru角色数据库中随机选择）
        if "随机角色" in prompt:
            random_character_tag = self._get_random_character_from_database()
            if random_character_tag:
                prompt = prompt.replace("随机角色", random_character_tag)
                logger.info(f"🎲 随机选择角色标签: {random_character_tag}")
            else:
                logger.warning("⚠️ 角色数据库未加载或为空，随机角色功能无效")
                prompt = prompt.replace("随机角色", "")

        return prompt

    def _process_single_character_word(self, word):
        """处理单个词，按优先级进行替换"""
        if not word:
            return word

        # 1. 检查是否是现有预设（最高优先级）
        if word in self.presets:
            return word

        # 2. 检查是否在角色数据库中
        character_tag = self._find_character_tag(word)
        if character_tag:
            logger.info(f"角色识别: {word} -> {character_tag}")
            return character_tag

        # 3. Preserve unknown Chinese text unchanged.
        if any("\u4e00" <= c <= "\u9fff" for c in word):
            return word

        # 4. 其他情况保持原样
        return word

    def _get_random_character_preset(self):
        """从人物预设中随机选择一个"""
        try:
            # 获取人物分类的预设列表（preset_categories的结构是 {"人物": [预设名列表]}）
            character_presets = self.preset_categories.get("人物", [])
            if character_presets:
                # 随机选择一个预设名
                preset_name = random.choice(character_presets)
                # 返回预设的内容
                return self.presets.get(preset_name, preset_name)
            else:
                # 如果没有人物分类或人物分类为空，从所有预设中查找可能的人物预设
                # 这里可以根据预设名称或内容进行简单的启发式判断
                all_presets = list(self.presets.keys())
                if all_presets:
                    # 简单随机选择（可以后续优化为更智能的筛选）
                    preset_name = random.choice(all_presets)
                    return self.presets.get(preset_name, preset_name)
            return None
        except Exception as e:
            logger.error(f"获取随机人物预设失败: {e}")
            return None

    def _get_random_character_from_database(self):
        """从Danbooru角色数据库中随机选择一个角色标签"""
        try:
            self._ensure_character_database_loaded()
            if not hasattr(self, "character_database") or not self.character_database:
                logger.warning("角色数据库未初始化或为空")
                return None

            # 从角色数据库中随机选择一个角色
            character_tags = list(self.character_database.keys())
            if character_tags:
                selected_tag = random.choice(character_tags)
                character_info = self.character_database[selected_tag]

                # 记录选择的角色信息（用于调试）
                chinese_name = character_info.get("chinese_name", "")
                series = character_info.get("series", "")
                logger.debug(
                    f"随机选择角色: {selected_tag} ({chinese_name}) - {series}"
                )

                return selected_tag
            return None
        except Exception as e:
            logger.error(f"从角色数据库获取随机角色失败: {e}")
            return None

    # 配置模式由 _conf_schema.json 文件定义，无需 get_config_schema 方法

    def validate_config(self, config_data: dict) -> dict:
        """Validate plugin configuration.

        Args:
            config_data: Configuration values to validate.

        Returns:
            Validation errors keyed by configuration field.
        """
        errors = {}

        api_backend = str(config_data.get("api_backend", "novelai") or "").strip()
        if api_backend not in {"novelai", "newapi"}:
            errors["api_backend"] = "请选择受支持的图像 API 后端"
        if not str(config_data.get("api_token", "") or "").strip():
            errors["api_token"] = "NovelAI Token不能为空"

        newapi_base_url = str(
            config_data.get("newapi_base_url", "http://localhost:3000") or ""
        ).strip()
        if not newapi_base_url.startswith(("http://", "https://")):
            errors["newapi_base_url"] = "new-api 地址必须以 http:// 或 https:// 开头"
        if api_backend == "newapi":
            if not str(config_data.get("newapi_api_key", "") or "").strip():
                errors["newapi_api_key"] = "new-api API Key不能为空"

        if not isinstance(config_data.get("enable_v5", True), bool):
            errors["enable_v5"] = "V5 画图开关必须为开启或关闭"

        # 验证百度审核配置
        audit_api_key = config_data.get("baidu_audit_api_key", "").strip()
        audit_secret_key = config_data.get("baidu_audit_secret_key", "").strip()

        if audit_api_key and not audit_secret_key:
            errors["baidu_audit_secret_key"] = "设置了API Key但未设置Secret Key"
        elif audit_secret_key and not audit_api_key:
            errors["baidu_audit_api_key"] = "设置了Secret Key但未设置API Key"

        # 验证审核失败保存路径
        audit_path = config_data.get("audit_failed_save_path", "").strip()
        if audit_path and not Utils.create_directory_if_not_exists(audit_path):
            errors["audit_failed_save_path"] = "无效的保存路径"

        # 验证审核严格程度
        conclusion_type = config_data.get("audit_conclusion_type", 2)
        if (
            not isinstance(conclusion_type, int)
            or conclusion_type < 1
            or conclusion_type > 3
        ):
            errors["audit_conclusion_type"] = "审核严格程度必须在1-3之间"

        # 验证数值范围
        steps = config_data.get("default_steps", 28)
        if not isinstance(steps, int) or steps < 1 or steps > 50:
            errors["default_steps"] = "步数必须在1-50之间"

        group_daily_image_limit = config_data.get(
            "group_daily_image_limit", Constants.GROUP_DAILY_IMAGE_LIMIT
        )
        if (
            isinstance(group_daily_image_limit, bool)
            or not isinstance(group_daily_image_limit, int)
            or group_daily_image_limit < 0
        ):
            errors["group_daily_image_limit"] = (
                "每群每日免费额度必须是大于或等于0的整数"
            )

        model_name = config_data.get("default_model", Constants.DEFAULT_MODEL)
        if model_name not in Constants.IMAGE_MODELS.values():
            errors["default_model"] = "请选择受支持的 NovelAI Diffusion 模型"

        resolution = config_data.get("default_resolution", Constants.DEFAULT_RESOLUTION)
        if resolution not in Constants.IMAGE_RESOLUTIONS.values():
            errors["default_resolution"] = "请选择受支持的分辨率"

        sampler = config_data.get("default_sampler", Constants.DEFAULT_SAMPLER)
        if sampler not in Constants.IMAGE_SAMPLERS.values():
            errors["default_sampler"] = "请选择受支持的采样器"

        scale = config_data.get("default_scale", Constants.DEFAULT_SCALE)
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or scale < 1
            or scale > 15
        ):
            errors["default_scale"] = "权重必须在1-15之间"

        cfg_rescale = config_data.get("prompt_guidance_rescale", 0.0)
        if (
            isinstance(cfg_rescale, bool)
            or not isinstance(cfg_rescale, (int, float))
            or cfg_rescale < 0
            or cfg_rescale > 1
        ):
            errors["prompt_guidance_rescale"] = "V4.5 引导缩放必须在0.0-1.0之间"

        noise_schedule = str(
            config_data.get("v4_5_noise_schedule", "karras") or ""
        ).lower()
        if noise_schedule not in Constants.V4_5_NOISE_SCHEDULES:
            errors["v4_5_noise_schedule"] = "请选择受支持的 V4.5 Noise Schedule"

        for field_name in (
            "v4_5_variety_boost",
            "v4_5_decrisper",
            "v4_5_auto_smea",
            "v4_5_smea",
            "v4_5_smea_dyn",
        ):
            if not isinstance(config_data.get(field_name, False), bool):
                errors[field_name] = "该 V4.5 选项必须为开启或关闭"

        return errors

    def on_config_changed(self, new_config: dict):
        """Apply updated plugin configuration.

        Args:
            new_config: Newly saved configuration values.
        """
        # 更新实例变量
        configured_backend = (
            str(new_config.get("api_backend", "novelai") or "novelai").strip().lower()
        )
        self.api_backend = (
            configured_backend
            if configured_backend in {"novelai", "newapi"}
            else "novelai"
        )
        self.api_token = new_config.get("api_token", "")
        self.account_tokens = self._normalize_account_tokens(
            new_config.get("account_tokens", [])
        )
        self.newapi_base_url = str(
            new_config.get("newapi_base_url", "http://localhost:3000")
            or "http://localhost:3000"
        ).strip()
        self.newapi_api_key = str(new_config.get("newapi_api_key", "") or "").strip()
        self.enable_v5 = bool(new_config.get("enable_v5", True))
        self.default_model = new_config.get("default_model", Constants.DEFAULT_MODEL)
        self.default_resolution = new_config.get(
            "default_resolution", Constants.DEFAULT_RESOLUTION
        )
        self.default_steps = int(
            new_config.get("default_steps", Constants.DEFAULT_STEPS)
        )
        self.default_scale = float(
            new_config.get("default_scale", Constants.DEFAULT_SCALE)
        )
        self.default_sampler = new_config.get(
            "default_sampler", Constants.DEFAULT_SAMPLER
        )
        try:
            self.cooldown_seconds = int(new_config.get("cooldown_seconds", 10))
        except (ValueError, TypeError):
            self.cooldown_seconds = 10
        self.group_daily_image_limit = int(
            new_config.get("group_daily_image_limit", Constants.GROUP_DAILY_IMAGE_LIMIT)
        )
        # 更新请求失败重试配置
        self.enable_request_retry = bool(new_config.get("enable_request_retry", True))
        try:
            self.request_retry_count = max(
                0,
                int(
                    new_config.get("request_retry_count", Constants.DEFAULT_RETRY_COUNT)
                ),
            )
        except (ValueError, TypeError):
            self.request_retry_count = Constants.DEFAULT_RETRY_COUNT
        try:
            self.request_retry_max_delay = max(
                0.0,
                float(
                    new_config.get(
                        "request_retry_max_delay", Constants.DEFAULT_RETRY_MAX_DELAY
                    )
                ),
            )
        except (ValueError, TypeError):
            self.request_retry_max_delay = Constants.DEFAULT_RETRY_MAX_DELAY
        # 更新日志等级配置
        self.roll_log_level = int(new_config.get("roll_log_level", 3))
        self.draw_log_level = int(new_config.get("draw_log_level", 3))

        # 更新roll图配置
        try:
            self.roll_image_count = int(new_config.get("roll_image_count", 5))
        except (ValueError, TypeError):
            self.roll_image_count = 5

        # 更新自动撤回配置
        self.auto_recall_enabled = bool(new_config.get("enable_auto_recall", False))
        try:
            self.auto_recall_delay = int(new_config.get("auto_recall_delay", 30))
        except (ValueError, TypeError):
            self.auto_recall_delay = 30

        # 更新预设画风配置
        self.enable_style_preset = bool(new_config.get("enable_style_preset", False))
        self.selected_style_preset = new_config.get("selected_style_preset", "")

        # 更新管理员命令配置
        self.admin_only_commands = self._load_admin_commands()
        self.enable_auto_save = bool(new_config.get("enable_auto_save", False))
        self.show_params = bool(new_config.get("show_params", True))
        self.image_proxy = new_config.get("image_proxy", "")
        self.api_proxy = new_config.get("api_proxy", "")
        self.novelai_proxy = new_config.get("novelai_proxy", "")  # 画图API专用HTTP代理

        # 图生图与 V4.5 兼容参数
        self.img2img_strength = float(new_config.get("img2img_strength", 0.5))
        self.img2img_noise = float(new_config.get("img2img_noise", 0.2))
        self.prompt_guidance_rescale = float(
            new_config.get("prompt_guidance_rescale", 0.0)
        )
        self.v4_5_noise_schedule = str(
            new_config.get("v4_5_noise_schedule", "karras") or "karras"
        ).lower()
        self.v4_5_variety_boost = bool(new_config.get("v4_5_variety_boost", False))
        self.v4_5_decrisper = bool(new_config.get("v4_5_decrisper", False))
        self.v4_5_auto_smea = bool(new_config.get("v4_5_auto_smea", False))
        self.v4_5_smea = bool(new_config.get("v4_5_smea", False))
        self.v4_5_smea_dyn = bool(new_config.get("v4_5_smea_dyn", False))

        # 记录配置更新
        logger.info("📝 配置已更新，所有设置立即生效")
        if hasattr(self, "auto_recall_enabled"):
            if self.auto_recall_enabled:
                logger.info(f"🔄 自动撤回: 已启用，延迟 {self.auto_recall_delay} 秒")
            else:
                logger.info("🔄 自动撤回: 已禁用")

        if (
            hasattr(self, "enable_style_preset")
            and self.enable_style_preset
            and self.selected_style_preset
        ):
            logger.info(f"🎨 预设画风: {self.selected_style_preset}")

        if hasattr(self, "roll_image_count"):
            logger.info(f"🎲 Roll图数量: {self.roll_image_count} 张")

        # 审核配置更新
        self.baidu_audit_api_key = new_config.get("baidu_audit_api_key", "")
        self.baidu_audit_secret_key = new_config.get("baidu_audit_secret_key", "")
        self.image_audit_enabled = bool(
            new_config.get("enable_image_audit", False)
        )
        self.audit_failed_save_path = new_config.get(
            "audit_failed_save_path", "novelai_audit_failed"
        )
        self.audit_conclusion_type = int(new_config.get("audit_conclusion_type", 2))
        self.audit_strategy_id = new_config.get("audit_strategy_id", "")
        self.audit_whitelist_groups = self._normalize_whitelist_items(
            new_config.get("audit_whitelist_groups", [])
        )
        self.audit_whitelist_bots = self._normalize_whitelist_items(
            new_config.get("audit_whitelist_bots", [])
        )

        self._ensure_audit_failed_path()

        # 更新使用量限制
        if hasattr(self, "usage_manager"):
            self.usage_manager.max_daily_usage = 0  # 0 表示无限制
            self.usage_manager.max_group_daily_usage = self.group_daily_image_limit

        # 预设配置需要立即刷新，避免管理面板保存后仍使用旧数据。
        self.presets = Utils.safe_json_loads(new_config.get("presets", {}), {})
        self.preset_categories = Utils.safe_json_loads(
            new_config.get("preset_categories", {}), {}
        )
        self.style_presets = normalize_style_presets(
            Utils.safe_json_loads(new_config.get("style_presets", {}), {})
        )
        logger.info("配置已更新")

    async def cleanup_resources(self):
        """清理资源"""
        try:
            await self.http_manager.close_all()
            self.cache.clear()
            self.config_manager.clear_cache()
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")

    def _load_whitelist(self) -> Set[str]:
        """加载白名单群组"""
        whitelist_str = self.config.get("whitelist_groups", "")
        if not whitelist_str:
            return set()

        # 解析逗号分隔的群号
        groups = set()
        for group_id in whitelist_str.split(","):
            group_id = group_id.strip()
            if group_id:
                groups.add(group_id)

        return groups

    def _load_admin_commands(self) -> Set[str]:
        """加载仅管理员可用的命令"""
        admin_commands_str = self.config.get("admin_only_commands", "")
        logger.error(f"🔍 [管理员命令加载] 原始配置字符串: '{admin_commands_str}'")

        if not admin_commands_str:
            logger.error(f"⚠️ [管理员命令加载] 配置为空，返回空集合")
            return set()

        # 解析逗号分隔的命令
        commands = set()
        for cmd in admin_commands_str.split(","):
            cmd = cmd.strip()
            if cmd:
                commands.add(cmd)
                logger.error(f"✅ [管理员命令加载] 添加命令: '{cmd}'")

        logger.error(f"🔍 [管理员命令加载] 最终命令集合: {commands}")
        return commands

    def _check_whitelist(self, event) -> bool:
        """检查插件原有的群白名单。

        NAI 群授权是“共享个人密钥”的路由，不是群白名单；未开启群授权时，
        仍保持插件原有的群聊使用行为。
        """
        group_id = self._current_group_id(event)
        if not group_id or group_id == "0" or not self.whitelist_groups:
            return True
        return group_id in self.whitelist_groups

    def _check_admin_permission(self, event, command_name: str) -> bool:
        """检查是否需要管理员权限"""
        # 添加详细调试日志
        logger.error(f"🔍 [权限检查] 开始检查命令: {command_name}")
        logger.error(f"🔍 [权限检查] 管理员命令列表: {list(self.admin_only_commands)}")

        # 如果命令不在管理员限制列表中，所有人都可以使用
        if command_name not in self.admin_only_commands:
            logger.error(
                f"⚠️ [权限检查] 命令 {command_name} 不在管理员限制列表中，允许所有人使用"
            )
            return True

        logger.error(f"🔒 [权限检查] 命令 {command_name} 需要管理员权限")

        # 检查是否为管理员
        try:
            # 获取发送者ID
            sender_id = str(event.get_sender_id())
            logger.error(f"🔍 [权限检查] 发送者ID: {sender_id}")

            # 从 AstrBot 主配置中获取管理员列表
            # 获取 AstrBot 的主配置（不是插件配置）
            astrbot_config = self.context.get_config()
            admins = astrbot_config.get("admins_id", [])
            logger.error(f"🔍 [权限检查] 主配置管理员列表: {admins}")

            # 确保管理员ID都是字符串类型进行比较
            admins_str = [str(admin_id) for admin_id in admins]
            logger.error(f"🔍 [权限检查] 字符串化管理员列表: {admins_str}")

            # 检查发送者是否在管理员列表中
            if sender_id in admins_str:
                logger.error(
                    f"✅ [权限检查] 用户 {sender_id} 是管理员，允许使用命令 {command_name}"
                )
                return True

            # 不是管理员
            logger.error(
                f"❌ [权限检查] 用户 {sender_id} 不是管理员，拒绝使用命令 {command_name}"
            )
            return False

        except Exception as e:
            logger.error(f"💥 [权限检查] 检查管理员权限时出错: {e}")
            return False

    def _is_user_admin(self, event):
        """检查用户是否是管理员（不检查特定命令权限）"""
        try:
            is_admin = getattr(event, "is_admin", None)
            if callable(is_admin):
                return bool(is_admin())
            if is_admin is not None:
                return bool(is_admin)

            # 获取发送者ID
            sender_id = str(event.get_sender_id())

            # 从 AstrBot 主配置中获取管理员列表
            astrbot_config = self.context.get_config()
            admins = astrbot_config.get("admins_id", [])

            # 确保管理员ID都是字符串类型进行比较
            admins_str = [str(admin_id) for admin_id in admins]

            # 检查发送者是否在管理员列表中
            return sender_id in admins_str

        except Exception as e:
            logger.error(f"检查用户管理员身份时出错: {e}")
            return False

    async def _save_image_locally(
        self,
        image_data: bytes,
        image_type: str = "txt2img",
    ) -> Optional[str]:
        """Save untouched generated image bytes to a PNG file.

        Args:
            image_data: Original bytes returned by the image API.
            image_type: Prefix used to identify the generation workflow.

        Returns:
            Absolute path to the saved PNG, or ``None`` when saving is disabled
            or fails.
        """
        if not self.enable_auto_save:
            return None

        try:
            save_dir = Path(self.save_path).expanduser()
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{image_type}_{timestamp}.png"
            filepath = save_dir / filename
            filepath.write_bytes(image_data)

            logger.info(f"图片已保存: {filepath}")
            return str(filepath.resolve())

        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            return None

    def _ensure_config_str(self, key: str, value) -> None:
        """确保配置项在内存和文件中均为 JSON 字符串（兼容历史遗留的 dict 格式）"""
        if isinstance(value, dict):
            logger.info(f"[配置迁移] {key} 存储格式为 dict，自动迁移为 JSON 字符串")
            self.config[key] = json.dumps(value, ensure_ascii=False)
            self.config.save_config()

    def _load_presets(self):
        """从配置加载预设"""
        if "presets" not in self.config:
            # 第一次使用，初始化默认预设
            default_presets = {
                "动漫": "masterpiece, best quality, anime style, official art, 1girl, solo, looking at viewer",
                "写实": "photorealistic, realistic, detailed, high quality, professional photography",
                "水彩": "watercolor, soft colors, artistic, traditional art, delicate brushstrokes",
                "油画": "oil painting, classical art, rich colors, detailed brushwork, fine art",
            }
            self.config["presets"] = json.dumps(default_presets, ensure_ascii=False)
            self.config.save_config()
            self.presets = default_presets
            logger.info("首次使用，已初始化默认预设")
        else:
            presets_value = self.config.get("presets", {})
            # 兼容历史遗留 dict 格式：检测并迁移
            self._ensure_config_str("presets", presets_value)
            self.presets = Utils.safe_json_loads(presets_value, {})
            if self.presets:
                logger.info(f"成功加载预设配置: {len(self.presets)} 个预设")
            else:
                logger.warning("预设配置损坏，已清空预设列表")

    def _load_preset_categories(self):
        """从配置加载预设分类"""
        default_categories = {name: [] for name in Constants.DEFAULT_CATEGORIES.keys()}

        if "preset_categories" not in self.config:
            # 第一次使用，初始化默认分类
            self.config["preset_categories"] = json.dumps(
                default_categories, ensure_ascii=False
            )
            self.config.save_config()
            self.preset_categories = default_categories
            logger.info("首次使用，已初始化默认预设分类")
        else:
            categories_value = self.config.get("preset_categories", default_categories)
            # 兼容历史遗留 dict 格式：检测并迁移
            self._ensure_config_str("preset_categories", categories_value)
            self.preset_categories = Utils.safe_json_loads(
                categories_value, default_categories
            )
            if self.preset_categories:
                logger.info(
                    f"成功加载预设分类配置: {len(self.preset_categories)} 个分类"
                )
            else:
                self.preset_categories = default_categories
                logger.warning("预设分类配置损坏，已重置为默认分类")

    def _load_style_presets(self):
        """从配置加载预设画风"""
        if "style_presets" not in self.config:
            # 第一次使用，初始化默认画风
            default_styles = {
                "清新淡雅": {
                    "positive_prompt": "soft lighting, pastel colors, gentle atmosphere, elegant composition",
                    "negative_prompt": "harsh lighting, oversaturated colors, heavy shadows, cluttered composition",
                    "preview_image": "",
                },
                "赛博朋克": {
                    "positive_prompt": "cyberpunk, neon lights, futuristic, dark atmosphere, high contrast",
                    "negative_prompt": "daylight, pastoral scenery, muted colors, historical clothing",
                    "preview_image": "",
                },
                "古风水墨": {
                    "positive_prompt": "traditional chinese painting, ink wash, elegant brushstrokes, classical style",
                    "negative_prompt": "photorealistic, 3d render, neon colors, modern city",
                    "preview_image": "",
                },
                "梦幻唯美": {
                    "positive_prompt": "dreamy atmosphere, soft focus, magical lighting, ethereal beauty",
                    "negative_prompt": "harsh shadows, dull colors, gritty texture, documentary photography",
                    "preview_image": "",
                },
            }
            self.config["style_presets"] = json.dumps(
                default_styles, ensure_ascii=False
            )
            self.config.save_config()
            self.style_presets = default_styles
            logger.info("首次使用，已初始化默认画风")
        else:
            style_presets_value = self.config.get("style_presets", {})
            # 兼容历史遗留 dict 格式：检测并迁移
            self._ensure_config_str("style_presets", style_presets_value)
            parsed_styles = Utils.safe_json_loads(style_presets_value, {})
            self.style_presets = normalize_style_presets(parsed_styles)
            if self.style_presets != parsed_styles:
                self.config["style_presets"] = json.dumps(
                    self.style_presets, ensure_ascii=False
                )
                self.config.save_config()
                logger.info("[配置迁移] 画风已升级为正面词、负面词和示例图结构")
            if self.style_presets:
                logger.info(f"成功加载画风配置: {len(self.style_presets)} 个画风")
            else:
                logger.warning("画风配置损坏，已清空画风列表")

    def save_presets(self):
        """保存预设到配置文件"""
        try:
            self.config["presets"] = json.dumps(self.presets, ensure_ascii=False)
            self.config.save_config()  # 使用 AstrBotConfig 的保存方法
            logger.info(f"预设已保存到配置文件，共 {len(self.presets)} 个预设")
        except Exception as e:
            logger.error(f"保存预设失败: {e}")

    def save_preset_categories(self):
        """保存预设分类到配置文件"""
        try:
            self.config["preset_categories"] = json.dumps(
                self.preset_categories, ensure_ascii=False
            )
            self.config.save_config()
            logger.info(
                f"预设分类已保存到配置文件，共 {len(self.preset_categories)} 个分类"
            )
        except Exception as e:
            logger.error(f"保存预设分类失败: {e}")

    def save_style_presets(self):
        """保存预设画风到配置文件"""
        try:
            self.config["style_presets"] = json.dumps(
                self.style_presets, ensure_ascii=False
            )
            self.config.save_config()
            logger.info(
                f"预设画风已保存到配置文件，共 {len(self.style_presets)} 个画风"
            )
        except Exception as e:
            logger.error(f"保存预设画风失败: {e}")

    def add_preset_to_category(self, preset_name: str, category: str):
        """将预设添加到指定分类"""
        if category not in self.preset_categories:
            self.preset_categories[category] = []

        # 从其他分类中移除（如果存在）
        for _, presets in self.preset_categories.items():
            if preset_name in presets:
                presets.remove(preset_name)

        # 添加到新分类
        if preset_name not in self.preset_categories[category]:
            self.preset_categories[category].append(preset_name)

        self.save_preset_categories()

    def remove_preset_from_categories(self, preset_name: str):
        """从所有分类中移除预设"""
        for _, presets in self.preset_categories.items():
            if preset_name in presets:
                presets.remove(preset_name)
        self.save_preset_categories()

    def get_presets_by_category(self, category: str) -> Dict[str, str]:
        """获取指定分类的预设"""
        if category not in self.preset_categories:
            return {}

        category_presets = {}
        for preset_name in self.preset_categories[category]:
            if preset_name in self.presets:
                category_presets[preset_name] = self.presets[preset_name]

        return category_presets

    def get_uncategorized_presets(self) -> Dict[str, str]:
        """获取未分类的预设"""
        categorized_presets = set()
        for presets in self.preset_categories.values():
            categorized_presets.update(presets)

        uncategorized = {}
        for preset_name, preset_content in self.presets.items():
            if preset_name not in categorized_presets:
                uncategorized[preset_name] = preset_content

        return uncategorized

    def _load_last_draw_commands(self) -> Dict[str, Dict[str, Any]]:
        """加载上次画图命令记录"""
        try:
            if os.path.exists(self.last_draw_commands_file):
                with open(self.last_draw_commands_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"已加载 {len(data)} 个用户的画图命令记录")
                    return data
        except Exception as e:
            logger.error(f"加载画图命令记录失败: {e}")
        return {}

    def _ensure_character_database_loaded(self):
        """确保角色数据库已加载（懒加载）"""
        if not self._character_db_loaded:
            self._load_character_database()
            self._character_db_loaded = True

    def _load_character_database(self):
        """加载角色数据库"""
        try:
            csv_path = os.path.join(os.path.dirname(__file__), "danbooru角色标签.csv")
            if not os.path.exists(csv_path):
                logger.warning("角色数据库文件不存在: danbooru角色标签.csv")
                return

            logger.info("正在加载角色数据库...")

            with open(csv_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue

                        # 解析CSV格式: 标签名,评级,使用次数,"别名,中文名-作品名-英文标签"
                        parts = line.split(",", 3)
                        if len(parts) < 4:
                            continue

                        tag_name = parts[0].strip()
                        rating = int(parts[1])
                        usage_count = int(parts[2])
                        info_part = parts[3].strip('"')

                        # 解析信息部分
                        aliases = []
                        chinese_name = ""
                        series = ""

                        if info_part:
                            # 分割别名和描述
                            info_parts = info_part.split(",")
                            description_part = ""

                            # 找到包含中文的部分（通常是最后一部分）
                            for i, part in enumerate(info_parts):
                                if "-" in part and any(
                                    "\u4e00" <= c <= "\u9fff" for c in part
                                ):
                                    description_part = part
                                    aliases = [
                                        p.strip() for p in info_parts[:i] if p.strip()
                                    ]
                                    break

                            # 解析中文名和作品名
                            if description_part:
                                desc_parts = description_part.split("-")
                                if len(desc_parts) >= 2:
                                    chinese_name = desc_parts[0].strip()
                                    series = desc_parts[1].strip()

                                    # 统一括号格式：将 [内容] 和 (内容) 替换为 （内容）
                                    import re

                                    chinese_name = re.sub(
                                        r"\[([^\]]+)\]", r"（\1）", chinese_name
                                    )
                                    chinese_name = re.sub(
                                        r"\(([^)]+)\)", r"（\1）", chinese_name
                                    )
                                    series = re.sub(r"\[([^\]]+)\]", r"（\1）", series)
                                    series = re.sub(r"\(([^)]+)\)", r"（\1）", series)

                        # 该CSV主要是角色库，但可能混入场景/背景等通用标签。
                        # 它们不能进入角色别名表，否则“室内场景”会把“场景”
                        # 错误绑定成 scene_(arknights) 角色预设。
                        non_character_names = {
                            "场景",
                            "背景",
                            "物品",
                            "动作",
                            "服装",
                            "衣服",
                            "风格",
                        }
                        non_character_prefixes = (
                            "scene_",
                            "background_",
                            "location_",
                            "object_",
                            "item_",
                            "action_",
                            "clothing_",
                            "outfit_",
                            "style_",
                        )
                        if chinese_name in non_character_names or tag_name.casefold().startswith(
                            non_character_prefixes
                        ):
                            continue

                        # 存储角色信息
                        character_info = {
                            "tag": tag_name,
                            "chinese_name": chinese_name,
                            "series": series,
                            "aliases": aliases,
                            "rating": rating,
                            "usage_count": usage_count,
                        }

                        self.character_database[tag_name] = character_info

                        # 建立别名映射
                        for alias in aliases:
                            if alias and alias not in self.character_aliases:
                                self.character_aliases[alias] = tag_name
                            folded_alias = alias.casefold()
                            if (
                                folded_alias
                                and folded_alias not in self.character_aliases
                            ):
                                self.character_aliases[folded_alias] = tag_name

                        # 中文名也作为别名
                        if chinese_name and chinese_name not in self.character_aliases:
                            self.character_aliases[chinese_name] = tag_name

                    except Exception as e:
                        logger.debug(f"解析角色数据库第{line_num}行失败: {e}")
                        continue

            logger.info(
                f"角色数据库加载完成: {len(self.character_database)} 个角色, "
                f"{len(self.character_aliases)} 个别名"
            )

        except Exception as e:
            logger.error(f"加载角色数据库失败: {e}")
            self.character_database = {}
            self.character_aliases = {}

    def _save_last_draw_commands(self):
        """保存上次画图命令记录"""
        try:
            os.makedirs("data", exist_ok=True)
            with open(self.last_draw_commands_file, "w", encoding="utf-8") as f:
                json.dump(self.last_draw_commands, f, ensure_ascii=False, indent=2)
            logger.debug("画图命令记录已保存")
        except Exception as e:
            logger.error(f"保存画图命令记录失败: {e}")

    def _record_draw_command(
        self,
        user_id: str,
        command: str,
        event_type: str,
        image_url: Optional[str] = None,
        temp_strength: Optional[float] = None,
        temp_noise: Optional[float] = None,
        user_original_input: Optional[str] = None,
        event=None,
    ):
        """记录画图命令"""
        # 清理过期的图片URL（超过1小时的QQ图片链接）
        current_time = time.time()
        if image_url and image_url.startswith("https://multimedia.nt.qq.com.cn/"):
            # QQ图片链接通常1小时后失效，我们设置30分钟清理
            expiry_time = current_time + 30 * 60  # 30分钟后过期
        else:
            expiry_time = None

        # 只在首次记录或者不是重画时更新记录
        # 重画时不应该覆盖原始记录
        is_redraw = (
            hasattr(event, "_is_redraw") and event._is_redraw if event else False
        )
        if not is_redraw and (
            user_id not in self.last_draw_commands or user_original_input is not None
        ):
            self.last_draw_commands[user_id] = {
                "command": command,  # 完整的命令（包含预设画风）
                "user_original_input": user_original_input
                or command,  # 用户真正的原始输入
                "event_type": event_type,
                "image_url": image_url,
                "image_url_expiry": expiry_time,  # 图片URL过期时间
                "temp_strength": temp_strength,
                "temp_noise": temp_noise,
                "timestamp": current_time,  # 添加时间戳
            }
            # 异步保存，避免阻塞主流程
            task = asyncio.create_task(self._async_save_draw_commands())
            # 添加异常处理回调
            task.add_done_callback(self._handle_save_task_result)

    def _handle_save_task_result(self, task):
        """处理保存任务的结果"""
        try:
            task.result()  # 获取任务结果，如果有异常会抛出
        except Exception as e:
            logger.error(f"异步保存任务异常: {e}")

    async def _async_save_draw_commands(self):
        """异步保存画图命令记录"""
        try:
            await asyncio.sleep(0.1)  # 短暂延迟，避免频繁写入
            self._save_last_draw_commands()
        except Exception as e:
            logger.error(f"异步保存画图命令失败: {e}")

    async def _construct_image_message_for_redraw(self, event, image_url: str):
        """为重画功能构造包含图片的消息"""
        try:
            # 创建图片消息组件
            if image_url.startswith(("http://", "https://")):
                image_comp = Comp.Image.fromURL(image_url)
            else:
                image_comp = Comp.Image.fromFileSystem(image_url)

            # 将图片组件添加到消息中
            if (
                not hasattr(event.message_obj, "message")
                or not event.message_obj.message
            ):
                event.message_obj.message = []

            # 检查是否已经有图片，避免重复添加
            has_image = any(
                isinstance(seg, Comp.Image) for seg in event.message_obj.message
            )
            if not has_image:
                event.message_obj.message.append(image_comp)
                logger.info(f"为重画功能添加图片: {image_url}")

        except Exception:
            logger.warning(f"图片链接失效: {image_url}")
            # 抛出简洁的异常信息
            raise Exception("图片链接已过期")

    def _create_fake_forward_message(
        self, title: str, content: str, sender_name: str = "NovelAI助手"
    ) -> List:
        """创建伪造转发消息"""
        try:
            # 创建伪造转发的消息链
            # 使用Node组件来创建转发消息节点
            node = Comp.Node(
                uin=2854196310,  # 伪造的QQ号
                name=sender_name,
                content=[Comp.Plain(content)],
            )
            return [node]

        except Exception as e:
            logger.error(f"创建伪造转发消息失败: {e}")
            # 如果失败，返回普通文本
            return [Comp.Plain(f"📋 {title}\n\n{content}")]

    def _create_multi_fake_forward_message(self, messages: List[dict]) -> List:
        """
        创建一条包含多个子消息的伪造转发

        Args:
            messages: 消息列表，每个元素包含 {'uin': int, 'name': str, 'content': str}

        Returns:
            List: 包含一个Nodes组件的消息链，用于合并转发
        """
        try:
            # 使用正确的方式：创建多个Node，然后用Nodes包装
            nodes_list = []
            for msg in messages:
                node = Comp.Node(
                    uin=msg.get("uin", 2854196310),
                    name=msg.get("name", "用户"),
                    content=[Comp.Plain(msg.get("content", ""))],
                )
                nodes_list.append(node)

            # 使用Nodes包装所有节点，创建一条包含多个子消息的转发
            nodes = Comp.Nodes(nodes=nodes_list)
            return [nodes]

        except Exception as e:
            logger.error(f"创建多条伪造转发消息失败: {e}")
            # 如果失败，返回普通文本
            fallback_content = "\n".join(
                [
                    f"{msg.get('name', '用户')}: {msg.get('content', '')}"
                    for msg in messages
                ]
            )
            return [Comp.Plain(f"📋 多条消息\n\n{fallback_content}")]

    async def get_baidu_audit_access_token(self) -> Optional[str]:
        """获取百度审核API的access_token（带缓存）"""
        # 安全检查：确保属性已初始化
        if not hasattr(self, "baidu_audit_api_key") or not hasattr(
            self, "baidu_audit_secret_key"
        ):
            return None
        if not self.baidu_audit_api_key or not self.baidu_audit_secret_key:
            return None

        # 生成缓存键
        cache_key = Utils.generate_cache_key(
            "baidu_audit_token", self.baidu_audit_api_key
        )

        # 尝试从缓存获取
        cached_token = self.cache.get(cache_key)
        if cached_token:
            logger.debug("使用缓存的百度审核access_token")
            return cached_token

        try:
            params = {
                "grant_type": "client_credentials",
                "client_id": self.baidu_audit_api_key,
                "client_secret": self.baidu_audit_secret_key,
            }

            client = self.http_manager.get_client("baidu_audit_auth", 10.0)
            response = await client.post(Constants.BAIDU_OAUTH_API, params=params)

            # 使用统一的响应处理器
            response_data = ApiResponseHandler.handle_http_response(
                response, "百度审核认证API"
            )

            if response_data["success"]:
                result = response_data["data"]
                if isinstance(result, dict) and "access_token" in result:
                    token = result["access_token"]
                    # 缓存token，设置较短的TTL（比实际过期时间短一些）
                    self.cache.set(cache_key, token, ttl=1800)  # 30分钟
                    logger.info("百度审核API access_token获取成功并已缓存")
                    return token
                else:
                    logger.error(
                        f"获取百度审核access_token失败: 响应中无access_token字段"
                    )
                    return None
            else:
                error_msg = ApiResponseHandler.extract_error_message(
                    response_data, "审核认证请求失败"
                )
                logger.error(f"百度审核API认证失败: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"获取百度审核access_token异常: {e}")
            return None

    def _is_group_in_audit_whitelist(self, event) -> bool:
        """检查当前群是否在审核豁免白名单中。"""
        try:
            group_id = self._current_group_id(event)
            if group_id and group_id in self.audit_whitelist_groups:
                logger.info(f"群 {group_id} 在审核白名单中，跳过图片审核")
                return True
            return False
        except Exception as e:
            logger.error(f"检查审核白名单时出错: {e}")
            return False

    def _should_audit_image(self, event=None) -> bool:
        """判断当前 Bot 生成的图片是否应进入审核。"""
        if event is None:
            return False

        try:
            if not self.audit_whitelist_bots:
                return False

            bot_id = self._current_bot_id(event)
            if not bot_id or bot_id not in self.audit_whitelist_bots:
                return False

            return not self._is_group_in_audit_whitelist(event)
        except Exception as e:
            logger.error(f"检查 Bot 审核白名单时出错: {e}")
            return False

    async def audit_image_content(
        self, image_data: bytes, event=None
    ) -> Tuple[bool, str]:
        """审核图片内容"""
        # 百度审核
        if not self.image_audit_enabled:
            return True, "图片审核未启用"

        if not self._should_audit_image(event):
            return True, "当前图片不在审核白名单范围内，跳过审核"

        access_token = await self.get_baidu_audit_access_token()
        if not access_token:
            logger.warning("百度审核API未配置或认证失败，跳过图片审核")
            return True, "审核API未配置"

        try:
            # 将图片转换为base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            url = f"{Constants.BAIDU_AUDIT_API}?access_token={access_token}"

            # 百度图片审核API使用form-data格式
            data = {"image": image_base64}

            # 新版API不需要scenes参数，审核类型通过策略配置控制
            # 如果有策略ID配置，添加到请求中
            if hasattr(self, "audit_strategy_id") and self.audit_strategy_id:
                data["strategyId"] = self.audit_strategy_id

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            client = self.http_manager.get_client(
                "baidu_audit", 30.0
            )  # 图片审核可能需要更长时间
            response = await client.post(url, data=data, headers=headers)

            # 使用统一的响应处理器
            response_data = ApiResponseHandler.handle_http_response(
                response, "百度图片审核API"
            )

            if response_data["success"]:
                result = response_data["data"]
                if not isinstance(result, dict):
                    logger.error("百度图片审核API返回数据格式错误")
                    return True, "审核API返回格式错误，允许通过"

                logger.info(f"百度图片审核API响应: {result}")

                # 检查是否有业务错误
                if "error_code" in result:
                    error_code = result.get("error_code")
                    error_msg = result.get("error_msg", "未知错误")
                    logger.error(f"百度图片审核API业务错误: {error_code} - {error_msg}")
                    return True, f"审核API错误({error_code})，允许通过"

                # 检查审核结果
                conclusion_type = result.get("conclusionType", 1)
                conclusion = result.get("conclusion", "")

                # 记录图片大小信息
                image_size = Utils.format_file_size(len(image_data))
                self.log_draw(
                    f"图片审核完成: 大小={image_size}, 结论类型={conclusion_type}, 结论={conclusion}",
                    5,
                )

                # 根据配置的严格程度判断
                if conclusion_type == 1:
                    # 合规
                    return True, "图片内容合规"
                elif conclusion_type == 2:
                    # 不合规
                    return False, f"图片审核不通过: {conclusion}"
                elif conclusion_type == 3:
                    # 疑似 - 根据配置决定是否通过
                    if self.audit_conclusion_type <= 2:
                        # 严格模式或中等模式：疑似也拦截
                        return False, f"图片疑似违规: {conclusion}"
                    else:
                        # 宽松模式：疑似允许通过
                        return True, f"图片疑似违规但允许通过: {conclusion}"
                elif conclusion_type == 4:
                    # 审核失败
                    return True, "图片审核失败，允许通过"
                else:
                    return True, f"未知审核结果({conclusion_type})，允许通过"
            else:
                error_msg = ApiResponseHandler.extract_error_message(
                    response_data, "审核请求失败"
                )
                logger.error(f"百度图片审核API请求失败: {error_msg}")
                return True, "审核API请求失败，允许通过"

        except Exception as e:
            logger.error(f"图片审核异常: {e}")
            return True, "审核异常，允许通过"

    def _parse_temp_img2img_params(
        self, text: str
    ) -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
        """Parse temporary image generation parameters from a command.

        Args:
            text: Original drawing command text.

        Returns:
            Cleaned command text, strength, noise, and custom negative prompt.
        """
        import re

        # 初始化返回值
        cleaned_text = text
        temp_strength = None
        temp_noise = None
        custom_negative_prompt = None

        # 匹配 -s<数值> 格式的强度参数
        strength_pattern = r"-s([\d.]+)"
        strength_match = re.search(strength_pattern, text)
        if strength_match:
            try:
                temp_strength = float(strength_match.group(1))
                # 验证范围
                if not (0.1 <= temp_strength <= 1.0):
                    temp_strength = None
                else:
                    # 从文本中移除这个参数
                    cleaned_text = re.sub(strength_pattern, "", cleaned_text)
            except ValueError:
                temp_strength = None

        # 匹配 -n<数值> 格式的噪声参数
        noise_pattern = r"-n([\d.]+)"
        noise_match = re.search(noise_pattern, text)
        if noise_match:
            try:
                temp_noise = float(noise_match.group(1))
                # 验证范围
                if not (0.0 <= temp_noise <= 1.0):
                    temp_noise = None
                else:
                    # 从文本中移除这个参数
                    cleaned_text = re.sub(noise_pattern, "", cleaned_text)
            except ValueError:
                temp_noise = None

        # ``--uc`` consumes the rest of the command so negative terms cannot
        # leak into the positive prompt. Other temporary options should precede it.
        uc_match = re.search(
            r"(?<!\S)--uc(?:(?:\s+|=)(.*)|$)", cleaned_text, re.I | re.S
        )
        if uc_match:
            custom_negative_prompt = (
                " ".join((uc_match.group(1) or "").split()).strip(" ,") or None
            )
            cleaned_text = cleaned_text[: uc_match.start()]

        # 处理回车问题 - 将所有换行符和特殊空白字符替换为空格
        cleaned_text = (
            cleaned_text.replace("\n", " ").replace("\r", " ").replace("\r\n", " ")
        )
        # 处理其他可能的空白字符
        cleaned_text = cleaned_text.replace("\t", " ").replace(
            "\u00a0", " "
        )  # tab和不间断空格
        # 清理多余的空格
        cleaned_text = " ".join(cleaned_text.split())

        return cleaned_text, temp_strength, temp_noise, custom_negative_prompt

    async def _save_audit_failed_image(
        self, image_data: bytes, image_type: str = "unknown"
    ) -> Optional[str]:
        """保存审核失败的图片到指定路径"""
        if not self.image_audit_enabled:
            return None

        try:
            # 确保目录存在
            Utils.create_directory_if_not_exists(self.audit_failed_save_path)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_failed_{image_type}_{timestamp}.png"
            filepath = os.path.join(self.audit_failed_save_path, filename)

            # 保存图片
            with open(filepath, "wb") as f:
                f.write(image_data)

            logger.info(f"审核失败图片已保存: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"保存审核失败图片时出错: {e}")
            return None

    def _smart_split_prompt(self, prompt: str) -> List[str]:
        """智能分割提示词：识别中文词汇、数字编号、英文短语"""
        import re

        parts = []
        # 按空格分割
        words = prompt.split()

        current_part = []
        for word in words:
            # 检查是否是预设名称
            is_preset = False
            for preset_name in self.presets.keys():
                if word.lower() == preset_name.lower():
                    is_preset = True
                    break

            # 检查是否是画风预设
            if not is_preset:
                for style_name in self.style_presets.keys():
                    if word.lower() == style_name.lower():
                        is_preset = True
                        break

            # 检查是否是数字编号（001-430）
            is_number = re.match(r"^(0\d{2}|[1-4]\d{2}|430)$", word)

            # 检查是否是中文词汇
            is_chinese = re.search(r"[\u4e00-\u9fff]", word)

            if is_preset or is_number or is_chinese:
                # 如果当前有累积的英文部分，先保存
                if current_part:
                    parts.append(" ".join(current_part))
                    current_part = []
                # 预设、编号、中文词汇作为独立部分
                parts.append(word)
            else:
                # 英文词汇累积
                current_part.append(word)

        # 处理剩余的英文部分
        if current_part:
            parts.append(" ".join(current_part))

        return [part.strip() for part in parts if part.strip()]

    def apply_preset_and_style(
        self,
        prompt: str,
        apply_global_style: bool = True,
        drawing_settings: Optional[dict] = None,
    ) -> Tuple[str, Optional[str]]:
        """Apply prompt presets and structured styles in input order.

        Args:
            prompt: Original user prompt.
            apply_global_style: Whether to apply the effective default/session style.
            drawing_settings: Effective settings for the current request. Inline
                resolution and style selection update this mapping for this request.

        Returns:
            Processed prompt and the applied style name, if any.
        """
        processed_prompt = prompt
        applied_presets = []
        detected_style = None
        settings = drawing_settings if drawing_settings is not None else {}

        # 先处理逗号：将中文逗号转为英文逗号
        normalized_prompt = prompt.replace("，", ",")

        prompt_tokens = re.split(r"([,\s]+)", normalized_prompt)
        inline_resolutions = []
        filtered_tokens = []
        for token in prompt_tokens:
            resolution_name = token.strip()
            if resolution_name in Constants.IMAGE_RESOLUTIONS:
                inline_resolutions.append(resolution_name)
            else:
                filtered_tokens.append(token)

        if inline_resolutions:
            resolution_name = inline_resolutions[-1]
            settings["resolution"] = Constants.IMAGE_RESOLUTIONS[resolution_name]
            normalized_prompt = "".join(filtered_tokens).strip(" ,")
            logger.info(f"本次绘图使用{resolution_name}: {settings['resolution']}")

        # 改进的分割逻辑：优先按逗号分割，如果没有逗号则按空格分割中文部分
        if "," in normalized_prompt:
            # 有逗号时按逗号分割
            parts = [part.strip() for part in normalized_prompt.split(",")]
            logger.info(f"按逗号分割提示词: {len(parts)} 部分")
            for i, part in enumerate(parts[:3]):  # 只显示前3部分避免日志过长
                logger.info(f"  部分{i + 1}: '{part}'")
        else:
            # 没有逗号时，智能分割：将连续的中文词作为独立部分
            parts = self._smart_split_prompt(normalized_prompt)
            logger.info(f"智能分割提示词: {len(parts)} 部分")

        # 简化逻辑：直接处理每个部分，在原位置展开预设
        processed_parts = []

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # 统一处理所有部分（不再对第一部分特殊处理）
            found_preset = False

            # 检查整个部分是否是预设
            for preset_name, preset_content in self.presets.items():
                if part.lower() == preset_name.lower():
                    # 将预设内容分割并添加
                    preset_words = [w.strip() for w in preset_content.split(",")]
                    processed_parts.extend(preset_words)
                    applied_presets.append((preset_name, preset_content))
                    found_preset = True
                    logger.info(f"识别到预设: {preset_name}")
                    break

            # 检查整个部分是否是画风
            if not found_preset:
                for style_name in self.style_presets.keys():
                    if part.lower() == style_name.lower():
                        detected_style = style_name
                        found_preset = True
                        logger.info(f"识别到画风: {style_name}")
                        # 画风不添加到输出中
                        break

            # 如果不是预设或画风，保留原样
            if not found_preset:
                processed_parts.append(part)

        # 组合所有部分
        processed_prompt = ", ".join(processed_parts) if processed_parts else ""

        if applied_presets:
            logger.info(
                f"应用了 {len(applied_presets)} 个预设: {[p[0] for p in applied_presets]}"
            )

        # 应用画风
        if detected_style and detected_style in self.style_presets:
            style_prompt = normalize_style_entry(self.style_presets[detected_style])[
                "positive_prompt"
            ]
            settings["applied_style_name"] = detected_style
            final_prompt = (
                f"{processed_prompt}, {style_prompt}"
                if processed_prompt
                else style_prompt
            )
            logger.info(f"应用检测到的画风 '{detected_style}'")
            return final_prompt, detected_style

        # Apply the effective session style when no style was explicit in the prompt.
        style_enabled = settings.get("style_enabled", self.enable_style_preset)
        style_name = settings.get("style_name", self.selected_style_preset)
        if (
            apply_global_style
            and style_enabled
            and style_name
            and style_name in self.style_presets
        ):
            style_prompt = normalize_style_entry(self.style_presets[style_name])[
                "positive_prompt"
            ]
            settings["applied_style_name"] = style_name
            final_prompt = (
                f"{style_prompt}, {processed_prompt}"
                if processed_prompt
                else style_prompt
            )
            logger.info(f"应用会话画风 '{style_name}'")
            return final_prompt, style_name

        return processed_prompt, None

    def get_negative_prompt(self, drawing_settings: Optional[dict] = None) -> str:
        """Return the effective negative prompt for a drawing request.

        Args:
            drawing_settings: Effective request drawing settings.

        Returns:
            Negative prompt text, or an empty string when disabled.
        """
        settings = drawing_settings or {}
        if "custom_negative_prompt" in settings:
            custom_negative_prompt = str(settings["custom_negative_prompt"]).strip(" ,")
            logger.info("本次绘图使用 --uc 自定义负面提示词")
            return deduplicate_negative_prompt(custom_negative_prompt)

        style_name = settings.get("applied_style_name")
        if not style_name and settings.get("style_enabled"):
            style_name = settings.get("style_name")
        negative_prompt = ""
        if style_name and style_name in self.style_presets:
            negative_prompt = normalize_style_entry(self.style_presets[style_name])[
                "negative_prompt"
            ]
            if negative_prompt:
                logger.info(f"使用画风 '{style_name}' 的负面提示词")

        return deduplicate_negative_prompt(negative_prompt)

    def get_api_url(
        self, endpoint: str = "generate-image", use_newapi: bool | None = None
    ) -> str:
        """Return the image endpoint for the selected request route.

        Args:
            endpoint: NovelAI-compatible endpoint name.
            use_newapi: Optional request-level backend override.

        Returns:
            The complete image API URL.
        """
        request_uses_newapi = self._uses_newapi() if use_newapi is None else use_newapi
        if request_uses_newapi:
            return f"{self.newapi_base_url.rstrip('/')}/ai/{endpoint}"
        if self.api_proxy:
            return f"{self.api_proxy.rstrip('/')}/ai/{endpoint}"
        return f"https://image.novelai.net/ai/{endpoint}"

    def _uses_newapi(self) -> bool:
        """Return whether image requests should use new-api."""
        return getattr(self, "api_backend", "novelai") == "newapi"

    def _get_active_api_token(self) -> str:
        """Return the credential for the selected image API backend."""
        if self._uses_newapi():
            return str(getattr(self, "newapi_api_key", "") or "").strip()
        return str(getattr(self, "api_token", "") or "").strip()

    def _get_ai_video_user_data_path(self) -> Path | None:
        """Locate the shared user data file owned by astrbot_plugin_ai_video."""
        user_data_path = getattr(self, "ai_video_user_data_path", None)
        if user_data_path is not None:
            return Path(user_data_path)
        if get_astrbot_data_path is None:
            return None
        try:
            return (
                Path(get_astrbot_data_path())
                / "plugin_data"
                / "astrbot_plugin_ai_video"
                / "users.json"
            )
        except Exception as exc:
            logger.warning("无法定位 AI 图片插件用户数据: %s", exc)
            return None

    def _load_ai_video_user_data(self) -> dict[str, Any]:
        """Load ai_video's user/key and group-authorization data."""
        user_data_path = self._get_ai_video_user_data_path()
        if user_data_path is None or not user_data_path.exists():
            return {"users": {}, "groups": {}}
        try:
            payload = json.loads(user_data_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {"users": {}, "groups": {}}
            if not isinstance(payload.get("users"), dict):
                payload["users"] = {}
            if not isinstance(payload.get("groups"), dict):
                payload["groups"] = {}
            return payload
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("读取 AI 图片插件用户数据失败: %s", exc)
            return {"users": {}, "groups": {}}

    def _save_ai_video_user_data(self, payload: dict[str, Any]) -> bool:
        """Persist shared ai_video data without dropping unrelated fields."""
        user_data_path = self._get_ai_video_user_data_path()
        if user_data_path is None:
            return False
        try:
            user_data_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = user_data_path.with_suffix(user_data_path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp_path.replace(user_data_path)
            return True
        except OSError as exc:
            logger.warning("保存 AI 图片插件用户数据失败: %s", exc)
            return False

    @staticmethod
    def _get_event_bot_id(event) -> str:
        """Return the current Bot id used for per-Bot key bindings."""
        try:
            getter = getattr(event, "get_self_id", None)
            if callable(getter):
                return str(getter() or "").strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def _mask_api_key(api_key: str | None) -> str:
        """Mask a key for diagnostics without writing the credential to logs."""
        value = str(api_key or "").strip()
        if not value:
            return "<none>"
        if len(value) <= 12:
            return f"{value[:3]}***"
        return f"{value[:6]}...{value[-6:]}"

    def _get_bound_user_api_key(
        self, user_id: str, bot_id: str | None = None
    ) -> str | None:
        """Read the current Bot's personal key, then the legacy first key.

        ``astrbot_plugin_ai_video`` keeps both a user-wide ``api_keys`` list and
        an optional ``bot_chat_keys`` mapping.  The latter is authoritative when
        a user has selected different keys for different Bots; falling back to
        the first list item preserves compatibility with older data files.
        """
        payload = self._load_ai_video_user_data()
        normalized_user_id = str(user_id or "").strip()
        user = payload.get("users", {}).get(normalized_user_id, {})
        keys = user.get("api_keys", []) if isinstance(user, dict) else []
        if not isinstance(keys, (list, tuple, set)):
            return None
        owned_keys = [
            key
            for item in keys
            if (key := str(item or "").strip()).startswith("sk-")
        ]
        if not owned_keys:
            return None
        owned_key_set = set(owned_keys)

        normalized_bot_id = str(bot_id or "").strip()
        if normalized_bot_id:
            bindings = payload.get("bot_chat_keys", {})
            binding = (
                bindings.get(f"{normalized_bot_id}:{normalized_user_id}", {})
                if isinstance(bindings, dict)
                else {}
            )
            bot_key = (
                str(binding.get("api_key", "") or "").strip()
                if isinstance(binding, dict)
                else ""
            )
            if bot_key in owned_key_set:
                return bot_key

        return owned_keys[0]

    def _get_group_authorized_user_id(self, event) -> str | None:
        """Return the user who authorized the current group, if any."""
        group_id = self._current_group_id(event)
        if not group_id or group_id == "0":
            return None
        payload = self._load_ai_video_user_data()
        binding = payload.get("groups", {}).get(str(group_id))
        if not isinstance(binding, dict) or not binding.get("enabled"):
            return None
        owner_uid = str(binding.get("owner_uid", "")).strip()
        return owner_uid or None

    def _get_group_authorized_api_key(self, event) -> str | None:
        """Return the current group's owner's key, without exposing it to users."""
        owner_uid = self._get_group_authorized_user_id(event)
        return (
            self._get_bound_user_api_key(owner_uid, self._get_event_bot_id(event))
            if owner_uid
            else None
        )

    def parse_api_error(self, response) -> str:
        """解析API错误响应"""
        try:
            if hasattr(response, "text"):
                error_text = response.text
            elif hasattr(response, "content"):
                error_text = response.content.decode("utf-8", errors="ignore")
            else:
                error_text = str(response)

            # 尝试解析JSON错误信息
            try:
                error_data = json.loads(error_text)
                if "message" in error_data:
                    return error_data["message"]
                elif "error" in error_data:
                    return error_data["error"]
                else:
                    return error_text
            except json.JSONDecodeError:
                return error_text

        except Exception as e:
            return f"解析错误响应失败: {e}"

    def get_image_proxy_url(self, original_url: str) -> str:
        """获取图片代理URL"""
        if self.image_proxy and original_url.startswith("https://image.novelai.net"):
            return original_url.replace("https://image.novelai.net", self.image_proxy)
        return original_url

    def uses_v4_prompt_structure(self, model_name: str) -> bool:
        """Check whether a model uses the structured V4 prompt fields.

        Args:
            model_name: NovelAI image model identifier.

        Returns:
            Whether the model expects ``v4_prompt`` and ``v4_negative_prompt``.
        """
        return model_name.lower().startswith(("nai-diffusion-4", "nai-diffusion-5"))

    def get_model_version_label(self, model_name: str) -> str:
        """Return a human-readable NovelAI image model version.

        Args:
            model_name: NovelAI image model identifier.

        Returns:
            A short version label for logs and user-facing status messages.
        """
        normalized_model = model_name.lower()
        if normalized_model.startswith("nai-diffusion-5"):
            return "v5"
        if normalized_model.startswith(("nai-diffusion-4-5", "nai-diffusion-4.5")):
            return "v4.5"
        if normalized_model.startswith("nai-diffusion-4"):
            return "v4"
        if "diffusion-3" in normalized_model:
            return "v3"
        return "未知版本"

    @staticmethod
    def _is_v4_5_model(model_name: str) -> bool:
        """Return whether the final request model is a V4.5 model."""
        normalized_model = (model_name or "").lower()
        return normalized_model.startswith(("nai-diffusion-4-5", "nai-diffusion-4.5"))

    def _get_v4_5_sampling_parameters(
        self,
        width: int,
        height: int,
        drawing_settings: dict,
    ) -> dict:
        """Build V4.5-only advanced sampling parameters.

        Auto SMEA is a website-side convenience toggle, so it is resolved into
        the API's ``sm`` and ``sm_dyn`` fields before the request is sent.
        """
        noise_schedule = str(
            drawing_settings.get(
                "v4_5_noise_schedule",
                getattr(self, "v4_5_noise_schedule", "karras"),
            )
            or "karras"
        ).lower()
        if noise_schedule not in Constants.V4_5_NOISE_SCHEDULES:
            noise_schedule = "karras"

        smea_dyn = bool(
            drawing_settings.get("v4_5_smea_dyn", getattr(self, "v4_5_smea_dyn", False))
        )
        auto_smea = bool(
            drawing_settings.get(
                "v4_5_auto_smea", getattr(self, "v4_5_auto_smea", False)
            )
        )
        if auto_smea:
            smea = width * height >= Constants.V4_5_AUTO_SMEA_MIN_PIXELS
            if not smea:
                smea_dyn = False
        else:
            smea = bool(
                drawing_settings.get("v4_5_smea", getattr(self, "v4_5_smea", False))
            )
            smea = smea or smea_dyn

        cfg_rescale = drawing_settings.get(
            "cfg_rescale", getattr(self, "prompt_guidance_rescale", 0.0)
        )
        return {
            "params_version": 4,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": noise_schedule,
            "dynamic_thresholding": bool(
                drawing_settings.get(
                    "v4_5_decrisper", getattr(self, "v4_5_decrisper", False)
                )
            ),
            "skip_cfg_above_sigma": (
                Constants.V4_5_CFG_DELAY_SIGMA
                if drawing_settings.get(
                    "v4_5_variety_boost",
                    getattr(self, "v4_5_variety_boost", False),
                )
                else None
            ),
            "sm": smea,
            "sm_dyn": smea_dyn,
        }

    def _apply_daily_model_fallback(
        self, drawing_settings: dict, quota_scope: str
    ) -> bool:
        """Route requests to V4.5 Full after a group's daily quota is exhausted.

        Args:
            drawing_settings: Mutable per-request drawing settings.
            quota_scope: Group or session owning the daily free quota.

        Returns:
            Whether this call changed the request model to the fallback model.
        """
        if self.usage_manager.check_group_daily_limit(quota_scope):
            return False

        previous_model = drawing_settings.get("model")
        drawing_settings["model"] = Constants.DAILY_FALLBACK_MODEL
        if previous_model == Constants.DAILY_FALLBACK_MODEL:
            return False

        logger.info(
            "本群今日免费额度已用完，模型自动降级: %s -> %s",
            previous_model,
            Constants.DAILY_FALLBACK_MODEL,
        )
        return True

    def _resolve_daily_generation_route(
        self, drawing_settings: dict, user_id: str, event=None
    ) -> tuple[bool, str | None, bool, bool]:
        """Select the effective model, credential, and backend for a request.

        Args:
            drawing_settings: Mutable settings for the current request.
            user_id: Current message sender identifier.
            event: Current message event, used for group authorization.

        Returns:
            A ``(model_changed, personal_key, uses_newapi, personal_key_route)``
            tuple. ``uses_newapi`` selects the New API backend (either a
            personal/group key or the operator's global ``newapi_api_key`` when
            ``api_backend`` is ``newapi``). ``personal_key_route`` is ``True``
            only when a personal or group key funds the request, which keeps
            personal-key-only behaviour (free-quota exemption and the V4.5
            access-denied fallback) intact.
        """
        quota_scope = self._resolve_quota_scope(event, user_id)
        use_global_newapi = self._uses_newapi()

        if drawing_settings.get(Constants.PERSONAL_V5_FALLBACK_FLAG):
            previous_model = str(drawing_settings.get("model") or "")
            drawing_settings["model"] = Constants.DAILY_FALLBACK_MODEL
            return previous_model != Constants.DAILY_FALLBACK_MODEL, None, False, False

        # A group authorization explicitly shares the owner's personal key with
        # members who do not have their own key. It is intentionally checked
        # before the free daily quota: unlike the shared token, this route
        # must actually use the key that was authorized for this group.
        personal_key = self._get_bound_user_api_key(
            user_id, self._get_event_bot_id(event)
        )
        group_key = (
            self._get_group_authorized_api_key(event)
            if event is not None and not personal_key
            else None
        )
        if group_key and not personal_key:
            previous_model = str(drawing_settings.get("model") or "")
            if not getattr(self, "enable_v5", True):
                if previous_model.lower().startswith("nai-diffusion-5"):
                    drawing_settings["model"] = Constants.DAILY_FALLBACK_MODEL
                    return True, group_key, True, True
                return False, group_key, True, True
            if previous_model.lower().startswith("nai-diffusion-5"):
                return False, group_key, True, True
            drawing_settings["model"] = Constants.PERSONAL_KEY_MODEL
            logger.info(
                "当前群已授权，用户 %s 使用群授权人的个人密钥，模型: %s -> %s",
                user_id,
                previous_model,
                Constants.PERSONAL_KEY_MODEL,
            )
            return True, group_key, True, True

        # If the shared NovelAI token is not configured, a sender's own key
        # remains usable even while the free daily quota is available.
        if personal_key and not str(self.api_token or "").strip():
            previous_model = str(drawing_settings.get("model") or "")
            if not getattr(self, "enable_v5", True) and previous_model.lower().startswith(
                "nai-diffusion-5"
            ):
                drawing_settings["model"] = Constants.DAILY_FALLBACK_MODEL
                return True, personal_key, True, True
            if previous_model.lower().startswith("nai-diffusion-5"):
                return False, personal_key, True, True
            drawing_settings["model"] = Constants.PERSONAL_KEY_MODEL
            return True, personal_key, True, True

        if not getattr(self, "enable_v5", True):
            previous_model = str(drawing_settings.get("model") or "")
            if previous_model.lower().startswith("nai-diffusion-5"):
                drawing_settings["model"] = Constants.DAILY_FALLBACK_MODEL
                logger.info(
                    "V5 画图已关闭，模型自动降级: %s -> %s",
                    previous_model,
                    Constants.DAILY_FALLBACK_MODEL,
                )
                return True, None, use_global_newapi, False
            return False, None, use_global_newapi, False

        if self.usage_manager.check_group_daily_limit(quota_scope):
            return False, None, use_global_newapi, False

        if not personal_key:
            return (
                self._apply_daily_model_fallback(drawing_settings, quota_scope),
                None,
                use_global_newapi,
                False,
            )

        previous_model = str(drawing_settings.get("model") or "")
        if previous_model.lower().startswith("nai-diffusion-5"):
            return False, personal_key, True, True

        drawing_settings["model"] = Constants.PERSONAL_KEY_MODEL
        logger.info(
            "本群今日免费额度已用完，用户 %s 改用个人密钥并强制 V5: %s -> %s",
            user_id,
            previous_model,
            Constants.PERSONAL_KEY_MODEL,
        )
        return True, personal_key, True, True

    def _activate_personal_v5_access_fallback(
        self,
        drawing_settings: dict,
        meta_info: dict | None,
        personal_key_active: bool,
    ) -> bool:
        """Switch a denied personal-key V5 request to the NovelAI V4.5 route.

        Only requests funded by a personal or group key are handled here; the
        shared-token and global New API routes never carry a personal key, so
        they must not fall back to NovelAI. The marker is stored only in the
        mutable per-request settings so Roll requests keep using the fallback
        after the first denial without changing the session defaults.
        """
        if not personal_key_active or not isinstance(meta_info, dict):
            return False

        model_name = str(drawing_settings.get("model") or "")
        if not model_name.lower().startswith("nai-diffusion-5"):
            return False

        status_code = meta_info.get("status_code")
        error_text = " ".join(
            str(meta_info.get(key) or "") for key in ("response_text", "error")
        ).casefold()
        access_denied_markers = (
            "no access to model",
            "does not have access to model",
            "model is not allowed",
        )
        if status_code != 403 or not any(
            marker in error_text for marker in access_denied_markers
        ):
            return False

        drawing_settings[Constants.PERSONAL_V5_FALLBACK_FLAG] = True
        drawing_settings["model"] = Constants.DAILY_FALLBACK_MODEL
        logger.warning(
            "个人 Key 无权访问模型 %s，自动回退到 NovelAI %s",
            model_name,
            Constants.DAILY_FALLBACK_MODEL,
        )
        return True

    async def _retry_with_daily_v45_on_personal_access_denied(
        self,
        image_data,
        meta_info: dict,
        *,
        drawing_settings: dict,
        personal_key_active: bool,
        prompt: str,
        prompt_before_normalization: str,
        event,
        scene_user_prompt: str,
        request_callback,
    ) -> tuple:
        """Retry one denied personal-key V5 request on the shared V4.5 route."""
        if image_data or not self._activate_personal_v5_access_fallback(
            drawing_settings,
            meta_info,
            personal_key_active,
        ):
            return image_data, meta_info, prompt, False

        try:
            fallback_prompt = await self.normalize_prompt_for_novelai(
                prompt_before_normalization,
                event,
                drawing_settings["model"],
                user_prompt=scene_user_prompt,
                drawing_settings=drawing_settings,
            )
        except NAIErrors.ParameterError as exc:
            return (
                None,
                {
                    "error": str(exc),
                    "error_type": "param",
                    "fallback_attempted": True,
                },
                prompt,
                True,
            )

        fallback_image_data, fallback_meta = await request_callback(fallback_prompt)
        fallback_meta = dict(fallback_meta or {})
        fallback_meta.update(
            {
                "fallback_attempted": True,
                "fallback_reason": "personal_key_v5_access_denied",
                "fallback_model": Constants.DAILY_FALLBACK_MODEL,
            }
        )
        return (
            fallback_image_data,
            fallback_meta,
            fallback_prompt,
            True,
        )

    def create_v4_prompt_structure(
        self,
        prompt: str,
        width: int,
        height: int,
        max_characters: int,
        character_positions: Optional[list[dict[str, float] | None]] = None,
    ) -> dict:
        """Create the structured prompt used by V4 and later models.

        Args:
            prompt: Base image generation prompt.
            width: Output image width.
            height: Output image height.
            max_characters: Maximum character-caption count for the model.
            character_positions: Optional explicit centers aligned with the
                character captions. The normal drawing flow does not provide
                planner-generated positions.

        Returns:
            Structured prompt payload accepted by the ``v4_prompt`` API field.
        """
        return build_v4_prompt(
            prompt,
            width=width,
            height=height,
            max_characters=max_characters,
            character_positions=character_positions,
        )

    def get_model_parameters(
        self,
        model_name: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        scale: float,
        sampler: str,
        drawing_settings: Optional[dict] = None,
    ) -> dict:
        """Build image generation parameters for a NovelAI model.

        Args:
            model_name: NovelAI image model identifier.
            prompt: Positive image generation prompt.
            width: Output image width in pixels.
            height: Output image height in pixels.
            steps: Number of diffusion sampling steps.
            scale: Prompt guidance scale.
            sampler: NovelAI sampler identifier.
            drawing_settings: Effective settings for the current session.

        Returns:
            Parameters for the NovelAI ``generate-image`` request.
        """

        # 获取负面提示词
        settings = drawing_settings or {}
        negative_prompt = self.get_negative_prompt(settings)
        # Keep a tag that is requested positively out of UC.  This is done
        # after style/scene/custom UC composition so every request path uses
        # the same conservative, weight-aware cleanup.
        negative_prompt = remove_positive_duplicates_from_negative(
            prompt,
            negative_prompt,
        )

        is_v5_model = model_name.lower().startswith("nai-diffusion-5")
        is_v4_5_model = self._is_v4_5_model(model_name)
        base_params = {
            "params_version": 4 if is_v5_model or is_v4_5_model else 3,
            "width": width,
            "height": height,
            "scale": scale,
            "sampler": sampler,
            "steps": steps,
            "n_samples": 1,
            "ucPreset": 0,
            # Do not let the NovelAI API inject its built-in quality tags.
            # Positive prompts must contain only user-provided content.
            "qualityToggle": False,
            "dynamic_thresholding": False,
            "controlnet_strength": 1,
            "legacy": False,
            "add_original_image": False,
            # Preserve the existing neutral V5 request values. V4.5 settings
            # replace these fields below only when V4.5 is the final model.
            "cfg_rescale": 0.0,
            "noise_schedule": "karras",
            "legacy_v3_extend": False,
            "skip_cfg_above_sigma": None,
            "negative_prompt": negative_prompt,
            "seed": random.randint(0, 4294967295),
        }

        if is_v4_5_model:
            base_params.update(
                self._get_v4_5_sampling_parameters(width, height, settings)
            )

        if self.uses_v4_prompt_structure(model_name):
            # V4 and later models use the structured prompt fields.
            max_characters = 22 if is_v5_model else 6
            # Character fields are created only from explicit ``|`` sections in
            # the prompt. Legacy planner coordinates and per-character UC are
            # deliberately ignored so stale request state cannot add content.
            structured_prompt = self.create_v4_prompt_structure(
                prompt,
                width,
                height,
                max_characters,
            )
            character_count = len(structured_prompt["caption"]["char_captions"])
            character_positions = [
                item["centers"][0]
                for item in structured_prompt["caption"]["char_captions"]
            ]
            if (
                character_count > 1
                and "custom_negative_prompt" not in settings
                and any(
                    re.search(
                        rf"\b{re.escape(tag)}\b",
                        negative_prompt,
                        re.IGNORECASE,
                    )
                    for tag in ("little dolls", "extra characters")
                )
            ):
                negative_prompt = remove_negative_prompt_tags(
                    negative_prompt,
                    {"little dolls", "extra characters"},
                )
                base_params["negative_prompt"] = negative_prompt
                logger.info("多人图已移除与主体数量冲突的单人 UC 项")
            base_params.update(
                {
                    "v4_prompt": structured_prompt,
                    "v4_negative_prompt": build_v4_negative_prompt(
                        negative_prompt,
                        character_count=character_count,
                        width=width,
                        height=height,
                        character_positions=character_positions,
                        character_captions=None,
                        use_coords=structured_prompt["use_coords"],
                    ),
                }
            )

        return base_params

    async def _create_http_client_and_headers(
        self,
        api_token: str | None = None,
        use_newapi: bool | None = None,
    ) -> tuple:
        """Create the HTTP client settings and authorization headers.

        Args:
            api_token: Optional per-request credential overriding the global key.
            use_newapi: Optional request-level backend override.

        Returns:
            A pair containing the httpx client configuration and request headers.
        """
        client_config = {
            "timeout": 60.0,
            "limits": httpx.Limits(
                max_keepalive_connections=2, max_connections=3, keepalive_expiry=30.0
            ),
        }

        # 画图API专用HTTP代理（novelai_proxy 优先；向下兼容 api_proxy 也视为反代不作为网络代理）
        request_uses_newapi = self._uses_newapi() if use_newapi is None else use_newapi
        novelai_proxy = getattr(self, "novelai_proxy", "")
        if (
            not request_uses_newapi
            and novelai_proxy
            and novelai_proxy.startswith(("http://", "https://", "socks5://"))
        ):
            client_config["proxy"] = novelai_proxy
            logger.info(f"画图请求使用代理: {novelai_proxy}")
        elif not request_uses_newapi and getattr(self, "api_proxy", ""):
            # api_proxy 为反代地址，不需要设置网络代理
            logger.debug(f"画图请求使用反代地址: {self.api_proxy}")
        elif request_uses_newapi:
            logger.debug(f"画图请求使用 new-api: {self.newapi_base_url}")

        request_api_token = api_token or (
            self.newapi_api_key if request_uses_newapi else self.api_token
        )
        logger.debug(
            "画图请求凭据: backend=%s, key=%s",
            "new-api" if request_uses_newapi else "novelai",
            self._mask_api_key(request_api_token),
        )
        headers = {
            "Authorization": f"Bearer {request_api_token}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Connection": "close",  # 强制关闭连接，避免连接复用问题
        }

        return client_config, headers

    def _validate_common_api_params(
        self,
        prompt: str = None,
        steps: int = None,
        scale: float = None,
        model_name: str = None,
        api_token: str | None = None,
        use_newapi: bool | None = None,
    ):
        """Validate shared image API request parameters.

        Args:
            prompt: Positive prompt to validate.
            steps: Sampling step count.
            scale: Prompt guidance scale.
            model_name: Requested NovelAI model identifier.
            api_token: Optional per-request credential overriding the global key.
            use_newapi: Optional request-level backend override.

        Raises:
            NAIErrors.ConfigurationError: If no credential or model is available.
            NAIErrors.ParameterError: If a request parameter is outside its range.
        """
        request_uses_newapi = self._uses_newapi() if use_newapi is None else use_newapi
        request_api_token = api_token or (
            self.newapi_api_key if request_uses_newapi else self.api_token
        )
        if not request_api_token:
            credential_name = (
                "new-api API Key" if request_uses_newapi else "NovelAI Token"
            )
            raise NAIErrors.ConfigurationError(f"未配置 {credential_name}")

        if not (model_name or self.default_model):
            raise NAIErrors.ConfigurationError("未配置模型名称")

        if prompt is not None and (not prompt or not prompt.strip()):
            raise NAIErrors.ParameterError("提示词不能为空")

        if steps is not None and (steps < 1 or steps > 50):
            raise NAIErrors.ParameterError(f"生成步数 {steps} 超出有效范围 (1-50)")

        if scale is not None and (scale < 1 or scale > 15):
            raise NAIErrors.ParameterError(
                f"提示词引导权重 {scale} 超出有效范围 (1-15)"
            )

    async def _download_image_with_client(
        self, image_url: str, timeout: float = 30.0
    ) -> bytes:
        """Read an image from any reference format emitted by AstrBot.

        Image components are not limited to HTTP URLs. Depending on the platform,
        ``file``/``url`` may contain a local path, a file URI, an inline Base64
        payload, or a protocol-relative URL. Resolve those forms locally instead
        of passing every value to httpx, which rejects non-HTTP references as a
        URL without a protocol.
        """
        try:
            if isinstance(image_url, bytes):
                return image_url
            if not isinstance(image_url, str) or not image_url.strip():
                raise NAIErrors.ImageProcessingError("图片来源为空")

            image_ref = image_url.strip()
            lowered_ref = image_ref.lower()

            if lowered_ref.startswith("base64://"):
                try:
                    encoded_data = image_ref[len("base64://") :]
                    return base64.b64decode(encoded_data, validate=True)
                except (ValueError, TypeError) as exc:
                    raise NAIErrors.ImageProcessingError(
                        "图片 Base64 数据无效", exc
                    ) from exc

            if lowered_ref.startswith("data:"):
                try:
                    metadata, encoded_data = image_ref.split(",", 1)
                    if ";base64" not in metadata.lower():
                        raise ValueError("only Base64 data URIs are supported")
                    return base64.b64decode(encoded_data, validate=True)
                except (ValueError, TypeError) as exc:
                    raise NAIErrors.ImageProcessingError(
                        "图片 data URI 无效", exc
                    ) from exc

            if lowered_ref.startswith("file:"):
                parsed = urlsplit(image_ref)
                decoded_path = url2pathname(unquote(parsed.path))
                if parsed.netloc and parsed.netloc.lower() != "localhost":
                    decoded_path = f"//{parsed.netloc}{decoded_path}"
                image_path = Path(decoded_path)
                if not image_path.is_file():
                    raise NAIErrors.ImageProcessingError("本地图片文件不存在")
                return image_path.read_bytes()

            image_path = Path(image_ref)
            try:
                is_local_file = image_path.is_file()
            except OSError:
                is_local_file = False
            if is_local_file:
                return image_path.read_bytes()

            if image_ref.startswith("//"):
                image_ref = f"https:{image_ref}"
            elif re.match(
                r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*\.[A-Za-z]{2,})(?::\d+)?(?:[/?#]|$)",
                image_ref,
            ):
                # Some adapters expose a valid host/path but omit the scheme.
                image_ref = f"https://{image_ref}"

            if image_ref.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(image_ref)
                    if response.status_code != 200:
                        raise NAIErrors.NetworkError(
                            f"下载图片失败: HTTP {response.status_code}"
                        )
                    return response.content

            # Older adapters may place bare Base64 directly in Image.file.
            compact_ref = "".join(image_ref.split())
            if len(compact_ref) >= 128:
                try:
                    return base64.b64decode(compact_ref, validate=True)
                except (ValueError, TypeError):
                    pass

            if "://" in image_ref:
                protocol = image_ref.split("://", 1)[0]
                raise NAIErrors.ImageProcessingError(
                    f"不支持的图片地址协议: {protocol}"
                )
            raise NAIErrors.ImageProcessingError("无法识别图片来源")
        except (NAIErrors.NetworkError, NAIErrors.ImageProcessingError):
            raise
        except httpx.RequestError as e:
            raise NAIErrors.NetworkError(f"网络请求失败: {str(e)}")
        except OSError as e:
            raise NAIErrors.ImageProcessingError(f"读取图片失败: {str(e)}", e)

    def _get_user_friendly_error_message(
        self, error: Exception, context: str = "操作"
    ) -> str:
        """获取用户友好的错误信息"""
        # 处理自定义异常类型
        if isinstance(error, NAIErrors.NetworkError):
            return f"❌ {str(error)}"

        error_str = str(error)
        if "Internal Server Error" in error_str:
            return f"❌ 服务器内部错误，请稍后重试"
        elif "timeout" in error_str.lower():
            return f"❌ 网络连接超时，请检查网络连接"
        elif "connection" in error_str.lower():
            return f"❌ 网络连接失败，请检查网络连接"
        else:
            return f"❌ {context}失败，请稍后重试"

    def _extract_command_prompt(self, message_text: str) -> tuple[str, str]:
        """
        从消息中提取命令类型和prompt
        返回: (command_type, prompt)
        """
        # 命令前缀映射配置
        COMMAND_PREFIXES = {
            "画图": ["/画图", "画图"],
            "roll图": ["/roll图", "roll图"],
        }

        # 按前缀长度排序，优先匹配长前缀
        all_prefixes = []
        for command_type, prefixes in COMMAND_PREFIXES.items():
            for prefix in prefixes:
                all_prefixes.append((prefix, command_type))

        # 按前缀长度降序排序，确保长前缀优先匹配
        all_prefixes.sort(key=lambda x: len(x[0]), reverse=True)

        # 匹配命令前缀
        for prefix, command_type in all_prefixes:
            if message_text.startswith(prefix):
                prompt = message_text[len(prefix) :].strip()
                return command_type, prompt

        # 未匹配到任何命令前缀
        return "unknown", message_text

    def extract_prompt_from_message(
        self, event, drawing_settings: Optional[dict] = None
    ) -> Tuple[str, Optional[str]]:
        """Extract and process a prompt from a drawing command.

        Args:
            event: AstrBot message event.
            drawing_settings: Effective settings for the current session.

        Returns:
            Processed prompt and the detected style name, if any.
        """
        try:
            # 获取原始消息文本
            message_text = event.message_str.strip()
            logger.info(f"原始消息文本: '{repr(message_text)}'")

            # 使用配置化的命令解析
            command_type, prompt = self._extract_command_prompt(message_text)

            if command_type == "unknown":
                # 对于command触发器，AstrBot可能已经移除了命令部分
                # 直接使用原始文本作为prompt
                prompt = message_text
                logger.info(
                    "AstrBot command触发器已处理命令，直接使用原始文本作为prompt"
                )
            else:
                logger.debug(f"识别命令类型: {command_type}")

            # 处理回车问题 - 将所有换行符和特殊空白字符替换为空格
            prompt = self._clean_prompt_whitespace(prompt)

            logger.info(
                f"命令移除后的原始文本: '{message_text[:50]}{'...' if len(message_text) > 50 else ''}'"
            )
            logger.info(
                f"提取的prompt (已处理回车): '{prompt[:100]}{'...' if len(prompt) > 100 else ''}')"
            )

            # 应用预设和检测画风
            final_prompt, detected_style = self.apply_preset_and_style(
                prompt, drawing_settings=drawing_settings
            )

            logger.info(f"处理后的prompt长度: {len(final_prompt)} 字符")

            return final_prompt, detected_style

        except Exception as e:
            logger.error(f"提取prompt失败: {e}")
            return "", None

    def _clean_prompt_whitespace(self, prompt: str) -> str:
        """Normalize whitespace while retaining intentional prompt paragraphs."""
        text = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", " ").replace("\u00a0", " ")
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n", text):
            normalized = " ".join(paragraph.split())
            if normalized:
                paragraphs.append(normalized)
        return "\n\n".join(paragraphs)

    async def normalize_prompt_for_novelai(
        self,
        prompt: str,
        event,
        model_name: str,
        *,
        user_prompt: Optional[str] = None,
        drawing_settings: Optional[dict] = None,
    ) -> str:
        """Validate and minimally format user input before NovelAI submission.

        Args:
            prompt: User-provided prompt after any explicitly requested preset use.
            event: Retained for compatibility; never used to access an LLM provider.
            model_name: Target NovelAI image model identifier.
            user_prompt: Retained for compatibility; never translated or expanded.
            drawing_settings: Per-request settings. Legacy planner fields are removed.

        Returns:
            Locally formatted prompt with explicit ``base | character`` sections
            preserved exactly as user structure.

        Raises:
            NAIErrors.ParameterError: If the normalized prompt exceeds the model's
                supported character-caption count.
        """
        # This compatibility entry point is intentionally local-only.  Older
        # versions used this method for character lookup and LLM scene planning;
        # neither behavior is allowed during image generation now.
        del event, user_prompt
        max_characters = 22 if model_name.lower().startswith("nai-diffusion-5") else 6
        settings = drawing_settings if drawing_settings is not None else {}
        for key in (
            "character_positions",
            "scene_negative_prompt",
            "character_negative_prompts",
        ):
            settings.pop(key, None)
        try:
            return normalize_v5_prompt(
                str(prompt or "").strip(),
                max_characters=max_characters,
                infer_character_sections=False,
            )
        except ValueError as exc:
            raise NAIErrors.ParameterError(str(exc)) from exc

    async def download_images_parallel(self, image_urls: list) -> list:
        """并行下载多张图片"""
        import asyncio

        async def download_single(url):
            try:
                return await self._download_image_with_client(url)
            except Exception as e:
                logger.error(f"下载图片失败 {url}: {e}")
                return None

        # 创建下载任务
        tasks = [download_single(url) for url in image_urls]

        # 并行执行下载
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤成功的结果
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"下载图片异常 {image_urls[i]}: {result}")
            elif result is not None:
                successful_results.append((image_urls[i], result))

        return successful_results

    def extract_prompt_from_img2img_message(
        self, event, drawing_settings: Optional[dict] = None
    ) -> Tuple[str, Optional[str]]:
        """Extract and process a prompt from an image-to-image command.

        Args:
            event: AstrBot message event.
            drawing_settings: Effective settings for the current session.

        Returns:
            Processed prompt and the detected style name, if any.
        """
        try:
            message_text = event.message_str.strip()

            # 移除[图片]标记
            clean_text = re.sub(r"\$\$图片\$\$", "", message_text).strip()

            # 移除命令前缀和命令本身
            prompt = ""
            if clean_text.startswith("/图生图"):
                prompt = clean_text[4:].strip()  # 移除 "/图生图"
            elif clean_text.startswith("图生图"):
                prompt = clean_text[3:].strip()  # 兼容无前缀的情况
            elif clean_text.startswith("/画图"):
                prompt = clean_text[3:].strip()  # 从画图转换过来的情况
            elif clean_text.startswith("画图"):
                prompt = clean_text[2:].strip()  # 从画图转换过来的情况（无前缀）
            else:
                # 对于command触发器，有时候message_str可能不包含命令本身
                prompt = clean_text

            # 处理回车问题 - 将所有换行符替换为空格
            prompt = prompt.replace("\n", " ").replace("\r", " ")
            # 清理多余的空格
            prompt = " ".join(prompt.split())

            logger.info(
                f"图生图提取的原始prompt (已处理回车): {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
            )

            # 应用预设和检测画风
            if prompt:
                final_prompt, detected_style = self.apply_preset_and_style(
                    prompt, drawing_settings=drawing_settings
                )
                logger.info(f"图生图处理后的prompt长度: {len(final_prompt)} 字符")
                return final_prompt, detected_style

            return prompt, None

        except Exception as e:
            logger.error(f"图生图提取prompt失败: {e}")
            return "", None

    def extract_image_from_response(self, response_data: bytes) -> Optional[bytes]:
        """从NovelAI响应中提取图片数据"""
        try:
            # 尝试直接作为图片数据
            img = None
            try:
                img = PILImage.open(io.BytesIO(response_data))
                logger.info(f"直接识别为图片: {img.size}, 格式: {img.format}")
                return response_data
            except Exception:
                pass
            finally:
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass

            # 尝试作为ZIP文件处理
            try:
                with zipfile.ZipFile(io.BytesIO(response_data)) as zip_file:
                    file_list = zip_file.namelist()
                    logger.info(f"ZIP文件包含: {file_list}")

                    for filename in file_list:
                        if filename.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".webp")
                        ):
                            image_data = zip_file.read(filename)
                            img = None
                            try:
                                img = PILImage.open(io.BytesIO(image_data))
                                logger.info(
                                    f"从ZIP提取图片: {filename}, 大小: {img.size}, 格式: {img.format}"
                                )
                                return image_data
                            finally:
                                if img is not None:
                                    try:
                                        img.close()
                                    except Exception:
                                        pass
            except Exception as e:
                logger.debug(f"ZIP处理失败: {e}")

            # 尝试作为base64编码的JSON
            try:
                json_data = json.loads(response_data.decode("utf-8"))
                if "image" in json_data:
                    image_b64 = json_data["image"]
                    image_data = base64.b64decode(image_b64)
                    img = None
                    try:
                        img = PILImage.open(io.BytesIO(image_data))
                        logger.info(
                            f"从JSON base64提取图片: {img.size}, 格式: {img.format}"
                        )
                        return image_data
                    finally:
                        if img is not None:
                            try:
                                img.close()
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"JSON处理失败: {e}")

            logger.info(f"响应数据前100字节: {response_data[:100]}")
            logger.info(f"响应数据类型检测失败，大小: {len(response_data)} bytes")

            # 如果所有尝试都失败，抛出图片处理错误
            raise NAIErrors.ImageProcessingError("无法识别返回的图片格式")

        except NAIErrors.ImageProcessingError as e:
            # 直接将自定义错误向上传递
            raise e
        except Exception as e:
            # 将其他错误转换为自定义错误
            raise NAIErrors.ImageProcessingError("图片提取失败", e)

    def extract_images_from_messages(self, messages, include_files: bool = False):
        """Extract image references from message components.

        Args:
            messages: Message components to inspect.
            include_files: Also accept file attachments as image sources. This is
                used by metadata extraction for untouched PNG attachments.

        Returns:
            Image URL, local path, or inline data references. File attachments
            are returned before preview images when ``include_files`` is enabled.
        """
        image_urls = []

        if include_files:
            for seg in messages:
                if isinstance(seg, Comp.File):
                    # Prefer the local attachment when both references exist. It
                    # is normally the untouched file, while the URL may be a
                    # platform-generated preview or a transcoded download.
                    for file_ref in (
                        getattr(seg, "file_", ""),
                        getattr(seg, "url", ""),
                    ):
                        if file_ref and file_ref not in image_urls:
                            image_urls.append(file_ref)

        for seg in messages:
            if isinstance(seg, Comp.Image):
                # Prefer local/original references when a component carries
                # both a local path and a platform preview URL.
                candidates = []
                for attr_name in ("path", "file", "url"):
                    value = getattr(seg, attr_name, "")
                    if value and value not in candidates:
                        candidates.append(value)
                data = getattr(seg, "data", None)
                if isinstance(data, dict):
                    for value in (data.get("file"), data.get("url")):
                        if value and value not in candidates:
                            candidates.append(value)

                for image_ref in candidates:
                    image_ref = self.get_image_proxy_url(image_ref)
                    if image_ref not in image_urls:
                        image_urls.append(image_ref)

        return image_urls

    def get_reply_message_images(self, event, include_files: bool = False):
        """Extract image references from the replied-to message.

        Args:
            event: Current AstrBot message event.
            include_files: Also accept file attachments as image sources.

        Returns:
            Image references embedded in the first reply component.
        """
        try:
            messages = event.message_obj.message

            for seg in messages:
                if isinstance(seg, Comp.Reply):
                    if hasattr(seg, "chain") and seg.chain:
                        images = self.extract_images_from_messages(
                            seg.chain, include_files=include_files
                        )
                        if images:
                            return images
                    break

            return []
        except Exception as e:
            logger.error(f"获取引用消息图片失败: {e}")
            return []

    async def _get_reply_message_images_for_metadata(self, event) -> list[str]:
        """Resolve quoted images, including replies whose chain was not embedded.

        Some adapters leave ``Reply.chain`` empty and expose only the quoted
        message ID. AstrBot's quoted-message resolver can fetch that message
        through the platform API. This path is kept specific to metadata
        extraction because the image-to-image path has synchronous callers.
        """
        image_refs = self.get_reply_message_images(event, include_files=True)
        if extract_quoted_message_images is None:
            return image_refs

        try:
            fetched_refs = await extract_quoted_message_images(event)
        except Exception as exc:
            logger.debug(f"远程获取引用消息图片失败: {exc}")
            fetched_refs = []

        for image_ref in fetched_refs or []:
            if image_ref and image_ref not in image_refs:
                image_refs.append(image_ref)
        return image_refs

    async def download_and_encode_image(self, image_url: str) -> Optional[str]:
        """下载图片并转换为base64"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if image_url.startswith(("http://", "https://")):
                    try:
                        response = await client.get(image_url)
                        response.raise_for_status()  # 检查HTTP错误
                        image_data = response.content
                    except httpx.HTTPStatusError as e:
                        raise NAIErrors.NetworkError(
                            f"HTTP错误: {e.response.status_code}", e
                        )
                    except httpx.RequestError as e:
                        raise NAIErrors.NetworkError(f"请求错误: {str(e)}", e)
                else:
                    # 本地文件
                    if not os.path.exists(image_url):
                        raise NAIErrors.ImageProcessingError(
                            f"本地文件不存在: {image_url}"
                        )
                    with open(image_url, "rb") as f:
                        image_data = f.read()

            # 验证图片格式
            img = None
            try:
                img = PILImage.open(io.BytesIO(image_data))
                logger.info(f"图片验证成功: {img.size}, 格式: {img.format}")

                # 智能图片优化 - 处理尺寸和文件大小
                width, height = img.size
                file_size_mb = len(image_data) / 1024 / 1024

                # 设置目标限制
                max_dimension = 1536  # 降低最大尺寸限制，提高兼容性
                max_file_size_mb = 3.0  # 降低文件大小限制到3MB

                needs_resize = width > max_dimension or height > max_dimension
                needs_compress = file_size_mb > max_file_size_mb

                if needs_resize or needs_compress:
                    logger.info(
                        f"图片优化前: 尺寸={width}x{height}, 大小={file_size_mb:.2f}MB"
                    )

                    # 如果需要调整尺寸
                    if needs_resize:
                        scale_ratio = min(max_dimension / width, max_dimension / height)
                        new_width = int(width * scale_ratio)
                        new_height = int(height * scale_ratio)

                        logger.info(
                            f"调整尺寸: {width}x{height} → {new_width}x{new_height}"
                        )
                        img = img.resize(
                            (new_width, new_height), PILImage.Resampling.LANCZOS
                        )

                    # 智能压缩质量选择
                    if file_size_mb > 8:
                        quality = 70  # 超大文件用低质量
                    elif file_size_mb > 5:
                        quality = 80  # 大文件用中等质量
                    elif file_size_mb > 3:
                        quality = 85  # 中等文件用较高质量
                    else:
                        quality = 90  # 小文件用高质量

                    # 保存优化后的图片
                    img_io = io.BytesIO()
                    # 强制使用JPEG格式以获得更好的压缩率
                    if img.mode in ("RGBA", "LA", "P"):
                        # 处理透明图片，转换为RGB
                        background = PILImage.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        background.paste(
                            img,
                            mask=img.split()[-1]
                            if img.mode in ("RGBA", "LA")
                            else None,
                        )
                        img = background

                    img.save(img_io, format="JPEG", quality=quality, optimize=True)
                    image_data = img_io.getvalue()

                    new_file_size_mb = len(image_data) / 1024 / 1024
                    logger.info(
                        f"图片优化后: 大小={new_file_size_mb:.2f}MB (质量={quality})"
                    )

                    # 如果还是太大，进一步压缩
                    if new_file_size_mb > max_file_size_mb:
                        for retry_quality in [60, 50, 40]:
                            img_io = io.BytesIO()
                            img.save(
                                img_io,
                                format="JPEG",
                                quality=retry_quality,
                                optimize=True,
                            )
                            image_data = img_io.getvalue()
                            new_file_size_mb = len(image_data) / 1024 / 1024

                            logger.info(
                                f"进一步压缩: 质量={retry_quality}, 大小={new_file_size_mb:.2f}MB"
                            )

                            if new_file_size_mb <= max_file_size_mb:
                                break

                    self.log_draw(
                        f"✅ 图片优化完成: 最终大小={len(image_data) / 1024 / 1024:.2f}MB",
                        4,
                    )
            except Exception as e:
                raise NAIErrors.ImageProcessingError("无效的图片格式", e)
            finally:
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass

            return base64.b64encode(image_data).decode("utf-8")
        except NAIErrors.NetworkError as e:
            # 直接将自定义错误向上传递
            raise e
        except NAIErrors.ImageProcessingError as e:
            # 直接将自定义错误向上传递
            raise e
        except Exception as e:
            # 将其他错误转换为自定义错误
            raise NAIErrors.ImageProcessingError("图片下载失败", e)

    def _has_images_in_message(self, event) -> bool:
        """检查消息中是否包含图片"""
        try:
            messages = event.message_obj.message
            for seg in messages:
                if isinstance(seg, Comp.Image):
                    return True
            return False
        except Exception:
            return False

    def _check_reply_has_images(self, event) -> bool:
        """检查引用的消息中是否有图片"""
        try:
            messages = event.message_obj.message
            for seg in messages:
                if isinstance(seg, Comp.Reply):
                    if hasattr(seg, "chain") and seg.chain:
                        for reply_seg in seg.chain:
                            if isinstance(reply_seg, Comp.Image):
                                return True
                    break
            return False
        except Exception:
            return False

    def _get_last_image_url(self, event) -> Optional[str]:
        """获取最近的图片URL（当前消息或引用消息）"""
        # 先从当前消息获取
        image_urls = self.extract_images_from_messages(event.message_obj.message)
        if image_urls:
            return image_urls[0]

        # 再从引用消息获取
        reply_images = self.get_reply_message_images(event)
        if reply_images:
            return reply_images[0]

        return None

    async def _handle_text_to_image(self, event):
        """处理文生图的核心逻辑"""
        user_id = event.get_sender_id()
        drawing_settings = self._get_session_drawing_settings(event)
        # 先获取原始消息，解析临时参数，再进行预设识别
        original_message = event.message_str.strip()  # 保存原始消息

        # 从原始消息中解析临时参数
        temp_strength = None
        temp_noise = None
        custom_negative_prompt = None
        if original_message:
            cleaned_message, temp_strength, temp_noise, custom_negative_prompt = (
                self._parse_temp_img2img_params(original_message)
            )
            # 更新事件消息，移除临时参数
            event.message_str = cleaned_message
            if custom_negative_prompt is not None:
                drawing_settings["custom_negative_prompt"] = custom_negative_prompt

        _, scene_user_prompt = self._extract_command_prompt(event.message_str.strip())
        scene_user_prompt = self._clean_prompt_whitespace(scene_user_prompt)

        # 保存处理后的命令（移除临时参数，用于重画）
        clean_command_for_redraw = event.message_str.strip()

        # 现在进行预设识别和提示词提取
        prompt, detected_style = self.extract_prompt_from_message(
            event, drawing_settings=drawing_settings
        )

        # 检查是否有图片（当前消息或引用消息）
        has_current_images = self._has_images_in_message(event)
        has_reply_images = self._check_reply_has_images(event)

        # 如果消息包含图片或引用的消息有图片，自动转为图生图
        if has_current_images or has_reply_images:
            logger.info("检测到图片，自动转换为图生图模式")
            # 如果有临时参数，需要将它们重新添加到提示词部分，而不是整个消息
            if (
                temp_strength is not None
                or temp_noise is not None
                or custom_negative_prompt is not None
            ):
                param_parts = []
                if temp_strength is not None:
                    param_parts.append(f"-s{temp_strength}")
                if temp_noise is not None:
                    param_parts.append(f"-n{temp_noise}")
                param_str = " ".join(param_parts)
                # 只将临时参数添加到提示词前面，保持命令格式正确
                img2img_prefix = "/图生图"
                plain_img2img_prefix = "图生图"
                if param_str:
                    img2img_prefix += f" {param_str}"
                    plain_img2img_prefix += f" {param_str}"
                if event.message_str.startswith("/画图"):
                    event.message_str = event.message_str.replace(
                        "/画图", img2img_prefix, 1
                    )
                elif event.message_str.startswith("画图"):
                    event.message_str = event.message_str.replace(
                        "画图", plain_img2img_prefix, 1
                    )
                elif param_str:
                    event.message_str = f"{param_str} {event.message_str}".strip()
                if custom_negative_prompt is not None:
                    event.message_str = (
                        f"{event.message_str} --uc {custom_negative_prompt}".strip()
                    )
                logger.info("已将单次参数传递到自动图生图请求")

            async for result in self._handle_image_to_image(event):
                yield result
            return

        if (
            not str(self.api_token or "").strip()
            and not self._get_bound_user_api_key(
                user_id, self._get_event_bot_id(event)
            )
            and not self._get_group_authorized_api_key(event)
        ):
            yield event.plain_result(
                "❌ 请先在管理面板配置 NovelAI Token，或绑定个人密钥/开启本群 NAI 画图授权"
            )
            return

        if not drawing_settings["model"]:
            yield event.plain_result("❌ 请先在管理面板配置模型名称")
            return

        # prompt 和临时参数已经在前面解析过了

        if not prompt.strip():
            yield event.plain_result(
                "❌ 请提供图片描述\n用法: /画图 [长图|宽图|方图] 描述 或预设名\n💡 如果同时发送图片，将自动切换为图生图模式\n💡 临时参数: -s<强度> -n<噪声> (如: /画图 -s0.6 美女)"
            )
            return

        # 处理衣服标签
        clothing_original_prompt = prompt
        processed_prompt = self.clothing_manager.process_prompt(prompt)
        if processed_prompt != prompt:
            prompt = processed_prompt
            logger.info(f"衣服标签处理: {clothing_original_prompt} -> {prompt}")

        prompt_before_normalization = prompt
        self._resolve_daily_generation_route(drawing_settings, user_id, event)
        try:
            prompt = await self.normalize_prompt_for_novelai(
                prompt_before_normalization,
                event,
                drawing_settings["model"],
                user_prompt=scene_user_prompt,
                drawing_settings=drawing_settings,
            )
        except NAIErrors.ParameterError as e:
            yield event.plain_result(f"❌ 提示词格式错误: {e}")
            return

        prompt_for_display = prompt

        # 使用新的排队系统，每个请求都会排队
        (
            request_id,
            can_process,
            queue_position,
            _,
        ) = await self.usage_manager.add_to_queue(user_id)

        if not can_process:
            queue_message = f"🕒 排队中，当前第{queue_position}位"
            async for result in self._send_status_message_with_auto_recall(
                event, queue_message
            ):
                yield result

            # 等待轮到这个请求
            has_turn = await self.usage_manager.wait_for_turn(request_id)
            if not has_turn:
                yield event.plain_result("❌ 排队等待超时，请重新提交画图请求")
                return
            # 不再显示"轮到你了"的提示
        else:
            async for result in self._send_status_message_with_auto_recall(
                event, "🎨 正在绘制中，请稍候..."
            ):
                yield result

        try:
            (
                route_changed,
                request_api_token,
                request_uses_newapi,
                personal_key_active,
            ) = self._resolve_daily_generation_route(drawing_settings, user_id, event)
            if route_changed:
                try:
                    prompt = await self.normalize_prompt_for_novelai(
                        prompt_before_normalization,
                        event,
                        drawing_settings["model"],
                        user_prompt=scene_user_prompt,
                        drawing_settings=drawing_settings,
                    )
                    prompt_for_display = prompt
                except NAIErrors.ParameterError as e:
                    yield event.plain_result(f"❌ 额度路由切换后提示词格式错误: {e}")
                    return

            start_time = time.time()
            width, height = map(int, drawing_settings["resolution"].split("x"))

            # 修复：正确调用文生图API，只传递需要的参数
            image_data, meta_info = await self._call_novelai_txt2img(
                prompt=prompt,
                width=width,
                height=height,
                steps=drawing_settings["steps"],
                scale=drawing_settings["scale"],
                sampler=drawing_settings["sampler"],
                drawing_settings=drawing_settings,
                api_token=request_api_token,
                use_newapi=request_uses_newapi,
            )

            async def retry_txt2img_with_v45(fallback_prompt: str):
                return await self._call_novelai_txt2img(
                    prompt=fallback_prompt,
                    width=width,
                    height=height,
                    steps=drawing_settings["steps"],
                    scale=drawing_settings["scale"],
                    sampler=drawing_settings["sampler"],
                    drawing_settings=drawing_settings,
                    api_token=None,
                    use_newapi=False,
                )

            (
                image_data,
                meta_info,
                prompt,
                personal_v5_fallback_attempted,
            ) = await self._retry_with_daily_v45_on_personal_access_denied(
                image_data,
                meta_info,
                drawing_settings=drawing_settings,
                personal_key_active=personal_key_active,
                prompt=prompt,
                prompt_before_normalization=prompt_before_normalization,
                event=event,
                scene_user_prompt=scene_user_prompt,
                request_callback=retry_txt2img_with_v45,
            )
            prompt_for_display = prompt

            if image_data:
                elapsed_time = time.time() - start_time
                # Only shared-token generations consume the group's free quota;
                # personal keys and group authorizations do not.
                quota_scope = (
                    self._resolve_quota_scope(event, user_id)
                    if not personal_key_active or personal_v5_fallback_attempted
                    else None
                )
                self.usage_manager.record_usage(user_id, quota_scope)

                try:
                    final_image_data = self.extract_image_from_response(image_data)

                    if final_image_data:
                        self._cache_generated_image(event, final_image_data)
                        img = None
                        try:
                            img = PILImage.open(io.BytesIO(final_image_data))
                            logger.info(
                                f"最终图片验证成功: {img.size}, 格式: {img.format}"
                            )
                        except Exception as e:
                            logger.error(f"图片验证失败: {e}")
                            raise
                        finally:
                            if img is not None:
                                try:
                                    img.close()
                                except Exception:
                                    pass

                        # 图片内容审核
                        if self.image_audit_enabled and self._should_audit_image(event):
                            (
                                audit_passed,
                                audit_message,
                            ) = await self.audit_image_content(final_image_data, event)
                            if not audit_passed:
                                # 审核不通过，保存到审核失败路径
                                await self._save_audit_failed_image(
                                    final_image_data, "txt2img"
                                )
                                yield event.plain_result(f"❌ {audit_message}")
                                return
                            logger.info(f"图片审核通过: {audit_message}")

                        # 自动保存图片
                        saved_path = await self._save_image_locally(
                            final_image_data,
                            "txt2img",
                        )
                        if saved_path:
                            logger.info(f"图片已保存到: {saved_path}")

                        # 根据 show_params 配置决定显示内容
                        if self.show_params:
                            model_version = self.get_model_version_label(
                                drawing_settings["model"]
                            )

                            # 构建结果信息
                            style_info = ""
                            if detected_style:
                                style_info = f" | 画风: {detected_style}"
                            elif (
                                drawing_settings["style_enabled"]
                                and drawing_settings["style_name"]
                            ):
                                style_info = (
                                    f" | 画风: {drawing_settings['style_name']}"
                                )

                            negative_info = ""
                            if "custom_negative_prompt" in drawing_settings:
                                negative_info = " | 负面: --uc 自定义"

                            chain = [
                                Comp.At(qq=event.get_sender_id()),
                                *(
                                    [
                                        Comp.Plain(
                                            "\n"
                                            + Constants.PERSONAL_V5_FALLBACK_NOTICE
                                            + "\n"
                                        )
                                    ]
                                    if personal_v5_fallback_attempted
                                    else []
                                ),
                                Comp.Plain(f"\n🖼️ 图片生成完成！\n"),
                                Comp.Plain(
                                    f"📝 提示词: {prompt_for_display[:100]}{'...' if len(prompt_for_display) > 100 else ''}\n"
                                ),
                                Comp.Plain(
                                    f"⚙️ 参数: {drawing_settings['steps']}步 | 权重{drawing_settings['scale']} | {drawing_settings['resolution']}{style_info}{negative_info}\n"
                                ),
                                Comp.Plain(
                                    f"🤖 模型: {drawing_settings['model']} ({model_version})\n"
                                ),
                                Comp.Image.fromBytes(final_image_data),
                                Comp.Plain(f"\n🕐 耗时: {elapsed_time:.1f}秒"),
                            ]
                        else:
                            # 简洁模式：只显示图片和时间
                            chain = [
                                Comp.At(qq=event.get_sender_id()),
                                *(
                                    [
                                        Comp.Plain(
                                            "\n"
                                            + Constants.PERSONAL_V5_FALLBACK_NOTICE
                                            + "\n"
                                        )
                                    ]
                                    if personal_v5_fallback_attempted
                                    else []
                                ),
                                Comp.Plain("\n"),
                                Comp.Image.fromBytes(final_image_data),
                                Comp.Plain(f"\n🕐 {elapsed_time:.1f}秒"),
                            ]

                        # 记录画图成功日志
                        self.log_draw(f"画图成功: 耗时 {elapsed_time:.1f}秒", 1)

                        # 使用自动撤回功能发送图片
                        async for result in self._send_image_with_auto_recall(
                            event, chain
                        ):
                            yield result

                        # 记录这次的画图命令（在成功生成后记录完整信息）
                        # 构造完全处理后的命令（包含预设画风）
                        full_processed_command = f"/画图 {prompt_for_display}"

                        self._record_draw_command(
                            user_id,
                            full_processed_command,  # 完全处理后的命令
                            "text_to_image",
                            self._get_last_image_url(event),
                            temp_strength,
                            temp_noise,
                            original_message,  # 用户真正的原始输入
                            event,  # 传入event对象检查是否是重画
                        )
                        self.log_draw(
                            f"已记录画图命令用于重画: {clean_command_for_redraw}", 5
                        )
                    else:
                        yield event.plain_result("❌ 无法解析返回的图片数据")

                except NAIErrors.ImageProcessingError as img_error:
                    logger.error(f"图片处理错误: {img_error}")
                    yield event.plain_result(f"❌ 图片处理失败: {str(img_error)}")
                except Exception as img_error:
                    logger.error(f"图片处理错误: {img_error}")
                    yield event.plain_result(f"❌ 图片处理失败: {str(img_error)}")
            else:
                # 处理不同类型的错误
                error_message = "❌ 图片生成失败"
                if "error_type" in meta_info:
                    error_type = meta_info["error_type"]
                    if error_type == "network":
                        # 网络错误已经在错误类中优化过了，直接使用
                        error_message = f"❌ {meta_info['error']}"
                    elif error_type == "auth":
                        error_message = "❌ API认证失败，请检查Token是否正确"
                    elif error_type == "quota":
                        error_message = "❌ API使用次数超限，请稍后再试"
                    elif error_type == "server":
                        # 服务器错误已经在错误类中优化过了，直接使用
                        error_message = f"❌ {meta_info['error']}"
                    elif error_type == "config":
                        error_message = f"❌ 配置错误: {meta_info['error']}"
                    elif error_type == "param":
                        error_message = f"❌ 参数错误: {meta_info['error']}"
                    else:
                        error_message = "❌ 图片生成失败，请稍后重试"
                else:
                    error_message = "❌ 图片生成失败，请稍后重试"

                if personal_v5_fallback_attempted:
                    error_message = (
                        Constants.PERSONAL_V5_FALLBACK_NOTICE + "\n" + error_message
                    )
                yield event.plain_result(error_message)

        except Exception as e:
            logger.error(f"NovelAI文生图错误: {e}")
            yield event.plain_result(f"❌ 生成出错: {str(e)}")
        finally:
            self.usage_manager.finish_processing(request_id)
            # 开始处理队列中的下一个请求
            next_request_id = await self.usage_manager.start_next_in_queue()
            if next_request_id:
                logger.info(f"开始处理队列中的下一个请求: {next_request_id}")

    async def _handle_image_to_image(self, event):
        """处理图生图的核心逻辑 - 使用早期版本的有效实现"""
        user_id = event.get_sender_id()
        drawing_settings = self._get_session_drawing_settings(event)
        # 先获取原始消息，解析临时参数，再进行预设识别
        original_message = event.message_str.strip()  # 保存原始消息

        # 从原始消息中解析临时参数
        temp_strength = None
        temp_noise = None
        custom_negative_prompt = None
        if original_message:
            cleaned_message, temp_strength, temp_noise, custom_negative_prompt = (
                self._parse_temp_img2img_params(original_message)
            )
            # 更新事件消息，移除临时参数
            event.message_str = cleaned_message
            if custom_negative_prompt is not None:
                drawing_settings["custom_negative_prompt"] = custom_negative_prompt

        scene_user_prompt = self._clean_prompt_whitespace(
            self._get_command_argument(event, "图生图", "画图")
        )

        # 保存处理后的命令（移除临时参数，用于重画）
        clean_command_for_redraw = event.message_str.strip()

        # 现在进行预设识别和提示词提取
        prompt, detected_style = self.extract_prompt_from_img2img_message(
            event, drawing_settings=drawing_settings
        )

        if (
            not str(self.api_token or "").strip()
            and not self._get_bound_user_api_key(
                user_id, self._get_event_bot_id(event)
            )
            and not self._get_group_authorized_api_key(event)
        ):
            yield event.plain_result(
                "❌ 请先在管理面板配置 NovelAI Token，或绑定个人密钥/开启本群 NAI 画图授权"
            )
            return

        if not drawing_settings["model"]:
            yield event.plain_result("❌ 请先在管理面板配置模型名称")
            return

        # prompt 和临时参数已经在前面解析过了

        # 处理衣服标签
        if prompt:  # 只有当有提示词时才进行处理
            clothing_original_prompt = prompt
            processed_prompt = self.clothing_manager.process_prompt(prompt)
            if processed_prompt != prompt:
                prompt = processed_prompt
                logger.info(
                    f"图生图衣服标签处理: {clothing_original_prompt} -> {prompt}"
                )

        image_urls = []
        current_images = self.extract_images_from_messages(event.message_obj.message)
        image_urls.extend(current_images)

        if not image_urls:
            reply_images = self.get_reply_message_images(event)
            image_urls.extend(reply_images)

        if not image_urls:
            yield event.plain_result(
                "❌ 请发送图片后使用此指令\n\n💡 支持格式:\n• 发送图片 + /图生图 [描述]\n• /图生图 [描述] + 图片\n• /画图 [描述] + 图片（自动转换）\n• 回复包含图片的消息 + /图生图 [描述]"
            )
            return

        if not prompt.strip():
            yield event.plain_result(
                "❌ 请提供图片描述\n"
                "用法: /图生图 [描述]"
            )
            return

        prompt_before_normalization = prompt
        self._resolve_daily_generation_route(drawing_settings, user_id, event)
        try:
            prompt = await self.normalize_prompt_for_novelai(
                prompt_before_normalization,
                event,
                drawing_settings["model"],
                user_prompt=scene_user_prompt,
                drawing_settings=drawing_settings,
            )
        except NAIErrors.ParameterError as e:
            yield event.plain_result(f"❌ 提示词格式错误: {e}")
            return

        # 使用新的排队系统，每个请求都会排队
        (
            request_id,
            can_process,
            queue_position,
            _,
        ) = await self.usage_manager.add_to_queue(user_id)

        if not can_process:
            queue_message = f"🕒 排队中，当前第{queue_position}位"
            async for result in self._send_status_message_with_auto_recall(
                event, queue_message
            ):
                yield result

            # 等待轮到这个请求
            has_turn = await self.usage_manager.wait_for_turn(request_id)
            if not has_turn:
                yield event.plain_result("❌ 排队等待超时，请重新提交图生图请求")
                return
            # 不再显示"轮到你了"的提示
        else:
            # 检测到从画图命令转换过来
            original_command = (
                "画图"
                if event.message_str.strip().startswith(("/画图", "画图"))
                else "图生图"
            )
            status_message = f"🎨 正在进行图生图，请稍候..."
            async for result in self._send_status_message_with_auto_recall(
                event, status_message
            ):
                yield result

        try:
            (
                route_changed,
                request_api_token,
                request_uses_newapi,
                personal_key_active,
            ) = self._resolve_daily_generation_route(drawing_settings, user_id, event)
            if route_changed:
                try:
                    prompt = await self.normalize_prompt_for_novelai(
                        prompt_before_normalization,
                        event,
                        drawing_settings["model"],
                        user_prompt=scene_user_prompt,
                        drawing_settings=drawing_settings,
                    )
                except NAIErrors.ParameterError as e:
                    yield event.plain_result(f"❌ 额度路由切换后提示词格式错误: {e}")
                    return

            start_time = time.time()

            # 下载参考图片
            try:
                reference_image = await self.download_and_encode_image(image_urls[0])
                if not reference_image:
                    yield event.plain_result("❌ 无法下载参考图片，请检查图片是否有效")
                    return
            except Exception as img_error:
                logger.error(f"下载参考图片失败: {img_error}")
                yield event.plain_result(f"❌ 下载参考图片失败: {str(img_error)}")
                return

            width, height = map(int, drawing_settings["resolution"].split("x"))

            # 使用临时参数（如果有）或配置中的默认值
            final_strength = (
                temp_strength
                if temp_strength is not None
                else drawing_settings["img2img_strength"]
            )
            final_noise = (
                temp_noise
                if temp_noise is not None
                else drawing_settings["img2img_noise"]
            )

            # 记录参数使用情况
            if temp_strength is not None or temp_noise is not None:
                logger.info(f"使用临时参数: 强度={final_strength}, 噪声={final_noise}")

            # 尝试调用图生图API
            try:
                image_data, meta_info = await self._call_novelai_img2img(
                    prompt=prompt,
                    reference_image=reference_image,
                    width=width,
                    height=height,
                    steps=drawing_settings["steps"],
                    scale=drawing_settings["scale"],
                    strength=final_strength,
                    noise=final_noise,
                    drawing_settings=drawing_settings,
                    api_token=request_api_token,
                    use_newapi=request_uses_newapi,
                )
            except NAIErrors.ServerError as e:
                # 如果遇到500错误，尝试使用更保守的参数
                if "500" in str(e):
                    logger.warning("遇到500错误，尝试使用保守参数重试")
                    conservative_strength = min(
                        final_strength, 0.6
                    )  # 限制强度不超过0.6
                    conservative_noise = min(final_noise, 0.2)  # 限制噪声不超过0.2

                    logger.info(
                        f"重试参数: 强度={conservative_strength}, 噪声={conservative_noise}"
                    )

                    image_data, meta_info = await self._call_novelai_img2img(
                        prompt=prompt,
                        reference_image=reference_image,
                        width=width,
                        height=height,
                        steps=drawing_settings["steps"],
                        scale=drawing_settings["scale"],
                        strength=conservative_strength,
                        noise=conservative_noise,
                        drawing_settings=drawing_settings,
                        api_token=request_api_token,
                        use_newapi=request_uses_newapi,
                    )
                else:
                    # 其他服务器错误直接抛出
                    raise

            async def retry_img2img_with_v45(fallback_prompt: str):
                return await self._call_novelai_img2img(
                    prompt=fallback_prompt,
                    reference_image=reference_image,
                    width=width,
                    height=height,
                    steps=drawing_settings["steps"],
                    scale=drawing_settings["scale"],
                    strength=final_strength,
                    noise=final_noise,
                    drawing_settings=drawing_settings,
                    api_token=None,
                    use_newapi=False,
                )

            (
                image_data,
                meta_info,
                prompt,
                personal_v5_fallback_attempted,
            ) = await self._retry_with_daily_v45_on_personal_access_denied(
                image_data,
                meta_info,
                drawing_settings=drawing_settings,
                personal_key_active=personal_key_active,
                prompt=prompt,
                prompt_before_normalization=prompt_before_normalization,
                event=event,
                scene_user_prompt=scene_user_prompt,
                request_callback=retry_img2img_with_v45,
            )

            if image_data:
                elapsed_time = time.time() - start_time
                # Only shared-token generations consume the group's free quota;
                # personal keys and group authorizations do not.
                quota_scope = (
                    self._resolve_quota_scope(event, user_id)
                    if not personal_key_active or personal_v5_fallback_attempted
                    else None
                )
                self.usage_manager.record_usage(user_id, quota_scope)

                try:
                    final_image_data = self.extract_image_from_response(image_data)

                    if final_image_data:
                        self._cache_generated_image(event, final_image_data)
                        img = None
                        try:
                            img = PILImage.open(io.BytesIO(final_image_data))
                            logger.info(
                                f"图生图验证成功: {img.size}, 格式: {img.format}"
                            )
                        except Exception as e:
                            logger.error(f"图生图验证失败: {e}")
                            raise
                        finally:
                            if img is not None:
                                try:
                                    img.close()
                                except Exception:
                                    pass

                        # 图片内容审核
                        if self.image_audit_enabled and self._should_audit_image(event):
                            (
                                audit_passed,
                                audit_message,
                            ) = await self.audit_image_content(final_image_data, event)
                            if not audit_passed:
                                # 审核不通过，保存到审核失败路径
                                await self._save_audit_failed_image(
                                    final_image_data, "img2img"
                                )
                                yield event.plain_result(f"❌ {audit_message}")
                                return
                            logger.info(f"图生图图片审核通过: {audit_message}")

                        # 自动保存图片
                        saved_path = await self._save_image_locally(
                            final_image_data,
                            "img2img",
                        )
                        if saved_path:
                            logger.info(f"图片已保存到: {saved_path}")

                        # 根据 show_params 配置决定显示内容
                        if self.show_params:
                            model_version = self.get_model_version_label(
                                drawing_settings["model"]
                            )

                            # 显示不同的标题根据原始命令
                            original_command = (
                                "画图"
                                if event.message_str.strip().startswith(
                                    ("/画图", "画图")
                                )
                                else "图生图"
                            )
                            title = (
                                "🖼️ 图生图完成！"
                                if original_command == "图生图"
                                else "🖼️ 图生图完成！（自动切换）"
                            )

                            # 构建结果信息
                            style_info = ""
                            if detected_style:
                                style_info = f" | 画风: {detected_style}"
                            elif (
                                drawing_settings["style_enabled"]
                                and drawing_settings["style_name"]
                            ):
                                style_info = (
                                    f" | 画风: {drawing_settings['style_name']}"
                                )

                            negative_info = ""
                            if "custom_negative_prompt" in drawing_settings:
                                negative_info = " | 负面: --uc 自定义"

                            chain = [
                                Comp.At(qq=event.get_sender_id()),
                                *(
                                    [
                                        Comp.Plain(
                                            "\n"
                                            + Constants.PERSONAL_V5_FALLBACK_NOTICE
                                            + "\n"
                                        )
                                    ]
                                    if personal_v5_fallback_attempted
                                    else []
                                ),
                                Comp.Plain(f"\n{title}\n"),
                                Comp.Plain(
                                    f"📝 提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
                                ),
                                Comp.Plain(
                                    f"⚙️ 强度: {final_strength} | 噪声: {final_noise} | {drawing_settings['steps']}步 | {drawing_settings['resolution']}{style_info}{negative_info}\n"
                                ),
                                Comp.Plain(
                                    f"🤖 模型: {drawing_settings['model']} ({model_version})\n"
                                ),
                                Comp.Image.fromBytes(final_image_data),
                                Comp.Plain(f"\n🕐 耗时: {elapsed_time:.1f}秒"),
                            ]
                        else:
                            # 简洁模式：只显示图片和时间
                            chain = [
                                Comp.At(qq=event.get_sender_id()),
                                *(
                                    [
                                        Comp.Plain(
                                            "\n"
                                            + Constants.PERSONAL_V5_FALLBACK_NOTICE
                                            + "\n"
                                        )
                                    ]
                                    if personal_v5_fallback_attempted
                                    else []
                                ),
                                Comp.Plain("\n"),
                                Comp.Image.fromBytes(final_image_data),
                                Comp.Plain(f"\n🕐 {elapsed_time:.1f}秒"),
                            ]

                        # 记录图生图成功日志
                        self.log_draw(f"图生图成功: 耗时 {elapsed_time:.1f}秒", 1)

                        # 使用自动撤回功能发送图片
                        async for result in self._send_image_with_auto_recall(
                            event, chain
                        ):
                            yield result

                        # 记录这次的图生图命令（在成功生成后记录完整信息）
                        # 构造完全处理后的命令（包含预设画风）
                        full_processed_command = f"/图生图 {prompt}"

                        self._record_draw_command(
                            user_id,
                            full_processed_command,  # 完全处理后的命令
                            "image_to_image",
                            self._get_last_image_url(event),
                            temp_strength,
                            temp_noise,
                            original_message,  # 用户真正的原始输入
                            event,  # 传入event对象检查是否是重画
                        )
                        logger.info(
                            f"已记录图生图命令用于重画: {clean_command_for_redraw}"
                        )
                    else:
                        yield event.plain_result("❌ 无法解析返回的图片数据")

                except Exception as img_error:
                    logger.error(f"图生图处理错误: {img_error}")
                    yield event.plain_result(f"❌ 图片处理失败: {str(img_error)}")
            else:
                # 处理不同类型的错误
                error_message = "❌ 图生图失败"
                if "error_type" in meta_info:
                    error_type = meta_info["error_type"]
                    if error_type == "network":
                        # 网络错误已经在错误类中优化过了，直接使用
                        error_message = f"❌ {meta_info['error']}"
                    elif error_type == "auth":
                        error_message = "❌ API认证失败，请检查Token是否正确"
                    elif error_type == "quota":
                        error_message = "❌ API使用次数超限，请稍后再试"
                    elif error_type == "server":
                        # 服务器错误已经在错误类中优化过了，直接使用
                        error_message = f"❌ {meta_info['error']}"
                    elif error_type == "config":
                        error_message = f"❌ 配置错误: {meta_info['error']}"
                    elif error_type == "param":
                        error_message = f"❌ 参数错误: {meta_info['error']}"
                    else:
                        error_message = "❌ 图生图失败，请稍后重试"
                else:
                    error_message = "❌ 图生图失败，请稍后重试"

                if personal_v5_fallback_attempted:
                    error_message = (
                        Constants.PERSONAL_V5_FALLBACK_NOTICE + "\n" + error_message
                    )
                yield event.plain_result(error_message)

        except Exception as e:
            logger.error(f"NovelAI图生图错误: {e}")
            yield event.plain_result(f"❌ 生成出错: {str(e)}")
        finally:
            self.usage_manager.finish_processing(request_id)
            # 开始处理队列中的下一个请求
            next_request_id = await self.usage_manager.start_next_in_queue()
            if next_request_id:
                logger.info(f"开始处理队列中的下一个请求: {next_request_id}")

    # ========== 命令定义部分 ==========
    # 使用 filter.command 代替 filter.regex
    # filter.command 会自动处理唤醒前缀

    @filter.command("开启NAI画图")
    async def enable_nai_group_drawing(self, event):
        """Share the sender's bound personal key with the current group."""
        group_id = self._current_group_id(event)
        if not group_id or group_id == "0":
            yield event.plain_result("❌ 该命令只能在群聊中使用")
            return

        owner_uid = str(event.get_sender_id())
        if not self._get_bound_user_api_key(
            owner_uid, self._get_event_bot_id(event)
        ):
            yield event.plain_result(
                "❌ 你还没有绑定个人密钥，请先在 astrbot_plugin_ai_video 中使用 /设置API密钥 sk-xxxx"
            )
            return

        payload = self._load_ai_video_user_data()
        payload.setdefault("groups", {})[str(group_id)] = {
            "owner_uid": owner_uid,
            "enabled": True,
        }
        if not self._save_ai_video_user_data(payload):
            yield event.plain_result("❌ 保存本群 NAI 画图授权失败，请检查数据目录权限")
            return
        yield event.plain_result(
            "✅ 已开启本群 NAI 画图授权。\n"
            "本群其他成员可使用你的个人密钥画图；不会修改或复制任何人的个人密钥。"
        )

    @filter.command("关闭NAI画图")
    async def disable_nai_group_drawing(self, event):
        """Revoke the current group's shared personal-key authorization."""
        group_id = self._current_group_id(event)
        if not group_id or group_id == "0":
            yield event.plain_result("❌ 该命令只能在群聊中使用")
            return

        payload = self._load_ai_video_user_data()
        groups = payload.setdefault("groups", {})
        if str(group_id) not in groups:
            yield event.plain_result("当前群未开启 NAI 画图授权。")
            return
        groups.pop(str(group_id), None)
        if not self._save_ai_video_user_data(payload):
            yield event.plain_result("❌ 保存本群 NAI 画图授权失败，请检查数据目录权限")
            return
        yield event.plain_result(
            "✅ 已关闭本群 NAI 画图授权。\n"
            "不会删除或修改授权人的个人密钥。"
        )

    @filter.command("重画")
    async def redraw(self, event):
        """重复上一次的画图命令，支持添加预设"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        user_id = event.get_sender_id()

        # 检查是否有上次的命令
        if user_id not in self.last_draw_commands:
            yield event.plain_result(
                "❌ 没有找到上一次的画图命令\n\n💡 请先使用 /画图 或 /图生图 命令"
            )
            return

        # 解析用户输入的额外预设
        message_text = event.message_str.strip()
        additional_presets = ""
        if message_text.startswith("/重画"):
            additional_presets = message_text[3:].strip()  # 移除 "/重画"
        elif message_text.startswith("重画"):
            additional_presets = message_text[2:].strip()  # 兼容无前缀的情况

        last_cmd = self.last_draw_commands[user_id]

        # 使用保存的用户真正原始输入
        user_original_input = last_cmd.get(
            "user_original_input", last_cmd.get("original_command", last_cmd["command"])
        )

        # 如果有额外预设，需要将其添加到原始命令中
        if additional_presets:
            # 将额外预设添加到原始命令的提示词前面
            if user_original_input.startswith("/画图 "):
                prompt_part = user_original_input[4:].strip()  # 移除 "/画图 "
                event.message_str = f"/画图 {additional_presets}, {prompt_part}"
            elif user_original_input.startswith("/图生图 "):
                prompt_part = user_original_input[5:].strip()  # 移除 "/图生图 "
                event.message_str = f"/图生图 {additional_presets}, {prompt_part}"
            elif user_original_input.startswith("画图 "):
                prompt_part = user_original_input[3:].strip()  # 移除 "画图 "
                event.message_str = f"画图 {additional_presets}, {prompt_part}"
            elif user_original_input.startswith("图生图 "):
                prompt_part = user_original_input[4:].strip()  # 移除 "图生图 "
                event.message_str = f"图生图 {additional_presets}, {prompt_part}"
            else:
                event.message_str = f"{additional_presets}, {user_original_input}"

            logger.info(
                f"重画添加预设: {additional_presets} + 原始命令: {user_original_input}"
            )
        else:
            event.message_str = user_original_input
            logger.info(f"重画使用原始命令: {user_original_input}")

        # 如果有保存的临时参数，添加到命令中
        temp_strength = last_cmd.get("temp_strength")
        temp_noise = last_cmd.get("temp_noise")
        if temp_strength is not None or temp_noise is not None:
            param_str = ""
            if temp_strength is not None:
                param_str += f"-s{temp_strength}"
            if temp_noise is not None:
                param_str += f"-n{temp_noise}"

            # 将参数添加到命令开头（在提示词之前）
            original_cmd = event.message_str
            if original_cmd.startswith("/画图 ") or original_cmd.startswith("/图生图 "):
                cmd_part = original_cmd.split(" ", 1)[0]  # "/画图" 或 "/图生图"
                rest_part = original_cmd.split(" ", 1)[1] if " " in original_cmd else ""
                event.message_str = f"{cmd_part} {param_str} {rest_part}".strip()
                logger.info(f"重画使用临时参数: {param_str}")

        try:
            # 标记这是重画操作，避免覆盖原始记录
            event._is_redraw = True

            # 根据上次的命令类型调用相应的处理方法
            if last_cmd["event_type"] == "text_to_image":
                # 检查是否有图片URL，如果有则转为图生图
                image_url = last_cmd.get("image_url")
                if image_url:
                    try:
                        # 构造包含图片的消息
                        await self._construct_image_message_for_redraw(event, image_url)
                        async for result in self._handle_image_to_image(event):
                            yield result
                    except Exception as e:
                        if "图片链接已过期" in str(e):
                            yield event.plain_result("❌ 图片链接已过期")
                        else:
                            yield event.plain_result(f"❌ 重画失败: {str(e)}")
                else:
                    # 纯文生图
                    async for result in self._handle_text_to_image(event):
                        yield result
            elif last_cmd["event_type"] == "image_to_image":
                # 图生图重画
                image_url = last_cmd.get("image_url")
                if image_url:
                    try:
                        # 构造包含图片的消息
                        await self._construct_image_message_for_redraw(event, image_url)
                        async for result in self._handle_image_to_image(event):
                            yield result
                    except Exception as e:
                        if "图片链接已过期" in str(e):
                            yield event.plain_result("❌ 图片链接已过期")
                        else:
                            yield event.plain_result(f"❌ 重画失败: {str(e)}")
                else:
                    yield event.plain_result(
                        "❌ 重画图生图功能需要原始图片，但图片已丢失\n\n💡 请重新发送图片后使用 /图生图 命令"
                    )
            else:
                yield event.plain_result("❌ 未知的命令类型，无法重画")
        except Exception as e:
            logger.error(f"重画功能执行失败: {e}")
            yield event.plain_result(f"❌ 重画失败: {str(e)}")

    @filter.command("查询预设")
    async def query_preset(self, event):
        """查询指定预设的详细内容"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()
            # 移除命令部分
            if message_text.startswith("/查询预设"):
                preset_name = message_text[5:].strip()
            elif message_text.startswith("查询预设"):
                preset_name = message_text[4:].strip()
            else:
                preset_name = message_text

            if not preset_name:
                yield event.plain_result(
                    "❌ 用法: /查询预设 <预设名>\n\n💡 使用 /预设列表 查看所有可用预设"
                )
                return

            if preset_name not in self.presets:
                yield event.plain_result(f"❌ 预设 '{preset_name}' 不存在")
                return

            preset_content = self.presets[preset_name]
            yield event.plain_result(
                f"📋 预设详情\n\n"
                f"🏷️ 名称: {preset_name}\n"
                f"📝 内容: {preset_content}\n"
                f"📏 长度: {len(preset_content)} 字符"
            )

        except Exception as e:
            logger.error(f"查询预设失败: {e}")
            yield event.plain_result(f"❌ 查询预设失败: {str(e)}")

    @filter.command("查询画风")
    async def query_style(self, event):
        """以示例图卡片查询指定画风，不展示提示词。"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()
            # 移除命令部分
            if message_text.startswith("/查询画风"):
                style_name = message_text[5:].strip()
            elif message_text.startswith("查询画风"):
                style_name = message_text[4:].strip()
            else:
                style_name = message_text

            if not style_name:
                yield event.plain_result(
                    "❌ 用法: /查询画风 <画风名>\n\n💡 使用 /画风列表 查看所有可用画风"
                )
                return

            matched_name = next(
                (
                    name
                    for name in self.style_presets
                    if str(name).casefold() == style_name.casefold()
                ),
                None,
            )
            if matched_name is None:
                yield event.plain_result(f"❌ 画风 '{style_name}' 不存在")
                return

            drawing_settings = self._get_session_drawing_settings(event)
            rendered = await self._render_style_gallery(
                [(matched_name, self.style_presets[matched_name])],
                title=matched_name,
                summary="画风示例",
                drawing_settings=drawing_settings,
                single=True,
            )
            if rendered is not None:
                yield event.chain_result([rendered])
            else:
                # Keep prompt fields out of the fallback response as well.
                yield event.plain_result(
                    f"🎨 画风：{matched_name}\n🖼️ 示例图暂时无法渲染，请稍后重试"
                )

        except Exception as e:
            logger.error(f"查询画风失败: {e}")
            yield event.plain_result(f"❌ 查询画风失败: {str(e)}")

    @filter.command("画图帮助")
    async def drawing_help(self, event):
        """Send a Playwright-rendered guide for the drawing commands."""
        if not self._check_whitelist(event):
            return

        try:
            rendered = await self._render_drawing_help(event)
            yield event.chain_result([rendered])
        except Exception as exc:
            logger.error("画图帮助渲染失败: %s", exc)
            yield event.plain_result(
                "📖 画图帮助\n\n"
                "文生图: /画图 <描述>\n"
                "图生图: 发送或回复图片后使用 /图生图 <描述>；/画图 带图片会自动切换\n"
                "Roll图: /roll图 <描述>（是否限管理员由配置决定）\n"
                "重画: /重画 [额外预设]\n\n"
                "预设: /预设列表、/查询预设 <名称>\n"
                "画风: /画风列表、/查询画风 <名称>、/切换画风 <名称>\n"
                "参数: 长图/宽图/方图、-s<强度>、-n<噪声>、--uc <负面提示词>\n"
                "会话设置: /参数设置、/画图配置、/重置画图参数\n"
                "Roll数量: /roll次数 <1-50>（是否允许使用 /roll图 由管理员配置决定）\n"
                f"渲染图片失败：{exc}"
            )

    # 修复：画图命令使用 command 触发器
    @filter.command("画图")
    async def text_to_image(self, event):
        """根据描述生成图片（支持自动图生图）"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        async for result in self._handle_text_to_image(event):
            yield result

    def _build_generate_image_tool_command(
        self,
        prompt: str,
        character_prompts: list[str] | None = None,
        negative_prompt: str = "",
    ) -> str:
        """Convert function-tool fields into the existing ``/画图`` syntax.

        The command handler remains the single source of truth for presets, session
        settings, quota routing, auditing, saving, and response formatting.

        Args:
            prompt: Shared/base NovelAI prompt or a complete natural-language request.
            character_prompts: Ordered NovelAI Character Prompt fields.
            negative_prompt: Per-request undesired content, excluding the default UC.

        Returns:
            A command string accepted by :meth:`_handle_text_to_image`.
        """
        base_prompt = re.sub(
            r"^\s*Prompt\s*:\s*", "", str(prompt or ""), count=1, flags=re.I
        )
        base_prompt = self._clean_prompt_whitespace(base_prompt).strip(" ,")
        if not base_prompt:
            raise ValueError("prompt 不能为空")

        sections = [base_prompt]
        for item in character_prompts or []:
            character_prompt = re.sub(
                r"^\s*Character\s+\d+\s*:\s*",
                "",
                str(item or ""),
                count=1,
                flags=re.I,
            )
            character_prompt = self._clean_prompt_whitespace(character_prompt).strip(
                " ,"
            )
            if character_prompt:
                sections.append(character_prompt)

        uc = re.sub(
            r"^\s*(?:UC|Undesired\s+Content)\s*:\s*",
            "",
            str(negative_prompt or ""),
            count=1,
            flags=re.I,
        )
        uc = self._clean_prompt_whitespace(uc).strip(" ,。.")
        if uc.casefold() in {
            "用默认预设",
            "默认预设",
            "use default preset",
            "default preset",
        }:
            uc = ""

        command = f"/画图 {' | '.join(sections)}"
        if uc:
            command += f" --uc {uc}"
        return command

    @llm_tool("generate_image")
    async def generate_image(
        self,
        event,
        prompt: str = "",
        character_prompts: list[str] | None = None,
        negative_prompt: str = "",
    ):
        """使用用户提供或明确确认的提示词调用 NovelAI 生成一张图片并直接发送。仅在用户明确要画图、生成图片或出图时调用；如果用户只要讨论构思或查看提示词，不要调用。不要读取 Skill 来构思、翻译、补全或增强提示词，不要自行选择画面方案。工具会复用 /画图 的现有参数、预设、额度、审核、保存和重画流程。

        Args:
            prompt(string): 用户提供或确认后的完整正向提示词，不要添加画师、角色、质量词、风格、动作、场景细节或其它未提供内容。
            character_prompts(list[string]): 仅填写用户明确提供或确认的独立 Character Prompt；按顺序传入，不要自行拆分或补写角色信息。
            negative_prompt(string): 仅传入用户明确提供或确认的 UC 排除项；没有就传空字符串，不要复制默认预设。
        """
        if not self._check_whitelist(event):
            event.stop_event()
            yield event.plain_result("❌ 当前会话没有使用画图功能的权限")
            return

        try:
            command = self._build_generate_image_tool_command(
                prompt,
                character_prompts,
                negative_prompt,
            )
        except ValueError as exc:
            event.stop_event()
            yield event.plain_result(f"❌ 无法调用画图工具: {exc}")
            return

        original_message = event.message_str
        event.message_str = command
        try:
            async for result in self._handle_text_to_image(event):
                yield result
        finally:
            event.message_str = original_message

        # The image/error has already been sent by the command pipeline. Avoid a
        # redundant follow-up completion from the calling chat model.
        event.stop_event()

    @filter.command("roll图")
    async def roll_images(self, event):
        """生成多张图片并以伪造转发形式发送"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "roll图"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        async for result in self._handle_roll_images(event):
            yield result

    @filter.command("roll次数")
    async def set_roll_count(self, event):
        """Set roll image count for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        try:
            value_text = self._get_command_argument(event, "roll次数")
            current = self._get_session_drawing_settings(event)["roll_count"]
            if not value_text:
                yield event.plain_result(
                    f"❌ 请指定roll图次数\n\n📖 用法: /roll次数 <1-50>"
                    f"\n💡 当前会话次数: {current}"
                )
                return

            try:
                count = int(value_text)
                if not 1 <= count <= 50:
                    yield event.plain_result("❌ roll图次数必须在1-50之间")
                    return

                self._set_session_drawing_setting(event, "roll_count", count)
                yield event.plain_result(f"✅ 当前会话roll图次数已设为 {count} 次")

            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字")

        except Exception as e:
            logger.error(f"设置roll图次数失败: {e}")
            yield event.plain_result(f"❌ 设置失败: {str(e)}")

    @filter.command("添加预设")
    async def add_preset(self, event):
        """添加新的提示词预设"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "添加预设"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()
            # 移除命令部分
            if message_text.startswith("/添加预设"):
                args_text = message_text[5:].strip()
            elif message_text.startswith("添加预设"):
                args_text = message_text[4:].strip()
            else:
                args_text = message_text

            # 分割参数：支持可选的分类参数
            # 格式1: /添加预设 <预设名> <提示词>
            # 格式2: /添加预设 <分类> <预设名> <提示词>
            parts = args_text.split(None, 2)

            if len(parts) < 2:
                categories_list = ", ".join(self.preset_categories.keys())
                yield event.plain_result(
                    f"❌ 用法: \n"
                    f"• /添加预设 <预设名> <提示词>\n"
                    f"• /添加预设 <分类> <预设名> <提示词>\n\n"
                    f"📂 可用分类: {categories_list}\n\n"
                    f"例如: /添加预设 人物 可爱女孩 cute girl, kawaii, anime style"
                )
                return

            # 判断是否指定了分类
            if len(parts) == 3 and parts[0] in self.preset_categories:
                # 指定了分类
                category = parts[0]
                preset_name = parts[1]
                preset_prompt = parts[2]
            elif len(parts) >= 2:
                # 没有指定分类，归入"其他"
                category = "其他"
                preset_name = parts[0]
                preset_prompt = " ".join(parts[1:]) if len(parts) > 2 else parts[1]
            else:
                categories_list = ", ".join(self.preset_categories.keys())
                yield event.plain_result(
                    f"❌ 参数不足\n\n"
                    f"用法: \n"
                    f"• /添加预设 <预设名> <提示词>\n"
                    f"• /添加预设 <分类> <预设名> <提示词>\n\n"
                    f"📂 可用分类: {categories_list}"
                )
                return

            if not preset_name or not preset_prompt:
                yield event.plain_result("❌ 预设名和提示词不能为空")
                return

            # 检查预设名是否已存在
            if preset_name in self.presets:
                # 单次回复，只显示原内容
                result_message = (
                    f"⚠️ 预设 '{preset_name}' 已存在，预设将被覆盖\n"
                    f"📂 分类: {category}\n"
                    f"原内容: {self.presets[preset_name]}"
                )
            else:
                # 新预设
                result_message = (
                    f"✅ 预设 '{preset_name}' 添加成功！\n📂 分类: {category}"
                )

            # 添加或更新预设
            self.presets[preset_name] = preset_prompt
            # 添加到分类
            self.add_preset_to_category(preset_name, category)
            # 保存预设到配置文件
            self.save_presets()

            yield event.plain_result(result_message)

        except Exception as e:
            logger.error(f"添加预设失败: {e}")
            yield event.plain_result(f"❌ 添加预设失败: {str(e)}")

    @filter.command("添加人物")
    async def add_character_preset(self, event):
        """添加人物分类的预设"""
        # 添加调试日志
        logger.info(f"添加人物命令被触发: {event.message_str}")
        async for result in self._add_category_preset(event, "人物", "添加人物"):
            yield result

    @filter.command("添加衣服")
    async def add_clothing_preset(self, event):
        """添加衣服分类的预设"""
        async for result in self._add_category_preset(event, "衣服", "添加衣服"):
            yield result

    @filter.command("添加动作")
    async def add_action_preset(self, event):
        """添加动作分类的预设"""
        async for result in self._add_category_preset(event, "动作", "添加动作"):
            yield result

    @filter.command("添加场景")
    async def add_scene_preset(self, event):
        """添加场景分类的预设"""
        async for result in self._add_category_preset(event, "场景", "添加场景"):
            yield result

    @filter.command("添加风格")
    async def add_style_preset_category(self, event):
        """添加风格分类的预设"""
        async for result in self._add_category_preset(event, "风格", "添加风格"):
            yield result

    @filter.command("添加其他")
    async def add_other_preset(self, event):
        """添加其他分类的预设"""
        async for result in self._add_category_preset(event, "其他", "添加其他"):
            yield result

    async def _add_category_preset(self, event, category: str, command_name: str):
        """通用的分类预设添加方法"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, command_name):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 从消息中提取预设名和内容
            message_text = event.message_str.strip()

            # 移除命令部分
            if message_text.startswith(f"/{command_name}"):
                content = message_text[len(f"/{command_name}") :].strip()
            elif message_text.startswith(command_name):
                content = message_text[len(command_name) :].strip()
            else:
                content = message_text

            if not content:
                yield event.plain_result(
                    f"❌ 请提供预设内容\n\n📖 用法: /{command_name} <预设名> <预设内容>\n💡 例如: /{command_name} 美少女 1girl, beautiful, cute"
                )
                return

            # 解析预设名和内容
            parts = content.split(" ", 1)
            if len(parts) < 2:
                yield event.plain_result(
                    f"❌ 格式错误\n\n📖 用法: /{command_name} <预设名> <预设内容>\n💡 例如: /{command_name} 美少女 1girl, beautiful, cute"
                )
                return

            preset_name = parts[0].strip()
            preset_prompt = parts[1].strip()

            if not preset_name or not preset_prompt:
                yield event.plain_result(
                    f"❌ 预设名和内容不能为空\n\n📖 用法: /{command_name} <预设名> <预设内容>"
                )
                return

            # 检查预设名是否已存在
            if preset_name in self.presets:
                # 单次回复，只显示原内容
                result_message = (
                    f"⚠️ 预设 '{preset_name}' 已存在，预设将被覆盖\n"
                    f"📂 分类: {category}\n"
                    f"原内容: {self.presets[preset_name]}"
                )
            else:
                result_message = f"✅ 预设 '{preset_name}' 已添加到 '{category}' 分类"

            # 添加预设
            self.presets[preset_name] = preset_prompt

            # 保存预设
            self.config["presets"] = json.dumps(self.presets, ensure_ascii=False)

            # 添加到指定分类
            self.add_preset_to_category(preset_name, category)

            # 保存配置
            self.config.save_config()

            yield event.plain_result(result_message)

        except Exception as e:
            logger.error(f"{command_name}失败: {e}")
            yield event.plain_result(f"❌ {command_name}失败: {str(e)}")

    @filter.command("删除预设")
    async def delete_preset(self, event):
        """删除指定的提示词预设"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "删除预设"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()
            # 移除命令部分
            if message_text.startswith("/删除预设"):
                preset_name = message_text[5:].strip()
            elif message_text.startswith("删除预设"):
                preset_name = message_text[4:].strip()
            else:
                preset_name = message_text

            if not preset_name:
                yield event.plain_result("❌ 用法: /删除预设 <预设名>")
                return

            if preset_name not in self.presets:
                yield event.plain_result(f"❌ 预设 '{preset_name}' 不存在")
                return

            self.presets.pop(preset_name)
            # 从分类中移除
            self.remove_preset_from_categories(preset_name)
            # 重要：保存更新后的预设到配置文件
            self.save_presets()

            yield event.plain_result(f"✅ 预设 '{preset_name}' 删除成功！")

        except Exception as e:
            logger.error(f"删除预设失败: {e}")
            yield event.plain_result(f"❌ 删除预设失败: {str(e)}")

    @filter.command("预设列表")
    async def list_presets(self, event):
        """显示所有可用的提示词预设"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        if not self.presets:
            yield event.plain_result(
                "📋 当前没有任何预设\n\n💡 使用 /添加预设 <名称> <内容> 来添加预设"
            )
            return

        # 准备预设列表内容
        preset_names = list(self.presets.keys())
        preset_list_content = f"📋 预设列表: {', '.join(preset_names)}"

        # 使用方法说明
        usage_content = (
            "💡 使用方法:\n"
            "• /画图 <预设名>\n"
            "• /画图 <预设名> <预设名2> <额外描述>\n"
            "• /图生图 <预设名>\n"
            "• /查询预设 <预设名> - 查看详细内容"
        )

        # 使用伪造转发
        # 创建2条消息的伪造转发
        messages = [
            {"uin": 2854196310, "name": "预设管理器", "content": preset_list_content},
            {"uin": 2854196311, "name": "使用说明", "content": usage_content},
        ]
        forward_chain = self._create_multi_fake_forward_message(messages)
        yield event.chain_result(forward_chain)

    @filter.command("全部预设列表")
    async def list_all_presets_by_category(self, event):
        """按分类显示所有预设"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        if not self.presets:
            yield event.plain_result(
                "📋 当前没有任何预设\n\n💡 使用 /添加预设 <名称> <内容> 来添加预设"
            )
            return

        result_text = "📋 全部预设列表 (按分类)\n\n"

        # 显示各分类的预设
        for category, preset_names in self.preset_categories.items():
            if preset_names:  # 只显示有预设的分类
                valid_presets = [name for name in preset_names if name in self.presets]
                if valid_presets:
                    result_text += f"📂 **{category}** ({len(valid_presets)}个)\n"
                    result_text += f"({', '.join(valid_presets)})\n\n"

        # 显示未分类的预设
        uncategorized = self.get_uncategorized_presets()
        if uncategorized:
            result_text += f"📂 **未分类** ({len(uncategorized)}个)\n"
            result_text += f"({', '.join(uncategorized.keys())})\n\n"

        result_text += f"💡 总计: {len(self.presets)} 个预设\n"
        result_text += "使用方法:\n• /人物预设列表 - 查看人物分类\n• /查询预设 <名称> - 查看详细内容"

        # 使用伪造转发
        forward_chain = self._create_fake_forward_message(
            "全部预设列表", result_text, "预设管理器"
        )
        yield event.chain_result(forward_chain)

    @filter.command("人物预设列表")
    async def list_character_presets(self, event):
        """显示人物分类的预设"""
        async for result in self._list_category_presets(event, "人物", "人物预设列表"):
            yield result

    @filter.command("衣服预设列表")
    async def list_clothing_presets(self, event):
        """显示衣服分类的预设"""
        async for result in self._list_category_presets(event, "衣服", "衣服预设列表"):
            yield result

    @filter.command("动作预设列表")
    async def list_action_presets(self, event):
        """显示动作分类的预设"""
        async for result in self._list_category_presets(event, "动作", "动作预设列表"):
            yield result

    @filter.command("场景预设列表")
    async def list_scene_presets(self, event):
        """显示场景分类的预设"""
        async for result in self._list_category_presets(event, "场景", "场景预设列表"):
            yield result

    @filter.command("风格预设列表")
    async def list_style_presets_category(self, event):
        """显示风格分类的预设"""
        async for result in self._list_category_presets(event, "风格", "风格预设列表"):
            yield result

    @filter.command("其他预设列表")
    async def list_other_presets(self, event):
        """显示其他分类的预设"""
        async for result in self._list_category_presets(event, "其他", "其他预设列表"):
            yield result

    async def _list_category_presets(
        self, event, category: str, command_name: str = ""
    ):
        """通用的分类预设列表显示方法"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        category_presets = self.get_presets_by_category(category)

        if not category_presets:
            yield event.plain_result(
                f"📂 {category}分类暂无预设\n\n💡 使用 /添加预设 {category} <名称> <内容> 来添加{category}预设"
            )
            return

        result_text = f"📂 {category}预设列表\n\n"

        # 简洁的名称列表显示
        preset_names = list(category_presets.keys())
        result_text += f"({', '.join(preset_names)})\n\n"

        result_text += f"💡 共 {len(category_presets)} 个{category}预设\n"
        result_text += f"• /查询预设 <名称> - 查看完整内容\n"
        result_text += f"• /全部预设列表 - 查看所有分类"

        # 使用伪造转发
        title = command_name if command_name else f"{category}预设列表"
        forward_chain = self._create_fake_forward_message(
            title, result_text, f"{category}预设管理"
        )
        yield event.chain_result(forward_chain)

    @filter.command("添加预设分类")
    async def add_preset_category(self, event):
        """添加新的预设分类"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "添加预设分类"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()
            # 移除命令部分
            if message_text.startswith("/添加预设分类"):
                category_name = message_text[7:].strip()
            elif message_text.startswith("添加预设分类"):
                category_name = message_text[6:].strip()
            else:
                category_name = message_text

            if not category_name:
                yield event.plain_result(
                    "❌ 用法: /添加预设分类 <分类名>\n例如: /添加预设分类 表情"
                )
                return

            if category_name in self.preset_categories:
                yield event.plain_result(f"❌ 分类 '{category_name}' 已存在")
                return

            # 添加新分类
            self.preset_categories[category_name] = []
            self.save_preset_categories()

            yield event.plain_result(f"✅ 预设分类 '{category_name}' 添加成功！")

        except Exception as e:
            logger.error(f"添加预设分类失败: {e}")
            yield event.plain_result(f"❌ 添加预设分类失败: {str(e)}")

    @filter.command("预设分类列表")
    async def list_preset_categories(self, event):
        """显示所有预设分类"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        if not self.preset_categories:
            yield event.plain_result("📂 当前没有任何预设分类")
            return

        result_text = "📂 预设分类列表\n\n"

        for category, presets in self.preset_categories.items():
            result_text += f"• **{category}** ({len(presets)}个预设)\n"

        result_text += f"\n💡 总计: {len(self.preset_categories)} 个分类\n"
        result_text += "• /添加预设分类 <名称> - 添加新分类\n"
        result_text += "• /<分类名>预设列表 - 查看分类预设"

        yield event.plain_result(result_text)

    @filter.command("看参数")
    async def extract_nai_metadata(self, event):
        """从生成的图片中提取生成参数"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        try:
            # Prefer an explicitly supplied/replied image. When /看参数 is
            # sent by itself after generation, use the untouched API bytes
            # cached by the generation pipeline instead of the platform preview.
            image_urls = []
            current_images = self.extract_images_from_messages(
                event.message_obj.message, include_files=True
            )
            image_urls.extend(current_images)

            if not image_urls:
                reply_images = await self._get_reply_message_images_for_metadata(event)
                image_urls.extend(reply_images)

            cached_image_data = self._get_cached_generated_image(event)

            if not image_urls and cached_image_data is None:
                yield event.plain_result(
                    "❌ 未检测到图片\n\n📖 用法: 生成图片后直接使用 /看参数，或附带/回复图片"
                )
                return

            image_format = None
            metadata = None
            last_error = None
            image_sources = [("message", image_url) for image_url in image_urls]
            # A quoted platform preview is commonly JPEG even when the
            # original generated image was PNG. Keep explicit external PNGs
            # authoritative, but allow the untouched generation response to
            # rescue a preview after the explicit source fails metadata parsing.
            if cached_image_data is not None:
                image_sources.append(("generated", cached_image_data))
            for source_type, image_source in image_sources:
                try:
                    if source_type == "generated" and image_format == "PNG":
                        # A real PNG was explicitly supplied. Do not replace
                        # an unrelated external file with this session's cache.
                        continue
                    candidate_data = (
                        image_source
                        if source_type == "generated"
                        else await self._download_image_with_client(image_source)
                    )
                    if len(candidate_data) > Constants.MAX_IMAGE_SIZE:
                        yield event.plain_result(
                            f"❌ 图片过大 ({len(candidate_data) / 1024 / 1024:.1f}MB)，超过限制 ({Constants.MAX_IMAGE_SIZE / 1024 / 1024:.1f}MB)"
                        )
                        return

                    with PILImage.open(io.BytesIO(candidate_data)) as candidate:
                        candidate_format = candidate.format or "未知"
                        if image_format is None:
                            image_format = candidate_format
                        if max(candidate.size) > Constants.MAX_IMAGE_DIMENSION:
                            yield event.plain_result(
                                f"❌ 图片尺寸过大 ({candidate.width}x{candidate.height})，超过限制 ({Constants.MAX_IMAGE_DIMENSION}px)"
                            )
                            return
                        try:
                            candidate_metadata = self._extract_novelai_metadata(candidate)
                        except (MetadataExtractionError, ImageProcessingError) as exc:
                            last_error = exc
                            continue

                    image_format = candidate_format
                    metadata = candidate_metadata
                    break
                except NAIErrors.NetworkError as exc:
                    last_error = exc
                except Exception as exc:
                    last_error = exc

            if metadata is None:
                if image_format and image_format != "PNG":
                    yield event.plain_result(
                        f"❌ 当前图片格式为 {image_format}，该图片不是可提取 NovelAI 元数据的 PNG"
                    )
                elif last_error:
                    logger.debug(f"提取NovelAI元数据失败: {last_error}")
                    yield event.plain_result("❌ 该图片不包含参数")
                else:
                    yield event.plain_result("❌ 该图片不包含参数")
                return

            # 使用伪造转发
            forward_chain = self._create_nai_metadata_forward_message(metadata)
            yield event.chain_result(forward_chain)

        except Exception as e:
            logger.error(f"看参数命令失败: {e}")
            yield event.plain_result(f"❌ 命令执行失败: {str(e)}")

    def _extract_novelai_metadata(self, image: PILImage.Image) -> dict:
        """提取NovelAI图片的元数据"""

        # PNG text chunks are the portable metadata format used by the web UI.
        # Check them before the legacy alpha-channel steganography decoder.
        for key in ("Comment", "Description", "parameters", "Prompt"):
            value = image.info.get(key)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if not isinstance(value, str) or not value.strip():
                continue
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                if "Comment" not in parsed:
                    parsed = {"Comment": parsed}
                elif isinstance(parsed["Comment"], str):
                    try:
                        parsed["Comment"] = json.loads(parsed["Comment"])
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(parsed["Comment"], dict):
                        continue
                return parsed

        if image.format != "PNG":
            raise MetadataExtractionError("只支持PNG格式的图片")

        # 转换为numpy数组
        image_array = np.array(image.convert("RGBA"))

        if image.mode not in ("RGBA", "LA", "P"):
            raise MetadataExtractionError("图片不包含NovelAI隐藏参数")

        # 提取LSB数据
        alpha = image_array[..., -1]  # 获取Alpha通道
        alpha = alpha.T.reshape((-1,))
        alpha = alpha[: (alpha.shape[0] // 8) * 8]
        alpha = np.bitwise_and(alpha, 1)
        alpha = alpha.reshape((-1, 8))
        alpha = np.packbits(alpha, axis=1)

        # 创建数据读取器
        data = alpha
        pos = 0

        def get_next_n_bytes(n):
            nonlocal pos
            n_bytes = data[pos : pos + n]
            pos += n
            return bytearray(n_bytes)

        def read_32bit_integer():
            bytes_list = get_next_n_bytes(4)
            if len(bytes_list) == 4:
                return int.from_bytes(bytes_list, byteorder="big")
            return None

        # 验证魔数
        magic = "stealth_pngcomp"
        try:
            read_magic = get_next_n_bytes(len(magic)).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MetadataExtractionError("不是有效的NovelAI图片") from exc
        if magic != read_magic:
            raise MetadataExtractionError(
                "不是有效的NovelAI图片，缺少stealth pnginfo标识"
            )

        # 读取JSON数据长度
        read_len_bits = read_32bit_integer()
        if read_len_bits is None or read_len_bits % 8:
            raise MetadataExtractionError("无效的元数据长度")
        read_len = read_len_bits // 8
        if read_len <= 0:
            raise MetadataExtractionError("无效的元数据长度")

        # 读取并解压JSON数据
        json_data = get_next_n_bytes(read_len)
        try:
            json_data = json.loads(gzip.decompress(json_data).decode("utf-8"))
        except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
            raise MetadataExtractionError("NovelAI隐藏元数据损坏") from exc

        if not isinstance(json_data, dict):
            raise MetadataExtractionError("NovelAI隐藏元数据格式无效")

        # 处理Comment字段
        if "Comment" in json_data and isinstance(json_data["Comment"], str):
            try:
                json_data["Comment"] = json.loads(json_data["Comment"])
            except (TypeError, ValueError) as exc:
                raise MetadataExtractionError("NovelAI Comment元数据格式无效") from exc
            if not isinstance(json_data["Comment"], dict):
                raise MetadataExtractionError("NovelAI Comment元数据格式无效")

        return json_data

    def _format_novelai_metadata(self, metadata: dict) -> str:
        """格式化NovelAI元数据为可读文本"""
        result_lines = []

        # 基本信息
        if "Software" in metadata:
            result_lines.append(f"🎨 生成软件: {metadata['Software']}")

        if "Source" in metadata:
            result_lines.append(f"🌐 来源: {metadata['Source']}")

        # 处理Comment字段（包含主要参数）
        if "Comment" in metadata and isinstance(metadata["Comment"], dict):
            comment = metadata["Comment"]

            # 提示词
            if "prompt" in comment:
                result_lines.append(f"\n📝 正面提示词:")
                result_lines.append(f"{comment['prompt']}")

            # 负面提示词
            if "uc" in comment:
                result_lines.append(f"\n🚫 负面提示词:")
                result_lines.append(f"{comment['uc']}")

            # 生成参数
            result_lines.append(f"\n⚙️ 生成参数:")

            if "steps" in comment:
                result_lines.append(f"• 步数: {comment['steps']}")

            if "scale" in comment:
                result_lines.append(f"• 引导强度: {comment['scale']}")

            if "cfg_rescale" in comment:
                result_lines.append(f"• CFG重缩放: {comment['cfg_rescale']}")

            if "seed" in comment:
                result_lines.append(f"• 种子: {comment['seed']}")

            if "sampler" in comment:
                result_lines.append(f"• 采样器: {comment['sampler']}")

            if "noise_schedule" in comment:
                result_lines.append(f"• 噪声调度: {comment['noise_schedule']}")

            # 图片尺寸
            if "width" in comment and "height" in comment:
                result_lines.append(f"• 尺寸: {comment['width']}x{comment['height']}")

            # 模型信息
            if "model" in comment:
                result_lines.append(f"• 模型: {comment['model']}")

            # 图生图参数
            if "strength" in comment:
                result_lines.append(f"• 图生图强度: {comment['strength']}")

            if "noise" in comment:
                result_lines.append(f"• 图生图噪声: {comment['noise']}")

            # 其他参数
            if "sm" in comment:
                result_lines.append(f"• SM: {comment['sm']}")

            if "sm_dyn" in comment:
                result_lines.append(f"• SM动态: {comment['sm_dyn']}")

        # 其他元数据
        other_fields = ["Title", "Author", "Copyright"]
        for field in other_fields:
            if field in metadata and metadata[field]:
                result_lines.append(f"\n📋 {field}: {metadata[field]}")

        if not result_lines:
            return "❌ 未找到有效的生成参数"

        return "\n".join(result_lines)

    def _extract_v4_or_legacy_comment_field(
        self,
        comment: dict,
        *,
        v4_key: str,
        legacy_key: str,
        fallback: str,
        log_label: str,
    ) -> str:
        """Extract a V4 caption field and fall back to the legacy metadata key."""
        try:
            v4_value = comment.get(v4_key)
            if v4_value and isinstance(v4_value, dict):
                caption = v4_value.get("caption", {})
                if isinstance(caption, dict):
                    base_caption = caption.get("base_caption", "")
                    if base_caption:
                        return base_caption

            return comment.get(legacy_key, fallback)
        except Exception as e:
            logger.error(f"提取{log_label}失败: {e}")
            return comment.get(legacy_key, fallback)

    def _extract_prompt_content(self, comment: dict) -> str:
        """从comment中提取正面提示词内容，支持v4格式"""
        return self._extract_v4_or_legacy_comment_field(
            comment,
            v4_key="v4_prompt",
            legacy_key="prompt",
            fallback="未找到正面提示词",
            log_label="正面提示词",
        )

    def _extract_negative_content(self, comment: dict) -> str:
        """从comment中提取负面提示词内容，支持v4格式"""
        return self._extract_v4_or_legacy_comment_field(
            comment,
            v4_key="v4_negative_prompt",
            legacy_key="uc",
            fallback="未找到负面提示词",
            log_label="负面提示词",
        )

    def _extract_character_captions(self, comment: dict) -> list:
        """从comment中提取Character提示词列表，支持v4格式"""
        try:
            char_captions = []

            # 尝试v4格式
            v4_prompt = comment.get("v4_prompt")
            if v4_prompt and isinstance(v4_prompt, dict):
                caption = v4_prompt.get("caption", {})
                if isinstance(caption, dict):
                    char_captions_data = caption.get("char_captions", [])
                    if isinstance(char_captions_data, list):
                        for char_data in char_captions_data:
                            if isinstance(char_data, dict):
                                char_caption = char_data.get("char_caption", "")
                                if char_caption and char_caption.strip():
                                    char_captions.append(char_caption)

            # 如果v4格式没有找到，尝试旧格式的Character字段
            if not char_captions:
                import re

                char_dict = {}  # 使用字典收集，避免稀疏数组

                for key in comment.keys():
                    patterns = [
                        r"^Character\s*(\d+)\s*Prompt$",
                        r"^character\s*(\d+)\s*prompt$",
                        r"^Character(\d+)Prompt$",
                        r"^character(\d+)prompt$",
                    ]

                    for pattern in patterns:
                        match = re.match(pattern, key, re.IGNORECASE)
                        if match:
                            char_num = int(match.group(1))
                            char_content = comment.get(key, "")
                            if char_content and char_content.strip():
                                char_dict[char_num] = char_content
                            break

                # 按顺序转换为列表
                if char_dict:
                    max_num = max(char_dict.keys())
                    for i in range(1, max_num + 1):
                        char_captions.append(char_dict.get(i, ""))

            return char_captions
        except Exception as e:
            logger.error(f"提取Character提示词失败: {e}")
            return []

    def _create_nai_metadata_forward_message(self, metadata: dict) -> List:
        """
        创建NAI参数的多条消息伪造转发，支持Character角色：
        1. 正面提示词标题
        2. 正面提示词内容
        3. Character 1 Prompt: [内容] (如果存在)
        4. Character 2 Prompt: [内容] (如果存在)
        5. Character 3 Prompt: [内容] (如果存在)
        6. 🚫 负面提示词: [内容] (标题+内容合并)
        7. ⚙️ 生成参数: [参数列表]
        """
        try:
            messages = []

            # 处理Comment字段（包含主要参数）
            if "Comment" in metadata and isinstance(metadata["Comment"], dict):
                comment = metadata["Comment"]

                # 第1条：正面提示词标题
                messages.append(
                    {
                        "uin": 2854196310,
                        "name": "NovelAI分析",
                        "content": "📝 正面提示词:",
                    }
                )

                # 第2条：正面提示词内容 (处理v4格式的嵌套结构)
                prompt_content = self._extract_prompt_content(comment)
                messages.append(
                    {"uin": 2854196311, "name": "提示词内容", "content": prompt_content}
                )

                # Character检测：处理v4格式的char_captions数组
                character_count = 0

                # 提取v4格式的Character信息
                char_captions = self._extract_character_captions(comment)

                for i, char_caption in enumerate(char_captions, 1):
                    if char_caption and char_caption.strip():
                        character_count += 1
                        messages.append(
                            {
                                "uin": 2854196310 + 1 + character_count,
                                "name": f"角色 {i}",
                                "content": f"Character {i} Prompt: {char_caption}",
                            }
                        )

                # 负面提示词 (新格式：标题+内容合并为一条，处理v4格式)
                uc_content = self._extract_negative_content(comment)
                uc_uin = 2854196312 + character_count
                messages.append(
                    {
                        "uin": uc_uin,
                        "name": "负面提示词",
                        "content": f"🚫 负面提示词: {uc_content}",
                    }
                )

                # 生成参数 (UIN需要根据character数量调整)
                param_lines = ["⚙️ 生成参数:"]

                if "steps" in comment:
                    param_lines.append(f"• 步数: {comment['steps']}")
                if "scale" in comment:
                    param_lines.append(f"• 引导强度: {comment['scale']}")
                if "cfg_rescale" in comment:
                    param_lines.append(f"• CFG重缩放: {comment['cfg_rescale']}")
                if "seed" in comment:
                    param_lines.append(f"• 种子: {comment['seed']}")
                if "sampler" in comment:
                    param_lines.append(f"• 采样器: {comment['sampler']}")
                if "noise_schedule" in comment:
                    param_lines.append(f"• 噪声调度: {comment['noise_schedule']}")
                if "width" in comment and "height" in comment:
                    param_lines.append(
                        f"• 尺寸: {comment['width']}x{comment['height']}"
                    )
                if "model" in comment:
                    param_lines.append(f"• 模型: {comment['model']}")
                if "strength" in comment:
                    param_lines.append(f"• 图生图强度: {comment['strength']}")
                if "noise" in comment:
                    param_lines.append(f"• 图生图噪声: {comment['noise']}")
                if "sm" in comment:
                    param_lines.append(f"• SM: {comment['sm']}")
                if "sm_dyn" in comment:
                    param_lines.append(f"• SM动态: {comment['sm_dyn']}")

                param_content = "\n".join(param_lines)
                param_uin = (
                    2854196313 + character_count
                )  # 调整UIN，因为负面提示词现在只占一条
                messages.append(
                    {"uin": param_uin, "name": "生成参数", "content": param_content}
                )

            # 如果没有有效数据，返回错误消息
            if not messages:
                messages = [
                    {
                        "uin": 2854196310,
                        "name": "NovelAI分析",
                        "content": "❌ 未找到有效的生成参数",
                    }
                ]

            # 使用多条消息伪造转发方法
            return self._create_multi_fake_forward_message(messages)

        except Exception as e:
            logger.error(f"创建NAI参数伪造转发消息失败: {e}")
            # 如果失败，返回普通文本格式
            result_text = self._format_novelai_metadata(metadata)
            return [Comp.Plain(f"📋 图片参数\n\n{result_text}")]

    async def _handle_roll_images(self, event):
        """处理roll图的核心逻辑"""
        user_id = event.get_sender_id()
        drawing_settings = self._get_session_drawing_settings(event)
        # 先获取原始消息，解析临时参数，再进行预设识别
        original_message = event.message_str.strip()  # 保存原始消息

        # 从原始消息中解析临时参数
        temp_strength = None
        temp_noise = None
        custom_negative_prompt = None
        if original_message:
            cleaned_message, temp_strength, temp_noise, custom_negative_prompt = (
                self._parse_temp_img2img_params(original_message)
            )
            if custom_negative_prompt is not None:
                drawing_settings["custom_negative_prompt"] = custom_negative_prompt
            # 只在有临时参数时才记录日志
            if (
                temp_strength is not None
                or temp_noise is not None
                or custom_negative_prompt is not None
            ):
                self.log_roll(
                    "roll图临时参数: "
                    f"强度={temp_strength}, 噪声={temp_noise}, "
                    f"自定义UC={'是' if custom_negative_prompt is not None else '否'}",
                    5,
                )
            # 更新事件消息，移除临时参数
            event.message_str = cleaned_message

        _, scene_user_prompt = self._extract_command_prompt(event.message_str.strip())
        scene_user_prompt = self._clean_prompt_whitespace(scene_user_prompt)

        # 现在进行预设识别和提示词提取
        prompt, _ = self.extract_prompt_from_message(
            event, drawing_settings=drawing_settings
        )

        # 检查是否有图片（roll图不支持图生图）
        has_current_images = self._has_images_in_message(event)
        has_reply_images = self._check_reply_has_images(event)

        if has_current_images or has_reply_images:
            yield event.plain_result(
                "❌ roll图功能不支持图生图模式\n\n💡 请使用纯文本描述进行roll图"
            )
            return

        # 处理衣服标签
        if prompt:
            processed_prompt = self.clothing_manager.process_prompt(prompt)
            if processed_prompt != prompt:
                prompt = processed_prompt
                self.log_roll("roll图处理: 衣服标签已应用", 4)

        if (
            not str(self.api_token or "").strip()
            and not self._get_bound_user_api_key(
                user_id, self._get_event_bot_id(event)
            )
            and not self._get_group_authorized_api_key(event)
        ):
            yield event.plain_result(
                "❌ 请先在管理面板配置 NovelAI Token，或绑定个人密钥/开启本群 NAI 画图授权"
            )
            return

        if not drawing_settings["model"]:
            yield event.plain_result("❌ 请先在管理面板配置模型名称")
            return

        if not prompt.strip():
            yield event.plain_result(
                "❌ 请提供图片描述\n用法: /roll图 描述 或 /roll图 预设名\n💡 临时参数: -s<强度> -n<噪声> (如: /roll图 -s0.6 美女)"
            )
            return

        prompt_before_normalization = prompt
        self._resolve_daily_generation_route(drawing_settings, user_id, event)
        try:
            prompt = await self.normalize_prompt_for_novelai(
                prompt_before_normalization,
                event,
                drawing_settings["model"],
                user_prompt=scene_user_prompt,
                drawing_settings=drawing_settings,
            )
        except NAIErrors.ParameterError as e:
            yield event.plain_result(f"❌ 提示词格式错误: {e}")
            return

        # 使用排队系统，roll图也需要排队
        (
            request_id,
            can_process,
            queue_position,
            _,
        ) = await self.usage_manager.add_to_queue(user_id)

        if not can_process:
            queue_message = f"🕒 Roll图排队中，当前第{queue_position}位"
            async for result in self._send_status_message_with_auto_recall(
                event, queue_message
            ):
                yield result

            # 等待轮到这个请求
            has_turn = await self.usage_manager.wait_for_turn(request_id)
            if not has_turn:
                yield event.plain_result("❌ 排队等待超时，请重新提交roll图请求")
                return
            # 不再显示"轮到你了"的提示

        # 显示处理提示（排队完成后）
        async for result in self._send_status_message_with_auto_recall(
            event, "🎨 正在roll图，请稍候..."
        ):
            yield result

        # 开始计时（排除排队时间）
        start_time = time.time()

        # 解析分辨率
        width, height = map(int, drawing_settings["resolution"].split("x"))

        try:
            # 生成多张图片
            generated_images = []
            failed_count = 0
            fallback_error = None
            personal_v5_fallback_attempted = False

            roll_count = drawing_settings["roll_count"]

            # 开始批量生成
            self.log_roll(f"开始roll图批量生成，共 {roll_count} 张", 2)

            for i in range(roll_count):
                (
                    route_changed,
                    request_api_token,
                    request_uses_newapi,
                ) = self._resolve_daily_generation_route(drawing_settings, user_id, event)
                if route_changed:
                    try:
                        prompt = await self.normalize_prompt_for_novelai(
                            prompt_before_normalization,
                            event,
                            drawing_settings["model"],
                            user_prompt=scene_user_prompt,
                            drawing_settings=drawing_settings,
                        )
                    except NAIErrors.ParameterError as e:
                        fallback_error = str(e)
                        failed_count += roll_count - i
                        break

                try:
                    # 只在每5张或最后一张时显示进度
                    if (i + 1) % 5 == 0 or (i + 1) == roll_count:
                        self.log_roll(f"roll图进度: {i + 1}/{roll_count}", 3)

                    # 调用NovelAI API生成图片，带重试机制
                    max_retries = 3
                    retry_delay = 5  # 重试间隔5秒

                    for retry in range(max_retries):
                        try:
                            image_data, meta_info = await self._call_novelai_txt2img(
                                prompt=prompt,
                                width=width,
                                height=height,
                                steps=drawing_settings["steps"],
                                scale=drawing_settings["scale"],
                                sampler=drawing_settings["sampler"],
                                drawing_settings=drawing_settings,
                                api_token=request_api_token,
                                use_newapi=request_uses_newapi,
                            )
                            break  # 成功则跳出重试循环
                        except NAIErrors.QuotaExceededError as e:
                            if retry < max_retries - 1:
                                self.log_warning(
                                    f"roll图API限制，{retry_delay}秒后重试 ({retry + 1}/{max_retries})"
                                )
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # 指数退避
                            else:
                                self.log_error(
                                    f"roll图第 {i + 1} 张重试{max_retries}次后仍然失败"
                                )
                                raise
                        except Exception as e:
                            if retry < max_retries - 1:
                                self.log_warning(
                                    f"roll图生成失败，{retry_delay}秒后重试 ({retry + 1}/{max_retries}): {e}"
                                )
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                raise

                    async def retry_roll_txt2img_with_v45(fallback_prompt: str):
                        return await self._call_novelai_txt2img(
                            prompt=fallback_prompt,
                            width=width,
                            height=height,
                            steps=drawing_settings["steps"],
                            scale=drawing_settings["scale"],
                            sampler=drawing_settings["sampler"],
                            drawing_settings=drawing_settings,
                            api_token=None,
                            use_newapi=False,
                        )

                    (
                        image_data,
                        meta_info,
                        prompt,
                        fallback_attempted_now,
                    ) = await self._retry_with_daily_v45_on_personal_access_denied(
                        image_data,
                        meta_info,
                        drawing_settings=drawing_settings,
                        personal_key_active=personal_key_active,
                        prompt=prompt,
                        prompt_before_normalization=prompt_before_normalization,
                        event=event,
                        scene_user_prompt=scene_user_prompt,
                        request_callback=retry_roll_txt2img_with_v45,
                    )
                    personal_v5_fallback_attempted = (
                        personal_v5_fallback_attempted or fallback_attempted_now
                    )
                    if (
                        fallback_attempted_now
                        and meta_info.get("error_type") == "param"
                    ):
                        fallback_error = meta_info.get("error", "场景参数不兼容")
                        failed_count += roll_count - i
                        break

                    if image_data:
                        # 记录使用次数
                        quota_scope = (
                            self._resolve_quota_scope(event, user_id)
                            if not personal_key_active or fallback_attempted_now
                            else None
                        )
                        self.usage_manager.record_usage(user_id, quota_scope)

                        try:
                            final_image_data = self.extract_image_from_response(
                                image_data
                            )
                            if final_image_data and len(final_image_data) > 0:
                                self._cache_generated_image(event, final_image_data)
                                # 验证图片数据完整性
                                if (
                                    len(final_image_data) < 1024
                                ):  # 小于1KB可能是错误数据
                                    self.log_error(
                                        f"roll图第 {i + 1} 张数据异常，大小仅 {len(final_image_data)} 字节"
                                    )
                                    failed_count += 1
                                    continue

                                # 只在数据异常时记录日志，正常情况不记录

                                # 图片内容审核
                                audit_passed = True
                                if self.image_audit_enabled and self._should_audit_image(
                                    event
                                ):
                                    (
                                        audit_passed,
                                        audit_message,
                                    ) = await self.audit_image_content(
                                        final_image_data, event
                                    )
                                    if not audit_passed:
                                        # 审核不通过，保存到审核失败路径
                                        await self._save_audit_failed_image(
                                            final_image_data, "roll_images"
                                        )
                                        failed_count += 1
                                        self.log_error(
                                            f"roll图第 {i + 1} 张审核失败: {audit_message}"
                                        )
                                        continue
                                    # 审核通过时不记录日志，减少刷屏

                                if audit_passed:
                                    # 自动保存图片
                                    saved_path = await self._save_image_locally(
                                        final_image_data, "roll_images"
                                    )
                                    if saved_path:
                                        logger.info(
                                            f"第 {i + 1} 张图片已保存到: {saved_path}"
                                        )

                                    # 将图片数据添加到结果列表
                                    generated_images.append(
                                        {"image_data": final_image_data, "index": i + 1}
                                    )
                                    # 成功生成时不记录日志，减少刷屏
                            else:
                                failed_count += 1
                                logger.error(f"第 {i + 1} 张图片数据解析失败")
                        except Exception as e:
                            failed_count += 1
                            logger.error(f"第 {i + 1} 张图片处理失败: {e}")
                    else:
                        failed_count += 1
                        error_msg = meta_info.get("error", "未知错误")
                        logger.error(f"第 {i + 1} 张图片生成失败: {error_msg}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"第 {i + 1} 张图片生成异常: {e}")

            # 计算总用时
            elapsed_time = time.time() - start_time

            # 检查是否有成功生成的图片
            if not generated_images:
                if personal_v5_fallback_attempted:
                    yield event.plain_result(Constants.PERSONAL_V5_FALLBACK_NOTICE)
                if fallback_error:
                    yield event.plain_result(
                        f"❌ 额度路由切换后提示词格式错误: {fallback_error}"
                    )
                    return
                yield event.plain_result(
                    f"❌ roll图失败，{roll_count} 张图片全部生成失败"
                )
                return

            if fallback_error:
                yield event.plain_result(
                    f"⚠️ 已生成 {len(generated_images)} 张；其余图片因额度路由切换后的"
                    f"场景限制而停止: {fallback_error}"
                )

            if personal_v5_fallback_attempted:
                yield event.plain_result(Constants.PERSONAL_V5_FALLBACK_NOTICE)

            # 创建roll图结果的伪造转发消息
            success_count = len(generated_images)
            forward_chain = self._create_roll_images_forward_message(
                generated_images, success_count, failed_count, elapsed_time
            )

            yield event.chain_result(forward_chain)

            # 记录roll图完成日志
            if generated_images:
                self.log_roll(
                    f"roll图完成: 成功 {len(generated_images)} 张，失败 {failed_count} 张，用时 {elapsed_time:.1f}秒",
                    1,
                )

        except Exception as e:
            logger.error(f"roll图处理异常: {e}")
            yield event.plain_result(f"❌ roll图处理失败: {str(e)}")
        finally:
            # 完成处理并启动下一个请求
            self.usage_manager.finish_processing(request_id)
            next_request_id = await self.usage_manager.start_next_in_queue()
            if next_request_id:
                logger.info(
                    f"roll图完成，开始处理队列中的下一个请求: {next_request_id}"
                )

    def _create_roll_images_forward_message(
        self,
        generated_images: List[dict],
        success_count: int,
        failed_count: int,
        elapsed_time: float,
    ) -> List:
        """
        创建roll图结果的多条消息伪造转发

        Args:
            generated_images: 生成的图片列表，每个元素包含 {'image_data': bytes, 'index': int}
            success_count: 成功生成的图片数量
            failed_count: 失败的图片数量
            elapsed_time: 总用时（秒）

        Returns:
            List: 包含多条消息的伪造转发
        """
        try:
            # 验证输入数据
            if not generated_images or success_count <= 0:
                logger.error("roll图转发消息创建失败：没有有效的图片数据")
                raise ValueError("没有有效的图片数据")

            # 创建多个Node，每个包含不同的内容
            nodes_list = []

            # 第1条消息：统计信息
            stats_content = f"总计{success_count}张图，用时{elapsed_time:.1f}秒"
            if failed_count > 0:
                stats_content += f"（{failed_count}张失败）"

            stats_node = Comp.Node(
                uin=2854196310, name="Roll图统计", content=[Comp.Plain(stats_content)]
            )
            nodes_list.append(stats_node)

            # 第2条消息开始：每张图片
            for img_info in generated_images:
                try:
                    # 验证图片数据
                    if "image_data" not in img_info or not img_info["image_data"]:
                        logger.error(f"图片{img_info.get('index', '?')}数据无效，跳过")
                        continue

                    if len(img_info["image_data"]) < 1024:
                        logger.error(f"图片{img_info.get('index', '?')}数据过小，跳过")
                        continue

                    image_node = Comp.Node(
                        uin=2854196310 + img_info["index"],
                        name=f"图片{img_info['index']}",
                        content=[Comp.Image.fromBytes(img_info["image_data"])],
                    )
                    nodes_list.append(image_node)
                    logger.info(f"成功创建图片{img_info['index']}的转发节点")

                except Exception as e:
                    logger.error(
                        f"创建图片{img_info.get('index', '?')}转发节点失败: {e}"
                    )
                    continue

            # 检查是否有有效的图片节点
            if len(nodes_list) <= 1:  # 只有统计节点
                logger.error("没有有效的图片节点，无法创建转发")
                raise ValueError("没有有效的图片节点")

            # 使用Nodes包装所有节点，创建一条包含多个子消息的转发
            nodes = Comp.Nodes(nodes=nodes_list)
            self.log_roll(f"roll图转发消息创建完成，包含{len(nodes_list)}个节点", 4)
            return [nodes]

        except Exception as e:
            logger.error(f"创建roll图伪造转发消息失败: {e}")
            # 如果失败，返回普通文本格式
            result_text = (
                f"🎲 Roll图结果\n\n总计{success_count}张图，用时{elapsed_time:.1f}秒"
            )
            if failed_count > 0:
                result_text += f"（{failed_count}张失败）"
            return [Comp.Plain(result_text)]

    @filter.command("画图模型", alias={"NAI模型"})
    async def set_drawing_model(self, event):
        """Set the image model override for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        model_input = self._get_command_argument(event, "画图模型", "NAI模型")
        current = self._get_session_drawing_settings(event)["model"]
        if not model_input:
            model_lines = "\n".join(
                f"• {label}: {model_id}"
                for label, model_id in Constants.IMAGE_MODELS.items()
                if getattr(self, "enable_v5", True)
                or not model_id.startswith("nai-diffusion-5")
            )
            v5_status = "" if getattr(self, "enable_v5", True) else "\n⛔ V5 已关闭"
            yield event.plain_result(
                f"🤖 当前会话模型: {current}{v5_status}\n\n"
                f"用法: /画图模型 <名称或模型ID>\n{model_lines}"
            )
            return

        model_lookup = {
            **{
                label.lower(): model_id
                for label, model_id in Constants.IMAGE_MODELS.items()
            },
            **{
                model_id.lower(): model_id
                for model_id in Constants.IMAGE_MODELS.values()
            },
        }
        model_name = model_lookup.get(model_input.lower())
        if not model_name:
            yield event.plain_result("❌ 不支持的画图模型，发送 /画图模型 查看可选项")
            return
        if not getattr(self, "enable_v5", True) and model_name.startswith(
            "nai-diffusion-5"
        ):
            yield event.plain_result("❌ 管理面板已暂时关闭 V5，请选择 V4.5 模型")
            return

        self._set_session_drawing_setting(event, "model", model_name)
        yield event.plain_result(f"✅ 当前会话画图模型已设为: {model_name}")

    @filter.command("画图步数", alias={"NAI步数"})
    async def set_drawing_steps(self, event):
        """Report that sampling steps are controlled by administrators.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        current = self._get_session_drawing_settings(event)["steps"]
        yield event.plain_result(
            f"⛔ 画图步数由管理员统一设置，用户不可调整。\n当前步数: {current}"
        )

    @filter.command("画图权重", alias={"NAI权重"})
    async def set_drawing_scale(self, event):
        """Set prompt guidance scale for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        current = self._get_session_drawing_settings(event)["scale"]
        value_text, scale, error = self._parse_session_float_argument(
            event,
            "画图权重",
            "NAI权重",
            minimum=1.0,
            maximum=15.0,
        )
        if not value_text:
            yield event.plain_result(
                f"🎯 当前会话提示词权重: {current}\n用法: /画图权重 <1-15>"
            )
            return
        if error == "invalid":
            yield event.plain_result("❌ 权重必须是数值")
            return
        if error == "range":
            yield event.plain_result("❌ 权重必须在1-15之间")
            return

        self._set_session_drawing_setting(event, "scale", scale)
        yield event.plain_result(f"✅ 当前会话提示词权重已设为: {scale:g}")

    @filter.command("画图采样器", alias={"NAI采样器"})
    async def set_drawing_sampler(self, event):
        """Set the NovelAI sampler for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        sampler_input = self._get_command_argument(event, "画图采样器", "NAI采样器")
        current = self._get_session_drawing_settings(event)["sampler"]
        if not sampler_input:
            sampler_lines = "\n".join(
                f"• {alias}: {sampler}"
                for alias, sampler in Constants.IMAGE_SAMPLERS.items()
            )
            yield event.plain_result(
                f"🎛️ 当前会话采样器: {current}\n\n"
                f"用法: /画图采样器 <简称或采样器ID>\n{sampler_lines}"
            )
            return

        sampler_lookup = {
            **Constants.IMAGE_SAMPLERS,
            **{sampler: sampler for sampler in Constants.IMAGE_SAMPLERS.values()},
        }
        sampler = sampler_lookup.get(sampler_input.lower())
        if not sampler:
            yield event.plain_result("❌ 不支持的采样器，发送 /画图采样器 查看可选项")
            return

        self._set_session_drawing_setting(event, "sampler", sampler)
        yield event.plain_result(f"✅ 当前会话采样器已设为: {sampler}")

    @filter.command("重置画图参数", alias={"画图参数重置"})
    async def reset_drawing_settings(self, event):
        """Reset all current-session drawing overrides.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        had_overrides = self._reset_session_drawing_settings(event)
        if had_overrides:
            yield event.plain_result("✅ 当前会话的画图参数已恢复管理面板默认值")
        else:
            yield event.plain_result("ℹ️ 当前会话没有单独的画图参数")

    @filter.command("图生图强度")
    async def set_img2img_strength(self, event):
        """Set image-to-image strength for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        current = self._get_session_drawing_settings(event)["img2img_strength"]
        value_text, strength, error = self._parse_session_float_argument(
            event,
            "图生图强度",
            minimum=0.1,
            maximum=0.9,
        )
        if not value_text:
            yield event.plain_result(
                f"❌ 请指定强度值\n\n📖 用法: /图生图强度 <0.1-0.9>"
                f"\n💡 当前会话值: {current}"
            )
            return

        if error == "range":
            yield event.plain_result("❌ 强度值必须在0.1-0.9之间")
            return
        if error == "invalid":
            yield event.plain_result("❌ 请输入有效的数值")
            return

        self._set_session_drawing_setting(event, "img2img_strength", strength)
        yield event.plain_result(
            f"✅ 当前会话图生图强度已设为: {strength}"
            "\n\n💡 强度越高，生成图片与原图差异越大"
        )

    @filter.command("图生图噪声")
    async def set_img2img_noise(self, event):
        """Set image-to-image noise for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        current = self._get_session_drawing_settings(event)["img2img_noise"]
        value_text, noise, error = self._parse_session_float_argument(
            event,
            "图生图噪声",
            minimum=0.0,
            maximum=1.0,
        )
        if not value_text:
            yield event.plain_result(
                f"❌ 请指定噪声值\n\n📖 用法: /图生图噪声 <0.0-1.0>"
                f"\n💡 当前会话值: {current}"
            )
            return

        if error == "range":
            yield event.plain_result("❌ 噪声值必须在0.0-1.0之间")
            return
        if error == "invalid":
            yield event.plain_result("❌ 请输入有效的数值")
            return

        self._set_session_drawing_setting(event, "img2img_noise", noise)
        yield event.plain_result(
            f"✅ 当前会话图生图噪声已设为: {noise}"
            "\n\n💡 噪声越高，生成图片随机性越大"
        )

    @filter.command("缩放")
    async def set_prompt_guidance_rescale(self, event):
        """Set V4.5 CFG rescale for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        current = self._get_session_drawing_settings(event)["cfg_rescale"]
        value_text, rescale, error = self._parse_session_float_argument(
            event,
            "缩放",
            minimum=0.0,
            maximum=1.0,
        )
        if not value_text:
            yield event.plain_result(
                f"❌ 请指定缩放值\n\n📖 用法: /缩放 <0.0-1.0>"
                f"\n💡 当前会话值: {current}"
                "\n\n🔍 说明: 仅最终实际模型为 V4.5 时生效，"
                "缓解高引导值时的过度色彩饱和问题"
            )
            return

        if error == "range":
            yield event.plain_result("❌ 缩放值必须在0.0-1.0之间")
            return
        if error == "invalid":
            yield event.plain_result("❌ 请输入有效的数值")
            return

        self._set_session_drawing_setting(event, "cfg_rescale", rescale)
        yield event.plain_result(
            f"✅ 当前会话 V4.5 引导缩放已设为: {rescale}"
            "\n💡 仅直接使用 V4.5，或超额且未绑定个人密钥时生效"
        )

    @filter.command("分辨率")
    async def set_resolution(self, event):
        """Set image resolution for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        resolution_input = self._get_command_argument(event, "分辨率")
        current = self._get_session_drawing_settings(event)["resolution"]
        if not resolution_input:
            yield event.plain_result(
                f"❌ 请指定分辨率类型\n\n📖 用法: /分辨率 <类型或尺寸>"
                f"\n💡 当前会话分辨率: {current}"
                "\n\n🎨 支持: 长图/832x1216、宽图/1216x832、方图/1024x1024"
            )
            return

        resolution_lookup = {
            **Constants.IMAGE_RESOLUTIONS,
            **{
                resolution: resolution
                for resolution in Constants.IMAGE_RESOLUTIONS.values()
            },
        }
        new_resolution = resolution_lookup.get(resolution_input)
        if not new_resolution:
            yield event.plain_result("❌ 不支持的分辨率，发送 /分辨率 查看可选项")
            return

        self._set_session_drawing_setting(event, "resolution", new_resolution)
        yield event.plain_result(f"✅ 当前会话分辨率已设为: {new_resolution}")

    @filter.command("参数设置")
    async def show_img2img_params(self, event):
        """Show effective drawing parameters for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        settings = self._get_session_drawing_settings(event)
        session_id = self._get_session_identifier(event)
        override_count = len(self.session_drawing_settings.get(session_id, {}))
        result_text = f"⚙️ 当前会话画图参数（覆盖 {override_count} 项）\n\n"
        result_text += f"🤖 模型: {settings['model']}\n"
        result_text += f"📐 分辨率: {settings['resolution']}\n"
        result_text += f"⚙️ 步数: {settings['steps']}\n"
        result_text += f"🎯 权重: {settings['scale']}\n"
        result_text += f"🎛️ 采样器: {settings['sampler']}\n"
        result_text += (
            "\n🧩 V4.5 专用参数（含未绑定个人密钥时的超额兜底，仅最终 V4.5 生效）\n"
        )
        result_text += f"🎨 Prompt Guidance Rescale: {settings['cfg_rescale']}\n"
        result_text += f"🌊 Noise Schedule: {settings['v4_5_noise_schedule']}\n"
        result_text += (
            f"🎲 CFG Delay / Variety Boost: "
            f"{'开启' if settings['v4_5_variety_boost'] else '关闭'}\n"
        )
        result_text += (
            f"🧼 Decrisper: {'开启' if settings['v4_5_decrisper'] else '关闭'}\n"
        )
        result_text += (
            f"🌀 Auto SMEA: {'开启' if settings['v4_5_auto_smea'] else '关闭'} | "
            f"SMEA: {'开启' if settings['v4_5_smea'] else '关闭'} | "
            f"DYN: {'开启' if settings['v4_5_smea_dyn'] else '关闭'}\n"
        )
        result_text += f"🎭 画风: {settings['style_name'] if settings['style_enabled'] and settings['style_name'] else '关闭'}\n"
        result_text += "📝 提示词处理: 原样提交，不调用AI增强\n"
        result_text += f"🔁 Roll数量: {settings['roll_count']}\n\n"

        result_text += f"🎚️ 图生图强度: {settings['img2img_strength']}\n"
        result_text += "   • 范围: 0.1 - 0.9\n"
        result_text += "   • 说明: 控制生成图片与原图的相似度\n"
        result_text += "   • 效果: 值越高差异越大\n\n"

        result_text += f"🌊 图生图噪声: {settings['img2img_noise']}\n"
        result_text += "   • 范围: 0.0 - 1.0\n"
        result_text += "   • 说明: 控制生成过程的随机性\n"
        result_text += "   • 效果: 值越高随机性越大\n\n"

        result_text += "📝 修改命令:\n"
        result_text += "• /画图模型 /画图权重 /画图采样器\n"
        result_text += "• /分辨率 /缩放 /切换画风\n"
        result_text += "• /图生图强度 <值>\n"
        result_text += "• /图生图噪声 <值>\n"
        result_text += "• /重置画图参数 - 恢复管理面板默认值\n"
        result_text += "• 单次图生图临时参数: -s<强度> -n<噪声>"

        # 使用伪造转发
        forward_chain = self._create_fake_forward_message(
            "图生图参数设置", result_text, "参数管理器"
        )
        yield event.chain_result(forward_chain)

    @filter.command("开启审核")
    async def enable_image_audit(self, event):
        """开启图片审核功能"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "开启审核"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        if not self.baidu_audit_api_key or not self.baidu_audit_secret_key:
            yield event.plain_result(
                "❌ 百度审核API未配置\n请在后台管理面板中设置 baidu_audit_api_key 和 baidu_audit_secret_key"
            )
            return

        self.image_audit_enabled = True
        self.config["enable_image_audit"] = True
        self.config.save_config()

        # 创建审核失败图片保存目录
        Utils.create_directory_if_not_exists(self.audit_failed_save_path)

        if self.audit_whitelist_bots:
            yield event.plain_result("✅ 图片审核已开启")
        else:
            yield event.plain_result(
                "✅ 图片审核已开启\n"
                "⚠️ Bot 审核白名单为空，当前没有 Bot 生成的图片会进入审核"
            )

    @filter.command("关闭审核")
    async def disable_image_audit(self, event):
        """关闭图片审核功能"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "关闭审核"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        self.image_audit_enabled = False
        self.config["enable_image_audit"] = False
        self.config.save_config()

        yield event.plain_result("✅ 图片审核已关闭")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("审核白名单", alias={"NAI白名单"})
    async def audit_whitelist_status(self, event):
        """查看 Bot 审核白名单和群豁免白名单。"""
        if not self._is_user_admin(event):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        has_groups = bool(self.audit_whitelist_groups)
        has_bots = bool(self.audit_whitelist_bots)

        if not has_groups and not has_bots:
            yield event.plain_result(
                "📋 审核白名单为空\n\n"
                "💡 /添加审核白名单 [群号]   — 添加审核豁免群，不填使用当前群\n"
                "💡 /添加Bot白名单 [机器人ID] — 仅名单内Bot生成的图片会进入审核"
            )
            return

        whitelist_text = "📋 审核白名单\n"
        whitelist_text += "━" * 20 + "\n\n"

        if has_groups:
            whitelist_text += (
                f"🏠 群豁免白名单（{len(self.audit_whitelist_groups)} 个）：\n"
            )
            for i, gid in enumerate(self.audit_whitelist_groups, 1):
                whitelist_text += f"  {i}. {gid}\n"
            whitelist_text += "\n"

        if has_bots:
            whitelist_text += (
                f"🤖 Bot审核白名单（{len(self.audit_whitelist_bots)} 个）：\n"
            )
            for i, bid in enumerate(self.audit_whitelist_bots, 1):
                whitelist_text += f"  {i}. {bid}\n"
            whitelist_text += "\n"

        whitelist_text += "💡 仅 Bot 白名单内账号生成的图片会进入审核；群白名单内的图片仍会跳过审核"
        yield event.plain_result(whitelist_text)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("添加审核白名单", alias={"添加NAI群白名单"})
    async def add_audit_whitelist(self, event, group_id: str = ""):
        """添加群到审核豁免白名单"""
        if not self._is_user_admin(event):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        group_id = self._normalize_whitelist_id(group_id) or self._current_group_id(
            event
        )

        if not group_id:
            yield event.plain_result("❌ 请提供群号，或在群聊中发送：/添加审核白名单")
            return

        yield event.plain_result(
            self._update_audit_whitelist(
                attribute_name="audit_whitelist_groups",
                config_key="audit_whitelist_groups",
                value=group_id,
                noun="群",
                subject="该群",
                add=True,
            )
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("移除审核白名单", alias={"移除NAI群白名单"})
    async def remove_audit_whitelist(self, event, group_id: str = ""):
        """从审核白名单中移除群"""
        if not self._is_user_admin(event):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        group_id = self._normalize_whitelist_id(group_id) or self._current_group_id(
            event
        )

        if not group_id:
            yield event.plain_result("❌ 请提供群号，或在群聊中发送：/移除审核白名单")
            return

        yield event.plain_result(
            self._update_audit_whitelist(
                attribute_name="audit_whitelist_groups",
                config_key="audit_whitelist_groups",
                value=group_id,
                noun="群",
                subject="该群",
                add=False,
            )
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("添加Bot白名单", alias={"添加NAIBot白名单", "添加NAI机器人白名单"})
    async def add_bot_audit_whitelist(self, event, bot_id: str = ""):
        """添加机器人ID到审核白名单，仅该机器人生成的图片会进入审核"""
        if not self._is_user_admin(event):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        bot_id = self._normalize_whitelist_id(bot_id) or self._current_bot_id(event)

        if not bot_id:
            # 未提供参数时，自动填入当前机器人ID供参考
            current_self_id = self._current_bot_id(event)
            tip = f"\n💡 当前机器人ID: {current_self_id}" if current_self_id else ""
            yield event.plain_result(
                f"❌ 请提供机器人ID\n用法: /添加Bot白名单 <机器人ID>{tip}"
            )
            return

        yield event.plain_result(
            self._update_audit_whitelist(
                attribute_name="audit_whitelist_bots",
                config_key="audit_whitelist_bots",
                value=bot_id,
                noun="机器人",
                subject="该机器人",
                add=True,
                audit_on_add=True,
            )
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("移除Bot白名单", alias={"移除NAIBot白名单", "移除NAI机器人白名单"})
    async def remove_bot_audit_whitelist(self, event, bot_id: str = ""):
        """从审核白名单中移除机器人ID"""
        if not self._is_user_admin(event):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        bot_id = self._normalize_whitelist_id(bot_id) or self._current_bot_id(event)

        if not bot_id:
            yield event.plain_result("❌ 请提供机器人ID，或直接发送：/移除Bot白名单")
            return

        yield event.plain_result(
            self._update_audit_whitelist(
                attribute_name="audit_whitelist_bots",
                config_key="audit_whitelist_bots",
                value=bot_id,
                noun="机器人",
                subject="该机器人",
                add=False,
                audit_on_add=True,
            )
        )

    @filter.command("添加画风")
    async def add_style_preset(self, event):
        """添加新的预设画风"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "添加画风"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()
            # 移除命令部分
            if message_text.startswith("/添加画风"):
                args_text = message_text[5:].strip()
            elif message_text.startswith("添加画风"):
                args_text = message_text[4:].strip()
            else:
                args_text = message_text

            # Split name first, then use --uc to separate positive and negative prompts.
            parts = args_text.split(None, 1)

            if len(parts) < 2:
                yield event.plain_result(
                    "❌ 用法: /添加画风 <名称> <正面提示词> --uc <负面提示词>\n"
                    "例如: /添加画风 科幻风 sci-fi, futuristic --uc pastoral, historical"
                )
                return

            style_name = parts[0]
            prompt_parts = re.split(r"(?<!\S)--uc(?:\s+|=)", parts[1], maxsplit=1)
            positive_prompt = prompt_parts[0].strip(" ,")
            negative_prompt = (
                prompt_parts[1].strip(" ,") if len(prompt_parts) > 1 else ""
            )

            if not style_name or not positive_prompt or not negative_prompt:
                yield event.plain_result("❌ 画风名、正面提示词和负面提示词均不能为空")
                return

            # 检查画风名是否已存在
            if style_name in self.style_presets:
                result_message = f"⚠️ 画风 '{style_name}' 已存在，画风将被覆盖"
            else:
                result_message = f"✅ 画风 '{style_name}' 添加成功！"

            current = normalize_style_entry(self.style_presets.get(style_name))
            self.style_presets[style_name] = {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "preview_image": current["preview_image"],
            }
            # 保存画风到配置文件
            self.save_style_presets()

            yield event.plain_result(
                result_message + "\n💡 请在 AstrBot 画风库 WebUI 中上传示例图。"
            )

        except Exception as e:
            logger.error(f"添加画风失败: {e}")
            yield event.plain_result(f"❌ 添加画风失败: {str(e)}")

    @filter.command("撤回时间")
    async def set_auto_recall(self, event):
        """设置图片自动撤回时间"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "撤回时间"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()

            # 移除命令部分
            if message_text.startswith("/撤回时间"):
                args_text = message_text[4:].strip()
            elif message_text.startswith("撤回时间"):
                args_text = message_text[4:].strip()
            else:
                args_text = message_text

            # 如果没有参数，显示当前状态
            if not args_text:
                if self.auto_recall_enabled:
                    yield event.plain_result(
                        f"📋 当前撤回设置:\n✅ 自动撤回: 已启用\n⏰ 撤回延迟: {self.auto_recall_delay} 秒\n\n💡 用法: /撤回时间 <秒数>"
                    )
                else:
                    yield event.plain_result(
                        f"📋 当前撤回设置:\n❌ 自动撤回: 已禁用\n\n💡 用法: /撤回时间 <秒数> 或先使用 /开启撤回"
                    )
                return

            # 处理数字参数
            try:
                delay_seconds = int(args_text)

                if delay_seconds < 1:
                    yield event.plain_result("❌ 撤回延迟时间必须大于0秒")
                    return
                elif delay_seconds > 120:
                    yield event.plain_result("❌ 撤回延迟时间不能超过120秒")
                    return

                # 更新配置（设置时间会自动启用撤回功能）
                self.auto_recall_enabled = True
                self.auto_recall_delay = delay_seconds
                self.config["enable_auto_recall"] = True
                self.config["auto_recall_delay"] = delay_seconds
                self.config.save_config()

                yield event.plain_result(f"✅ 图片自动撤回已设置为 {delay_seconds} 秒")

            except ValueError:
                yield event.plain_result(
                    "❌ 请输入有效的数字\n💡 用法: /撤回时间 30 (设置30秒后撤回)"
                )

        except Exception as e:
            logger.error(f"设置撤回时间失败: {e}")
            yield event.plain_result(f"❌ 设置失败: {str(e)}")

    @filter.command("开启撤回")
    async def enable_auto_recall(self, event):
        """开启图片自动撤回功能"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "开启撤回"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            if self.auto_recall_enabled:
                yield event.plain_result(
                    f"✅ 自动撤回功能已经是开启状态\n⏰ 当前撤回延迟: {self.auto_recall_delay} 秒"
                )
            else:
                self.auto_recall_enabled = True
                self.config["enable_auto_recall"] = True
                self.config.save_config()
                yield event.plain_result(
                    f"✅ 图片自动撤回功能已开启\n⏰ 撤回延迟: {self.auto_recall_delay} 秒\n💡 使用 /撤回时间 <秒数> 可调整延迟时间"
                )

        except Exception as e:
            logger.error(f"开启撤回功能失败: {e}")
            yield event.plain_result(f"❌ 开启失败: {str(e)}")

    @filter.command("关闭撤回")
    async def disable_auto_recall(self, event):
        """关闭图片自动撤回功能"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "关闭撤回"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            if not self.auto_recall_enabled:
                yield event.plain_result("❌ 自动撤回功能已经是关闭状态")
            else:
                self.auto_recall_enabled = False
                self.config["enable_auto_recall"] = False
                self.config.save_config()
                yield event.plain_result("❌ 图片自动撤回功能已关闭")

        except Exception as e:
            logger.error(f"关闭撤回功能失败: {e}")
            yield event.plain_result(f"❌ 关闭失败: {str(e)}")

    @filter.command("搜索")
    async def search_character(self, event):
        """搜索角色标签"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        try:
            # 从消息中提取搜索关键词
            message_text = event.message_str.strip()

            # 移除命令部分
            if message_text.startswith("/搜索"):
                keyword = message_text[3:].strip()
            elif message_text.startswith("搜索"):
                keyword = message_text[2:].strip()
            else:
                keyword = message_text

            if not keyword:
                yield event.plain_result(
                    "❌ 请输入要搜索的角色名称\n💡 用法: /搜索 初音未来"
                )
                return

            # 搜索角色
            results = self._search_characters(keyword)

            if not results:
                yield event.plain_result(f"❌ 未找到包含 '{keyword}' 的角色")
                return

            # 限制显示结果数量
            max_results = 20
            if len(results) > max_results:
                results = results[:max_results]
                truncated_msg = f"\n\n⚠️ 结果过多，仅显示前{max_results}个"
            else:
                truncated_msg = ""

            # 格式化结果
            result_lines = []
            for i, char_info in enumerate(results, 1):
                chinese_name = char_info["chinese_name"] or char_info["tag"]
                series = char_info["series"]
                tag = char_info["tag"]
                usage_count = char_info["usage_count"]

                if series:
                    line = f"{i}. {chinese_name}（{series}）\n   标签: {tag} | 使用: {usage_count}"
                else:
                    line = f"{i}. {chinese_name}\n   标签: {tag} | 使用: {usage_count}"

                result_lines.append(line)

            result_text = (
                f"🔍 搜索结果 '{keyword}':\n\n"
                + "\n\n".join(result_lines)
                + truncated_msg
            )

            # 使用伪造转发
            forward_chain = self._create_fake_forward_message(
                "角色搜索结果", result_text, "角色数据库"
            )
            yield event.chain_result(forward_chain)

        except Exception as e:
            logger.error(f"搜索角色失败: {e}")
            yield event.plain_result(f"❌ 搜索失败: {str(e)}")

    def _search_characters(self, keyword):
        """搜索角色"""
        # 确保角色数据库已加载
        self._ensure_character_database_loaded()
        # 安全检查：确保角色数据库已初始化
        if not hasattr(self, "character_database") or not self.character_database:
            return []

        results = []
        keyword_lower = keyword.lower()

        # 搜索匹配的角色
        for tag, char_info in self.character_database.items():
            # 检查标签名
            if keyword_lower in tag.lower():
                results.append(char_info)
                continue

            # 检查中文名
            if char_info["chinese_name"] and keyword in char_info["chinese_name"]:
                results.append(char_info)
                continue

            # 检查作品名
            if char_info["series"] and keyword in char_info["series"]:
                results.append(char_info)
                continue

            # 检查别名
            for alias in char_info["aliases"]:
                if keyword_lower in alias.lower() or keyword in alias:
                    results.append(char_info)
                    break

        # 按使用次数排序
        results.sort(key=lambda x: x["usage_count"], reverse=True)
        return results

    @filter.command("添加角色")
    async def add_character_from_database(self, event):
        """从角色数据库添加角色预设"""
        # 添加调试日志
        logger.info(f"添加角色命令被触发: {event.message_str}")

        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "添加角色"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()

            # 移除命令部分
            if message_text.startswith("/添加角色"):
                args_text = message_text[5:].strip()
            elif message_text.startswith("添加角色"):
                args_text = message_text[4:].strip()
            else:
                args_text = message_text

            if not args_text:
                yield event.plain_result(
                    "❌ 请输入角色名称和预设名\n💡 用法: /添加角色 初音未来 初音"
                )
                return

            # 解析参数：角色名 预设名
            parts = args_text.split()
            if len(parts) < 2:
                yield event.plain_result(
                    "❌ 请输入角色名称和预设名\n💡 用法: /添加角色 初音未来 初音"
                )
                return

            character_name = parts[0]
            preset_name = parts[1]

            # 搜索角色
            character_tag = self._find_character_tag(character_name)
            if not character_tag:
                yield event.plain_result(
                    f"❌ 未找到角色 '{character_name}'\n💡 请使用 /搜索 {character_name} 查找正确的角色名"
                )
                return

            # 确保角色数据库已加载
            self._ensure_character_database_loaded()
            # 安全检查：确保角色数据库已初始化
            if (
                not hasattr(self, "character_database")
                or character_tag not in self.character_database
            ):
                yield event.plain_result(f"❌ 角色数据库未初始化或角色不存在")
                return

            character_info = self.character_database[character_tag]

            # 添加到人物预设
            if self._add_preset_to_category("人物", preset_name, character_tag):
                chinese_name = character_info["chinese_name"] or character_tag
                series = character_info["series"]

                success_msg = f"✅ 已添加角色预设:\n"
                success_msg += f"• 预设名: {preset_name}\n"
                success_msg += f"• 角色: {chinese_name}"
                if series:
                    success_msg += f"（{series}）"
                success_msg += f"\n• 标签: {character_tag}"

                yield event.plain_result(success_msg)
            else:
                yield event.plain_result(f"❌ 预设名 '{preset_name}' 已存在")

        except Exception as e:
            logger.error(f"添加角色预设失败: {e}")
            yield event.plain_result(f"❌ 添加失败: {str(e)}")

    def _find_character_tag(self, name):
        """根据名称查找角色标签，处理重名情况"""
        # 确保角色数据库已加载
        self._ensure_character_database_loaded()
        # 安全检查：确保角色数据库已初始化
        if not hasattr(self, "character_database") or not hasattr(
            self, "character_aliases"
        ):
            return None

        # 直接匹配标签名（最精确）
        if name in self.character_database:
            return name

        # 通过别名查找（如果别名是唯一的）
        if name in self.character_aliases:
            return self.character_aliases[name]

        # 查找所有匹配的角色
        matches = []
        name_lower = name.lower()

        for tag, char_info in self.character_database.items():
            # 检查标签名
            if name_lower == tag.lower():
                matches.append((tag, char_info, "tag"))
                continue

            # 检查中文名
            if char_info["chinese_name"] and name == char_info["chinese_name"]:
                matches.append((tag, char_info, "chinese_name"))
                continue

            # 检查别名
            for alias in char_info["aliases"]:
                if name_lower == alias.lower() or name == alias:
                    matches.append((tag, char_info, "alias"))
                    break

        if not matches:
            return None

        if len(matches) == 1:
            return matches[0][0]

        # 处理重名情况：按优先级和使用频率排序
        def sort_key(match):
            _, char_info, match_type = match
            # 优先级：tag > chinese_name > alias
            type_priority = {"tag": 3, "chinese_name": 2, "alias": 1}
            return (type_priority[match_type], char_info["usage_count"])

        matches.sort(key=sort_key, reverse=True)

        # 返回最高优先级和使用频率的角色
        best_match = matches[0]
        logger.info(
            f"重名角色处理: '{name}' 匹配到 {len(matches)} 个角色，选择: {best_match[0]} (使用次数: {best_match[1]['usage_count']})"
        )

        return best_match[0]

    @filter.command("角色详情")
    async def show_character_details(self, event):
        """查看角色详情，包括重名角色"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        try:
            # 从消息中提取角色名
            message_text = event.message_str.strip()

            # 移除命令部分
            if message_text.startswith("/角色详情"):
                character_name = message_text[5:].strip()
            elif message_text.startswith("角色详情"):
                character_name = message_text[4:].strip()
            else:
                character_name = message_text

            if not character_name:
                yield event.plain_result("❌ 请输入角色名称\n💡 用法: /角色详情 椿")
                return

            # 查找所有匹配的角色
            matches = self._find_all_character_matches(character_name)

            if not matches:
                yield event.plain_result(f"❌ 未找到角色 '{character_name}'")
                return

            # 格式化结果
            if len(matches) == 1:
                char_info = matches[0][1]
                result_text = f"📋 角色详情: {character_name}\n\n"
                result_text += f"• 中文名: {char_info['chinese_name'] or '无'}\n"
                result_text += f"• 作品: {char_info['series'] or '无'}\n"
                result_text += f"• 标签: {char_info['tag']}\n"
                result_text += f"• 使用次数: {char_info['usage_count']}\n"
                if char_info["aliases"]:
                    result_text += f"• 别名: {', '.join(char_info['aliases'])}"
            else:
                result_text = f"⚠️ 发现 {len(matches)} 个同名角色: {character_name}\n\n"
                for i, (tag, char_info, _) in enumerate(matches, 1):
                    result_text += f"{i}. {char_info['chinese_name'] or tag}\n"
                    result_text += f"   作品: {char_info['series'] or '无'}\n"
                    result_text += f"   标签: {tag}\n"
                    result_text += f"   使用: {char_info['usage_count']}\n"
                    if i < len(matches):
                        result_text += "\n"

                result_text += f"\n💡 画图时会自动选择使用次数最高的角色"

            yield event.plain_result(result_text)

        except Exception as e:
            logger.error(f"查看角色详情失败: {e}")
            yield event.plain_result(f"❌ 查看失败: {str(e)}")

    def _find_all_character_matches(self, name):
        """查找所有匹配的角色（用于显示重名情况）"""
        # 确保角色数据库已加载
        self._ensure_character_database_loaded()
        # 安全检查：确保角色数据库已初始化
        if not hasattr(self, "character_database") or not self.character_database:
            return []

        matches = []
        name_lower = name.lower()

        for tag, char_info in self.character_database.items():
            # 检查标签名
            if name_lower == tag.lower():
                matches.append((tag, char_info, "tag"))
                continue

            # 检查中文名
            if char_info["chinese_name"] and name == char_info["chinese_name"]:
                matches.append((tag, char_info, "chinese_name"))
                continue

            # 检查别名
            for alias in char_info["aliases"]:
                if name_lower == alias.lower() or name == alias:
                    matches.append((tag, char_info, "alias"))
                    break

        # 按优先级和使用频率排序
        def sort_key(match):
            _, char_info, match_type = match
            type_priority = {"tag": 3, "chinese_name": 2, "alias": 1}
            return (type_priority[match_type], char_info["usage_count"])

        matches.sort(key=sort_key, reverse=True)
        return matches

    @filter.command("删除画风")
    async def delete_style_preset(self, event):
        """删除指定的预设画风"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "删除画风"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 从消息中提取参数
            message_text = event.message_str.strip()
            # 移除命令部分
            if message_text.startswith("/删除画风"):
                style_name = message_text[5:].strip()
            elif message_text.startswith("删除画风"):
                style_name = message_text[4:].strip()
            else:
                style_name = message_text

            if not style_name:
                yield event.plain_result("❌ 用法: /删除画风 <画风名>")
                return

            if style_name not in self.style_presets:
                yield event.plain_result(f"❌ 画风 '{style_name}' 不存在")
                return

            self.preset_webui.delete_style({"name": style_name})

            yield event.plain_result(f"✅ 画风 '{style_name}' 删除成功！")

        except Exception as e:
            logger.error(f"删除画风失败: {e}")
            yield event.plain_result(f"❌ 删除画风失败: {str(e)}")

    @filter.command("画风列表")
    async def list_style_presets(self, event):
        """以画风名和示例图卡片展示可用画风，可附带名称关键词筛选。

        Args:
            event: AstrBot message event.
        """
        # 检查白名单
        if not self._check_whitelist(event):
            return

        if not self.style_presets:
            yield event.plain_result(
                "🎨 当前没有任何画风\n\n💡 使用 /添加画风 <名称> <描述> 来添加画风"
            )
            return

        drawing_settings = self._get_session_drawing_settings(event)
        query = self._get_command_argument(event, "画风列表").strip().casefold()
        styles = list(self.style_presets.items())
        if query:
            styles = [
                (name, entry) for name, entry in styles if query in str(name).casefold()
            ]

        title = "画风列表" if not query else f"画风搜索：{query}"
        summary = f"{len(styles)} 个画风 · 仅显示名称与示例图"
        try:
            rendered = await self._render_style_gallery(
                styles,
                title=title,
                summary=summary,
                drawing_settings=drawing_settings,
            )
        except Exception as exc:
            # Never leave the command without a response when a local browser
            # is unavailable or takes too long to start.
            logger.error("画风列表发送失败: %s", exc)
            names = "、".join(str(name) for name, _ in styles)
            if names:
                yield event.plain_result(
                    f"🎨 画风列表\n\n{names}\n\n🖼️ 示例图渲染失败：{exc}"
                )
            else:
                yield event.plain_result("❌ 没有找到匹配的画风")
            return

        if rendered is not None:
            yield event.chain_result([rendered])
        else:
            names = "、".join(str(name) for name, _ in styles)
            if names:
                yield event.plain_result(
                    f"🎨 画风列表\n\n{names}\n\n🖼️ 示例图暂时无法渲染，请稍后重试"
                )
            else:
                yield event.plain_result("❌ 没有找到匹配的画风")

    async def _render_style_gallery(
        self,
        styles,
        *,
        title: str,
        summary: str,
        drawing_settings: Optional[dict] = None,
        single: bool = False,
    ):
        """Render a style gallery HTML template into one chat image.

        Only names and managed preview images are passed to the template. Prompt
        fields deliberately never enter this data structure, so the chat card
        cannot accidentally expose style prompts.
        """
        try:
            current_name = ""
            if drawing_settings and drawing_settings.get("style_enabled"):
                current_name = str(drawing_settings.get("style_name") or "")

            items = []
            for name, raw_entry in styles:
                entry = normalize_style_entry(raw_entry)
                preview_data = ""
                preview_image = entry.get("preview_image", "")
                if preview_image:
                    image_path = self.preset_webui._style_image_path(preview_image)
                    if image_path and image_path.is_file():
                        try:
                            preview_data = "data:image/webp;base64," + base64.b64encode(
                                image_path.read_bytes()
                            ).decode("ascii")
                        except OSError:
                            logger.warning("读取画风示例图失败: %s", image_path)
                items.append(
                    {
                        "name": str(name),
                        "preview_data": preview_data,
                        "is_current": str(name) == current_name,
                    }
                )

            rendered_path = await render_style_gallery_image(
                STYLE_GALLERY_TEMPLATE_PATH,
                {
                    "title": title,
                    "summary": summary,
                    "single": single,
                    "items": items,
                },
            )
            return Comp.Image.fromFileSystem(str(rendered_path))
        except Exception as exc:
            logger.exception("渲染画风列表图片失败: %s", exc)
            detail = str(exc).splitlines()[0].strip() or type(exc).__name__
            raise RuntimeError(f"本地 Playwright 渲染失败：{detail}") from exc

    async def _render_drawing_help(self, event):
        """Render effective drawing state into a chat-friendly help image."""
        try:
            drawing_settings = self._get_session_drawing_settings(event)
            user_id = str(event.get_sender_id())
            has_global_token = bool(str(getattr(self, "api_token", "") or "").strip())
            has_personal_key = bool(
                self._get_bound_user_api_key(
                    user_id, self._get_event_bot_id(event)
                )
            )
            has_group_key = bool(self._get_group_authorized_api_key(event))
            credential_status = "已配置 Token" if has_global_token else "未配置 Token"
            if has_personal_key:
                credential_status += " · 已绑定个人密钥"
            elif has_group_key:
                credential_status += " · 当前群已授权"

            rendered_path = await render_drawing_help_image(
                DRAWING_HELP_TEMPLATE_PATH,
                {
                    "model": drawing_settings.get("model") or "未配置",
                    "resolution": drawing_settings.get("resolution") or "未配置",
                    "steps": drawing_settings.get("steps", "未配置"),
                    "scale": drawing_settings.get("scale", "未配置"),
                    "sampler": drawing_settings.get("sampler") or "未配置",
                    "credential_status": credential_status,
                    "preset_count": len(getattr(self, "presets", {})),
                    "style_count": len(getattr(self, "style_presets", {})),
                },
            )
            return Comp.Image.fromFileSystem(str(rendered_path))
        except Exception as exc:
            logger.exception("渲染画图帮助图片失败: %s", exc)
            detail = str(exc).splitlines()[0].strip() or type(exc).__name__
            raise RuntimeError(f"本地 Playwright 渲染失败：{detail}") from exc

    @filter.command("切换画风")
    async def switch_style_preset(self, event):
        """Switch the style preset for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        try:
            style_name = self._get_command_argument(event, "切换画风")
            drawing_settings = self._get_session_drawing_settings(event)

            if not style_name:
                # 显示可用画风列表
                if not self.style_presets:
                    yield event.plain_result(
                        "❌ 当前没有任何画风\n\n💡 使用 /添加画风 <名称> <描述> 来添加画风"
                    )
                    return

                style_list = "❌ 用法: /切换画风 <画风名>\n\n🎨 可用画风:\n"
                for name in self.style_presets.keys():
                    current = (
                        " (当前)"
                        if (
                            drawing_settings["style_enabled"]
                            and drawing_settings["style_name"] == name
                        )
                        else ""
                    )
                    style_list += f"  • {name}{current}\n"
                yield event.plain_result(style_list)
                return

            if style_name not in self.style_presets:
                yield event.plain_result(f"❌ 画风 '{style_name}' 不存在")
                return

            self._set_session_drawing_setting(event, "style_name", style_name)
            self._set_session_drawing_setting(event, "style_enabled", True)
            yield event.plain_result(f"✅ 当前会话已切换至画风 '{style_name}'")

        except Exception as e:
            logger.error(f"切换画风失败: {e}")
            yield event.plain_result(f"❌ 切换画风失败: {str(e)}")

    @filter.command("启用画风")
    async def enable_style(self, event):
        """Enable the selected style for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        try:
            self._set_session_drawing_setting(event, "style_enabled", True)
            drawing_settings = self._get_session_drawing_settings(event)
            if drawing_settings["style_name"]:
                yield event.plain_result(
                    f"✅ 当前会话画风已启用\n当前画风: {drawing_settings['style_name']}"
                )
            else:
                yield event.plain_result(
                    "✅ 画风功能已启用\n\n💡 使用 /切换画风 <名称> 来选择画风"
                )

        except Exception as e:
            logger.error(f"启用画风失败: {e}")
            yield event.plain_result(f"❌ 启用画风失败: {str(e)}")

    @filter.command("关闭画风")
    async def disable_style(self, event):
        """Disable the style preset for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        try:
            self._set_session_drawing_setting(event, "style_enabled", False)
            yield event.plain_result(
                "✅ 当前会话画风已关闭\n\n💡 在提示词中写入画风名仍可单次使用"
            )

        except Exception as e:
            logger.error(f"关闭画风失败: {e}")
            yield event.plain_result(f"❌ 关闭画风失败: {str(e)}")

    @filter.command("图生图")
    async def image_to_image(self, event):
        """进行图生图（需要先发送图片）"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        async for result in self._handle_image_to_image(event):
            yield result

    @filter.command("画图配置")
    async def show_config(self, event):
        """Show effective drawing settings for the current session.

        Args:
            event: AstrBot message event.
        """
        if not self._check_whitelist(event):
            return

        drawing_settings = self._get_session_drawing_settings(event)
        session_id = self._get_session_identifier(event)
        override_count = len(self.session_drawing_settings.get(session_id, {}))
        model_info = drawing_settings["model"] or "未配置"
        model_version = ""
        if drawing_settings["model"]:
            model_version = (
                f" ({self.get_model_version_label(drawing_settings['model'])})"
            )

        style_status = "未启用"
        if drawing_settings["style_enabled"]:
            style_status = (
                f"已启用 - {drawing_settings['style_name']}"
                if drawing_settings["style_name"]
                else "已启用但未选择画风"
            )

        # 图片审核功能状态
        audit_status = "未配置"
        if self.baidu_audit_api_key and self.baidu_audit_secret_key:
            audit_status = "已配置"
            if self.image_audit_enabled:
                # 严格程度描述
                strictness = {1: "严格", 2: "中等", 3: "宽松"}.get(
                    self.audit_conclusion_type, "未知"
                )

                # 策略信息
                strategy_info = (
                    f"策略ID: {self.audit_strategy_id}"
                    if self.audit_strategy_id
                    else "默认策略"
                )

                # 白名单信息
                whitelist_info = (
                    f"Bot审核白名单: {len(self.audit_whitelist_bots)}个"
                )
                if not self.audit_whitelist_bots:
                    whitelist_info += "（当前无Bot会进入审核）"
                if self.audit_whitelist_groups:
                    whitelist_info += (
                        f", 群豁免: {len(self.audit_whitelist_groups)}个"
                    )

                audit_status += f" | 图片审核已启用({strictness}模式, {strategy_info}, {whitelist_info})"
            else:
                audit_status += " | 图片审核未启用"

        config_text = (
            f"⚙️ 当前会话画图配置\n\n"
            f"🤖 模型: {model_info}{model_version}\n"
            f"📐 分辨率: {drawing_settings['resolution']}\n"
            f"⚙️ 参数: {drawing_settings['steps']}步 | "
            f"权重{drawing_settings['scale']} | {drawing_settings['sampler']}\n"
            "🧩 V4.5 专用（含无个人密钥时的超额兜底）: "
            f"Rescale {drawing_settings['cfg_rescale']} | "
            f"{drawing_settings['v4_5_noise_schedule']} | "
            f"CFG Delay {'开' if drawing_settings['v4_5_variety_boost'] else '关'} | "
            f"Decrisper {'开' if drawing_settings['v4_5_decrisper'] else '关'} | "
            f"Auto SMEA {'开' if drawing_settings['v4_5_auto_smea'] else '关'} | "
            f"SMEA {'开' if drawing_settings['v4_5_smea'] else '关'} | "
            f"DYN {'开' if drawing_settings['v4_5_smea_dyn'] else '关'}\n"
            f"🖼️ 图生图: 强度{drawing_settings['img2img_strength']} | "
            f"噪声{drawing_settings['img2img_noise']}\n"
            f"🎲 Roll图: {drawing_settings['roll_count']}张\n"
            f"🎨 画风: {style_status}\n"
            "📝 提示词处理: 原样提交，不调用AI增强\n"
            "🚫 负面提示词: 随画风自动应用，可用 --uc 单次覆盖\n"
            "📝 提示词处理: 原样提交，不调用AI增强\n"
            f"🧩 会话覆盖项: {override_count} 个\n"
            f"🔍 图片审核: {audit_status}\n"
            f"💾 自动保存: {'启用' if self.enable_auto_save else '未启用'}\n"
            f"📊 显示参数: {'启用' if self.show_params else '关闭'}\n"
            f"📋 已加载预设: {len(self.presets)} 个\n"
            f"🖌️ 已加载画风: {len(self.style_presets)} 个"
        )

        if self.whitelist_groups:
            config_text += f"\n\n🔒 群组限制: 仅限 {len(self.whitelist_groups)} 个群组"
        # 使用伪造转发
        forward_chain = self._create_fake_forward_message(
            "当前会话画图配置", config_text, "配置管理"
        )
        yield event.chain_result(forward_chain)

    @filter.command("衣服分类")
    async def show_clothing_categories(self, event):
        """显示衣服分类信息"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "衣服分类"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 检查是否指定了具体分类
            message_text = event.message_str.strip()
            if message_text.startswith("/衣服分类"):
                args_text = message_text[5:].strip()
            elif message_text.startswith("衣服分类"):
                args_text = message_text[4:].strip()
            else:
                args_text = ""

            if args_text:
                # 显示指定分类的详细信息
                category_info = self.clothing_manager.format_clothing_list(
                    args_text, 10
                )
                forward_chain = self._create_fake_forward_message(
                    f"{args_text}分类详情", category_info, "衣服管理器"
                )
                yield event.chain_result(forward_chain)
            else:
                # 显示所有分类概览
                categories = self.clothing_manager.get_categories()
                category_info = []
                category_info.append("🎀 衣服分类列表")
                category_info.append("")

                for category in categories:
                    count = self.clothing_manager.get_category_count(category)
                    range_info = self.clothing_manager.get_category_range(category)
                    category_info.append(f"{category}: {count}套 ({range_info})")

                category_info.append("")
                category_info.append("💡 使用方法:")
                category_info.append("• /衣服分类 职业装 - 查看职业装详情")
                category_info.append("• /画图 美少女 随机衣服 - 随机选择衣服")
                category_info.append("• /画图 美少女 001 - 使用指定编号衣服")

                result_text = "\n".join(category_info)

                forward_chain = self._create_fake_forward_message(
                    "衣服分类信息", result_text, "衣服管理器"
                )
                yield event.chain_result(forward_chain)

        except Exception as e:
            logger.error(f"显示衣服分类失败: {e}")
            yield event.plain_result(f"❌ 显示衣服分类失败: {str(e)}")

    @filter.command("衣服统计")
    async def show_clothing_statistics(self, event):
        """显示衣服数据库统计信息"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "衣服统计"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            stats_text = self.clothing_manager.get_statistics()

            forward_chain = self._create_fake_forward_message(
                "衣服数据库统计", stats_text, "衣服管理器"
            )
            yield event.chain_result(forward_chain)

        except Exception as e:
            logger.error(f"显示衣服统计失败: {e}")
            yield event.plain_result(f"❌ 显示衣服统计失败: {str(e)}")

    @filter.command("衣服搜索")
    async def search_clothing(self, event):
        """搜索衣服标签"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "衣服搜索"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 提取搜索关键词
            message_text = event.message_str.strip()
            if message_text.startswith("/衣服搜索"):
                keyword = message_text[5:].strip()
            elif message_text.startswith("衣服搜索"):
                keyword = message_text[4:].strip()
            else:
                keyword = message_text

            if not keyword:
                yield event.plain_result("❌ 请提供搜索关键词\n用法: /衣服搜索 关键词")
                return

            results = self.clothing_manager.search_clothing(keyword)

            if not results:
                yield event.plain_result(f"🔍 未找到包含 '{keyword}' 的衣服")
                return

            # 限制显示结果数量
            max_results = 10
            search_info = []
            search_info.append(f"🔍 搜索结果: '{keyword}' (共{len(results)}个)")
            search_info.append("")

            for number, category, name, tags in results[:max_results]:
                search_info.append(f"{number}: {category} - {name}")
                # 截断过长的标签
                short_tags = tags[:60] + "..." if len(tags) > 60 else tags
                search_info.append(f"     {short_tags}")
                search_info.append("")

            if len(results) > max_results:
                search_info.append(f"... 还有 {len(results) - max_results} 个结果")

            result_text = "\n".join(search_info)

            forward_chain = self._create_fake_forward_message(
                f"衣服搜索: {keyword}", result_text, "衣服管理器"
            )
            yield event.chain_result(forward_chain)

        except Exception as e:
            logger.error(f"搜索衣服失败: {e}")
            yield event.plain_result(f"❌ 搜索衣服失败: {str(e)}")

    @filter.command("衣服详情")
    async def clothing_detail(self, event):
        """查看指定编号的衣服详情"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "衣服详情"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            # 提取编号
            message_text = event.message_str.strip()
            if message_text.startswith("/衣服详情"):
                number = message_text[5:].strip()
            elif message_text.startswith("衣服详情"):
                number = message_text[4:].strip()
            else:
                number = message_text

            if not number:
                yield event.plain_result("❌ 请提供衣服编号\n用法: /衣服详情 001")
                return

            # 搜索指定编号的衣服
            detail = self.clothing_manager.get_clothing_detail(number)

            if not detail:
                yield event.plain_result(f"❌ 未找到编号为 '{number}' 的衣服")
                return

            category, name, tags = detail

            detail_info = []
            detail_info.append(f"👗 衣服详情: {number}")
            detail_info.append("")
            detail_info.append(f"📂 分类: {category}")
            detail_info.append(f"📝 名称: {name}")
            detail_info.append(f"🏷️ 标签:")
            detail_info.append(f"{tags}")

            result_text = "\n".join(detail_info)

            forward_chain = self._create_fake_forward_message(
                f"衣服详情: {number}", result_text, "衣服管理"
            )
            yield event.chain_result(forward_chain)

        except Exception as e:
            logger.error(f"查看衣服详情失败: {e}")
            yield event.plain_result(f"❌ 查看衣服详情失败: {str(e)}")

    async def _resolve_statistics_user(self, event, ranking: dict) -> dict:
        """Resolve the QQ display name and avatar for one ranked user.

        Args:
            event: Current AstrBot message event.
            ranking: Usage ranking entry containing a user ID and counts.

        Returns:
            Ranking entry enriched with nickname and avatar bytes.
        """
        user_id = str(ranking.get("user_id") or "")
        group_id = self._current_group_id(event)
        cache_key = f"statistics-user:{group_id}:{user_id}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            return {**ranking, **cached}

        nickname = user_id or "未知用户"
        avatar = None
        if (
            user_id.isdigit()
            and getattr(event, "get_platform_name", lambda: "")() == "aiocqhttp"
        ):
            member_info = {}
            bot = getattr(event, "bot", None)
            if bot is not None and group_id.isdigit():
                try:
                    member_info = (
                        await bot.get_group_member_info(
                            group_id=int(group_id), user_id=int(user_id)
                        )
                        or {}
                    )
                except Exception as exc:
                    logger.debug(f"Failed to get QQ group member {user_id}: {exc}")
            if bot is not None and not member_info:
                try:
                    member_info = (
                        await bot.get_stranger_info(user_id=int(user_id)) or {}
                    )
                except Exception as exc:
                    logger.debug(f"Failed to get QQ profile {user_id}: {exc}")
            nickname = str(
                member_info.get("card")
                or member_info.get("nickname")
                or member_info.get("nick")
                or user_id
            )
            nickname = " ".join(nickname.split()) or user_id

            try:
                avatar_url = (
                    f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
                )
                async with httpx.AsyncClient(
                    timeout=8.0, follow_redirects=True
                ) as client:
                    response = await client.get(avatar_url)
                    response.raise_for_status()
                    avatar = response.content
            except Exception as exc:
                logger.debug(f"Failed to download QQ avatar {user_id}: {exc}")

        profile = {"nickname": nickname, "avatar": avatar}
        self.cache.set(cache_key, profile)
        return {**ranking, **profile}

    @staticmethod
    def _normalize_account_tokens(entries: Any) -> List[dict]:
        """Normalize the ``account_tokens`` configuration into plain dicts.

        Args:
            entries: Raw configuration value. AstrBot stores ``template_list``
                entries as dictionaries carrying ``name``/``api_token``/``enable``.

        Returns:
            Only the dictionary entries, copied so callers cannot mutate config.
        """
        if not isinstance(entries, list):
            return []
        return [dict(entry) for entry in entries if isinstance(entry, dict)]

    def _collect_quota_accounts(self) -> List[Tuple[str, str]]:
        """Resolve which accounts ``/账户额度`` should query.

        The shared ``api_token`` is always listed first as ``主账户``; every enabled
        ``account_tokens`` entry follows in configuration order. Entries without a
        token, disabled entries and repeated tokens are skipped so one NovelAI
        account is never listed twice.

        Returns:
            ``(display label, persistent token)`` pairs ready for querying.
        """
        accounts: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        main_token = str(getattr(self, "api_token", "") or "").strip()
        if main_token:
            accounts.append((Constants.ACCOUNT_QUOTA_MAIN_LABEL, main_token))
            seen.add(main_token)

        entries = self._normalize_account_tokens(
            getattr(self, "account_tokens", []) or []
        )
        for entry in entries:
            enabled = entry.get("enable", True)
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() not in {"", "0", "false", "no", "off"}
            if not enabled:
                continue

            token = str(entry.get("api_token", "") or "").strip()
            if not token or token in seen:
                continue

            label = str(entry.get("name", "") or "").strip()
            accounts.append((label or f"账户{len(accounts) + 1}", token))
            seen.add(token)

        return accounts

    async def _fetch_account_quota(self, api_token: str) -> dict:
        """Query one account's NovelAI subscription snapshot.

        Args:
            api_token: Persistent NovelAI token of the account to inspect.

        Returns:
            Parsed ``GET /user/data`` payload holding ``subscription`` and
            ``priority`` (remaining quota lives in ``subscription.usage.percent``).

        Raises:
            httpx.HTTPError: When NovelAI rejects the token or the request fails.
        """
        client_config, base_headers = await self._create_http_client_and_headers(
            api_token=api_token, use_newapi=False
        )
        client_config["timeout"] = Constants.ACCOUNT_QUOTA_TIMEOUT
        api_url = f"{Constants.NOVELAI_API_BASE}{Constants.ACCOUNT_QUOTA_ENDPOINT}"

        # 网络代理经常抖动，因此沿用画图请求的重试配置；认证或参数类错误重试
        # 没有意义，且交互式命令不应等待太久，所以额外限制最大尝试次数。
        max_attempts = 1
        if self.enable_request_retry:
            max_attempts = min(
                max(0, self.request_retry_count) + 1,
                Constants.ACCOUNT_QUOTA_MAX_ATTEMPTS,
            )
        max_retry_delay = max(0.0, self.request_retry_max_delay)
        last_error: Optional[httpx.HTTPError] = None

        for attempt in range(max_attempts):
            if attempt > 0:
                base_delay = min(
                    Constants.RETRY_BACKOFF_FACTOR**attempt, max_retry_delay
                )
                retry_delay = base_delay * (0.5 + random.random() * 0.5)
                logger.debug(f"账户额度查询重试 {attempt + 1}/{max_attempts}")
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)

            headers = dict(base_headers)
            headers["x-correlation-id"] = hashlib.sha256(
                f"{time.time_ns()}:{attempt}:{random.random()}".encode()
            ).hexdigest()[:6]

            try:
                async with httpx.AsyncClient(**client_config) as client:
                    response = await client.get(api_url, headers=headers)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = exc

        raise last_error or httpx.HTTPError("账户额度查询失败")

    @staticmethod
    def _format_subscription_tier(tier: Any) -> str:
        """Translate the numeric NovelAI subscription tier into its name."""
        try:
            tier_number = int(tier)
        except (TypeError, ValueError):
            return "未知档位"
        return Constants.ACCOUNT_TIER_LABELS.get(tier_number, f"Tier {tier_number}")

    @staticmethod
    def _describe_quota_error(error: BaseException) -> str:
        """Translate a quota query failure into a short operator hint."""
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            if status in (401, 403):
                return f"Token 无效或不可用于额度查询（HTTP {status}）"
            return f"接口返回异常（HTTP {status}）"
        if isinstance(error, httpx.TimeoutException):
            return "请求超时"
        if isinstance(error, httpx.HTTPError):
            return "网络请求失败"
        if isinstance(error, ValueError):
            return "接口返回数据异常"
        return f"未知错误：{error}"

    @staticmethod
    def _format_anlas_balance(training_steps: Any) -> str:
        """Render the account's Anlas balance.

        NovelAI 把 Anlas 余额放在 ``subscription.trainingStepsLeft`` 里（官网
        购买页把它标为 "Your Subscription Anlas" / "Your Paid Anlas"），因此这里
        直接沿用这两个字段，而不是 ``/user/cb/get-customer``（该接口拒绝持久 Token）。

        Args:
            training_steps: Raw ``subscription.trainingStepsLeft`` value.

        Returns:
            A display fragment, or an empty string when the payload has no balance.
        """
        if isinstance(training_steps, (int, float)):
            return f"Anlas {int(training_steps)}"
        if not isinstance(training_steps, dict):
            return ""

        fixed = training_steps.get("fixedTrainingStepsLeft")
        purchased = training_steps.get("purchasedTrainingSteps")
        has_fixed = isinstance(fixed, (int, float))
        has_purchased = isinstance(purchased, (int, float))
        if not has_fixed and not has_purchased:
            return ""

        fixed_value = int(fixed or 0)
        purchased_value = int(purchased or 0)
        total = fixed_value + purchased_value
        return f"Anlas {total}（订阅 {fixed_value} + 自购 {purchased_value}）"

    def _format_account_quota_lines(
        self, index: int, label: str, result: Any
    ) -> List[str]:
        """Render one quota query result as chat lines.

        Args:
            index: 1-based position inside the command output.
            label: Account display label.
            result: Parsed subscription payload or the raised exception.

        Returns:
            One or two lines describing that account.
        """
        if isinstance(result, BaseException):
            detail = self._describe_quota_error(result)
            return [f"{index}. {label} · 查询失败：{detail}"]

        payload = result if isinstance(result, dict) else {}
        subscription = payload.get("subscription") or {}
        usage = subscription.get("usage") or {}

        tier = self._format_subscription_tier(subscription.get("tier"))
        header = f"{index}. {label} · {tier}"
        if not subscription.get("active", True):
            header += "（未激活）"

        details = []
        percent = usage.get("percent")
        if isinstance(percent, (int, float)):
            if usage.get("isNegative"):
                details.append("剩余额度 0%（已超额，出图改扣 Anlas）")
            else:
                remaining = max(0, int(percent))
                images = int(
                    round(remaining * Constants.ACCOUNT_QUOTA_IMAGES_PER_PERCENT)
                )
                details.append(f"剩余额度 {remaining}%（约 {images} 张）")
        anlas = self._format_anlas_balance(subscription.get("trainingStepsLeft"))
        if anlas:
            details.append(anlas)
        time_until = usage.get("timeUntilNextPercent")
        if isinstance(time_until, (int, float)) and time_until > 0:
            details.append(f"每天恢复 {round(86400 / time_until, 1)}%")
        expires_at = subscription.get("expiresAt")
        if isinstance(expires_at, (int, float)) and expires_at > 0:
            expires_on = datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d")
            remaining_days = int((expires_at - time.time()) // 86400)
            if remaining_days >= 0:
                details.append(f"到期 {expires_on}（剩 {remaining_days} 天）")
            else:
                details.append(f"已于 {expires_on} 到期")
        if subscription.get("isGracePeriod"):
            details.append("宽限期")

        if not details:
            return [header]
        return [header, "   " + " · ".join(details)]

    @filter.command("账户额度", alias={"账号额度", "额度查询"})
    async def show_account_quota(self, event):
        """展示配置中每个 NovelAI 账户的剩余额度。"""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "账户额度"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        accounts = self._collect_quota_accounts()
        if not accounts:
            yield event.plain_result(
                "❌ 未配置任何 NovelAI Token\n\n"
                "请在插件配置中填写 api_token，或在「多账户额度」中添加账户"
            )
            return

        results = await asyncio.gather(
            *(self._fetch_account_quota(token) for _, token in accounts),
            return_exceptions=True,
        )

        lines = [f"📊 NovelAI 账户额度（{len(accounts)} 个）", ""]
        for index, ((label, _), result) in enumerate(zip(accounts, results), start=1):
            if isinstance(result, BaseException):
                logger.warning(f"账户额度查询失败（{label}）: {result}")
            lines.extend(self._format_account_quota_lines(index, label, result))

        yield event.plain_result("\n".join(lines))

    @filter.command("画图统计", alias={"用户画图排行"})
    async def show_statistics(self, event):
        """Render drawing activity, user rankings, and local status as one image."""
        # 检查白名单
        if not self._check_whitelist(event):
            return

        # 检查管理员权限
        if not self._check_admin_permission(event, "画图统计"):
            yield event.plain_result("❌ 仅管理员可使用此命令")
            return

        try:
            global_stats = self.usage_manager.get_global_statistics(
                quota_scope=self._resolve_quota_scope(
                    event, str(event.get_sender_id() or "")
                )
            )
            rankings = global_stats.get("user_rankings", [])[:10]
            daily_rankings = global_stats.get("daily_user_rankings", [])[:10]
            resolved_users = await asyncio.gather(
                *(
                    self._resolve_statistics_user(event, item)
                    for item in [*rankings, *daily_rankings]
                ),
                return_exceptions=True,
            )
            leaderboards = []
            for ranking_group, resolved_group in (
                (rankings, resolved_users[: len(rankings)]),
                (daily_rankings, resolved_users[len(rankings) :]),
            ):
                leaderboard = []
                for ranking, resolved in zip(ranking_group, resolved_group):
                    if isinstance(resolved, Exception):
                        logger.debug(
                            f"Failed to resolve ranked user {ranking['user_id']}: {resolved}"
                        )
                        leaderboard.append(
                            {
                                **ranking,
                                "nickname": ranking["user_id"],
                                "avatar": None,
                            }
                        )
                    else:
                        leaderboard.append(resolved)
                leaderboards.append(leaderboard)
            leaderboard, daily_leaderboard = leaderboards

            image_bytes = await asyncio.to_thread(
                render_statistics_card,
                global_stats,
                {},
                leaderboard,
                datetime.now(),
                daily_leaderboard,
            )
            yield event.chain_result([Comp.Image.fromBytes(image_bytes)])
        except Exception as exc:
            logger.exception(f"Failed to render drawing statistics: {exc}")
            yield event.plain_result("生成统计图片失败，请稍后重试")

    async def _call_novelai_txt2img(
        self,
        prompt: str,
        width: int = 832,
        height: int = 1216,
        steps: int = 28,
        scale: float = Constants.DEFAULT_SCALE,
        sampler: str = Constants.DEFAULT_SAMPLER,
        drawing_settings: Optional[dict] = None,
        api_token: str | None = None,
        use_newapi: bool | None = None,
    ) -> tuple:
        """Call the NovelAI text-to-image API.

        Args:
            prompt: Positive image prompt.
            width: Output width in pixels.
            height: Output height in pixels.
            steps: Number of sampling steps.
            scale: Prompt guidance scale.
            sampler: NovelAI sampler identifier.
            drawing_settings: Effective settings for the current session.
            api_token: Optional per-request New API key.
            use_newapi: Optional request-level backend override.

        Returns:
            Image response bytes and metadata, or ``None`` and error metadata.
        """
        correlation_id = ""
        try:
            settings = drawing_settings or {}
            model_name = settings.get("model", self.default_model)
            request_uses_newapi = (
                self._uses_newapi() if use_newapi is None else use_newapi
            )

            # 参数验证
            self._validate_common_api_params(
                prompt,
                steps,
                scale,
                model_name,
                api_token=api_token,
                use_newapi=use_newapi,
            )

            # 创建HTTP客户端和头部
            client_config, headers = await self._create_http_client_and_headers(
                api_token, use_newapi
            )

            async with httpx.AsyncClient(**client_config) as client:
                try:
                    parameters = self.get_model_parameters(
                        model_name,
                        prompt,
                        width,
                        height,
                        steps,
                        scale,
                        sampler,
                        drawing_settings=settings,
                    )
                    api_input = prompt
                    if self.uses_v4_prompt_structure(model_name):
                        api_input = parameters["v4_prompt"]["caption"]["base_caption"]

                    payload = {
                        "input": api_input,
                        "model": model_name,
                        "action": "generate",
                        "parameters": parameters,
                    }
                    api_url = self.get_api_url("generate-image", use_newapi=use_newapi)
                    logger.info(f"发送NovelAI请求到: {api_url}")
                    logger.info(
                        f"模型: {model_name} "
                        f"({self.get_model_version_label(model_name)}), 采样器: {sampler}"
                    )
                    logger.info(f"实际发送的Prompt长度: {len(prompt)} 字符")

                    # Retry any failed request with exponential backoff plus jitter.
                    max_attempts = (
                        max(0, self.request_retry_count) + 1
                        if self.enable_request_retry
                        else 1
                    )
                    max_retry_delay = max(0.0, self.request_retry_max_delay)

                    for attempt in range(max_attempts):
                        if attempt > 0:
                            base_delay = min(
                                Constants.RETRY_BACKOFF_FACTOR**attempt,
                                max_retry_delay,
                            )
                            retry_delay = base_delay * (0.5 + random.random() * 0.5)
                            logger.warning(
                                f"重试请求 ({attempt + 1}/{max_attempts})，"
                                f"{retry_delay:.1f} 秒后重试"
                            )
                            if retry_delay > 0:
                                await asyncio.sleep(retry_delay)

                        correlation_id = hashlib.sha256(
                            f"{time.time_ns()}:{attempt}:{random.random()}".encode()
                        ).hexdigest()[:6]
                        headers["x-correlation-id"] = correlation_id

                        try:
                            response = await client.post(
                                api_url, headers=headers, json=payload
                            )
                        except httpx.TimeoutException as e:
                            # 超时错误 - 可以重试
                            if attempt < max_attempts - 1:
                                logger.warning(f"请求超时: {str(e)}，将重试")
                                continue
                            raise NAIErrors.NetworkError("请求超时 (60秒)", e)
                        except httpx.ConnectError as e:
                            # 连接错误 - 可以重试
                            if attempt < max_attempts - 1:
                                logger.warning(f"连接失败: {str(e)}，将重试")
                                continue
                            raise NAIErrors.NetworkError(f"连接失败: {str(e)}", e)
                        except httpx.RequestError as e:
                            # 其他请求错误（网络问题） - 可以重试
                            if attempt < max_attempts - 1:
                                logger.warning(f"网络请求失败: {str(e)}，将重试")
                                continue
                            raise NAIErrors.NetworkError(f"请求失败: {str(e)}", e)

                        logger.info(f"响应状态: {response.status_code}")

                        # 处理不同的状态码
                        if response.status_code in (200, 201):
                            # 成功
                            image_data = response.content
                            logger.info(
                                f"成功获取响应数据，大小: {len(image_data)} bytes"
                            )
                            meta_info = {"success": True}
                            if attempt:
                                meta_info["retries"] = attempt
                            return image_data, meta_info
                        elif response.status_code == 401:
                            # 认证失败
                            logger.error(
                                "画图请求认证失败: backend=%s, url=%s, key=%s, body=%s",
                                "new-api" if request_uses_newapi else "novelai",
                                api_url,
                                self._mask_api_key(
                                    api_token
                                    or (
                                        getattr(self, "newapi_api_key", "")
                                        if request_uses_newapi
                                        else getattr(self, "api_token", "")
                                    )
                                ),
                                response.text[:300],
                            )
                            error = NAIErrors.AuthenticationError(
                                response.status_code, response.text
                            )
                        elif response.status_code == 429:
                            # 超出配额
                            error = NAIErrors.QuotaExceededError(
                                response.status_code, response.text
                            )
                        elif response.status_code >= 500:
                            # 服务器错误
                            logger.warning(
                                f"NovelAI 服务器错误详情: status={response.status_code}, "
                                f"correlation_id={correlation_id}, "
                                f"body={response.text[:300]}"
                            )
                            error = NAIErrors.ServerError(
                                response.status_code, response.text
                            )
                        else:
                            # 其他API错误
                            error = NAIErrors.APIError(
                                response.status_code, response.text
                            )

                        # 所有错误都重试，重试次数用尽后再抛出
                        if (
                            attempt < max_attempts - 1
                            and response.status_code != 401
                        ):
                            logger.warning(
                                f"请求失败 ({response.status_code})，将重试: {error}"
                            )
                            continue
                        raise error
                except NAIErrors.APIError:
                    # Preserve HTTP status errors so the outer handler can classify them.
                    raise
                except Exception as e:
                    # 其他异常
                    logger.error(f"HTTP客户端异常: {e}")
                    raise NAIErrors.NetworkError(f"客户端异常: {str(e)}", e)

        except NAIErrors.NetworkError as e:
            logger.error(f"网络错误: {e}")
            return None, {"error": str(e), "error_type": "network"}
        except NAIErrors.AuthenticationError as e:
            logger.error(f"认证错误: {e}")
            return None, {"error": str(e), "error_type": "auth"}
        except NAIErrors.QuotaExceededError as e:
            logger.error(f"配额超限: {e}")
            return None, {"error": str(e), "error_type": "quota"}
        except NAIErrors.ServerError as e:
            logger.error(f"服务器错误: {e}")
            return None, {
                "error": str(e),
                "error_type": "server",
                "correlation_id": correlation_id,
            }
        except NAIErrors.APIError as e:
            logger.error(f"API错误: {e}")
            return None, {
                "error": str(e),
                "error_type": "api",
                "status_code": e.status_code,
                "response_text": e.response_text,
            }
        except NAIErrors.ConfigurationError as e:
            logger.error(f"配置错误: {e}")
            return None, {"error": str(e), "error_type": "config"}
        except NAIErrors.ParameterError as e:
            logger.error(f"参数错误: {e}")
            return None, {"error": str(e), "error_type": "param"}
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return None, {"error": f"未知错误: {str(e)}", "error_type": "unknown"}

    async def _call_novelai_img2img(
        self,
        prompt: str,
        reference_image: str,
        width: int = 832,
        height: int = 1216,
        steps: int = 28,
        scale: float = Constants.DEFAULT_SCALE,
        strength: float = 0.7,
        noise: float = 0.2,
        drawing_settings: Optional[dict] = None,
        api_token: str | None = None,
        use_newapi: bool | None = None,
    ) -> tuple:
        """Call the NovelAI image-to-image API.

        Args:
            prompt: Positive image prompt.
            reference_image: Base64-encoded reference image.
            width: Output width in pixels.
            height: Output height in pixels.
            steps: Number of sampling steps.
            scale: Prompt guidance scale.
            strength: Image-to-image transformation strength.
            noise: Image-to-image noise amount.
            drawing_settings: Effective settings for the current session.
            api_token: Optional per-request New API key.
            use_newapi: Optional request-level backend override.

        Returns:
            Image response bytes and metadata, or ``None`` and error metadata.
        """
        try:
            settings = drawing_settings or {}
            model_name = settings.get("model", self.default_model)
            sampler = settings.get("sampler", self.default_sampler)
            request_uses_newapi = (
                self._uses_newapi() if use_newapi is None else use_newapi
            )

            # 参数验证
            self._validate_common_api_params(
                prompt,
                steps,
                scale,
                model_name,
                api_token=api_token,
                use_newapi=use_newapi,
            )

            if not reference_image:
                raise NAIErrors.ParameterError("参考图片不能为空")

            if strength < 0.1 or strength > 0.9:
                raise NAIErrors.ParameterError(
                    f"图生图强度 {strength} 超出有效范围 (0.1-0.9)"
                )

            if noise < 0.0 or noise > 1.0:
                raise NAIErrors.ParameterError(
                    f"图生图噪声 {noise} 超出有效范围 (0.0-1.0)"
                )

            # 创建HTTP客户端和头部
            client_config, headers = await self._create_http_client_and_headers(
                api_token, use_newapi
            )

            async with httpx.AsyncClient(**client_config) as client:
                try:
                    # 获取基础参数
                    parameters = self.get_model_parameters(
                        model_name,
                        prompt,
                        width,
                        height,
                        steps,
                        scale,
                        sampler,
                        drawing_settings=settings,
                    )

                    # 添加图生图特有的参数
                    parameters.update(
                        {
                            "image": reference_image,
                            "strength": strength,
                            "noise": noise,
                        }
                    )
                    api_input = prompt
                    if self.uses_v4_prompt_structure(model_name):
                        api_input = parameters["v4_prompt"]["caption"]["base_caption"]

                    payload = {
                        "input": api_input,
                        "model": model_name,
                        "action": "img2img",
                        "parameters": parameters,
                    }

                    api_url = self.get_api_url("generate-image", use_newapi=use_newapi)
                    logger.info(
                        f"发送图生图请求，模型: {model_name}, action: img2img, "
                        f"强度: {strength}, 噪声: {noise}"
                    )

                    # 发送请求 - 任何失败都自动重试，重试次数用尽后再返回错误
                    max_attempts = (
                        max(0, self.request_retry_count) + 1
                        if self.enable_request_retry
                        else 1
                    )
                    max_retry_delay = max(0.0, self.request_retry_max_delay)

                    for attempt in range(max_attempts):
                        if attempt > 0:
                            base_delay = min(
                                Constants.RETRY_BACKOFF_FACTOR**attempt,
                                max_retry_delay,
                            )
                            retry_delay = base_delay * (0.5 + random.random() * 0.5)
                            logger.warning(
                                f"图生图重试请求 ({attempt + 1}/{max_attempts})，"
                                f"{retry_delay:.1f} 秒后重试"
                            )
                            if retry_delay > 0:
                                await asyncio.sleep(retry_delay)

                        correlation_id = hashlib.sha256(
                            f"{time.time_ns()}:{attempt}:{random.random()}".encode()
                        ).hexdigest()[:6]
                        headers["x-correlation-id"] = correlation_id

                        try:
                            response = await client.post(
                                api_url, headers=headers, json=payload
                            )
                        except httpx.TimeoutException as e:
                            # 超时错误 - 可以重试
                            if attempt < max_attempts - 1:
                                logger.warning(f"图生图请求超时: {str(e)}，将重试")
                                continue
                            raise NAIErrors.NetworkError("请求超时 (60秒)", e)
                        except httpx.ConnectError as e:
                            # 连接错误 - 可以重试
                            if attempt < max_attempts - 1:
                                logger.warning(f"图生图连接失败: {str(e)}，将重试")
                                continue
                            raise NAIErrors.NetworkError(f"连接失败: {str(e)}", e)
                        except httpx.RequestError as e:
                            # 请求错误（网络问题）- 可以重试
                            if attempt < max_attempts - 1:
                                logger.warning(f"图生图网络请求失败: {str(e)}，将重试")
                                continue
                            raise NAIErrors.NetworkError(f"请求失败: {str(e)}", e)

                        logger.info(f"图生图响应状态: {response.status_code}")

                        # 处理不同的状态码
                        if response.status_code in (200, 201):
                            # 成功
                            meta_info = {"success": True}
                            if attempt:
                                meta_info["retries"] = attempt
                            return response.content, meta_info
                        elif response.status_code == 401:
                            # 认证失败
                            logger.error(
                                "图生图请求认证失败: backend=%s, url=%s, key=%s, body=%s",
                                "new-api" if request_uses_newapi else "novelai",
                                api_url,
                                self._mask_api_key(
                                    api_token
                                    or (
                                        getattr(self, "newapi_api_key", "")
                                        if request_uses_newapi
                                        else getattr(self, "api_token", "")
                                    )
                                ),
                                response.text[:300],
                            )
                            error = NAIErrors.AuthenticationError(
                                response.status_code, response.text
                            )
                        elif response.status_code == 429:
                            # 超出配额
                            error = NAIErrors.QuotaExceededError(
                                response.status_code, response.text
                            )
                        elif response.status_code >= 500:
                            # 服务器错误 - 记录详细信息
                            error_details = {
                                "status_code": response.status_code,
                                "response_text": response.text,
                                "correlation_id": correlation_id,
                                "request_params": {
                                    "model": model_name,
                                    "action": "img2img",
                                    "strength": strength,
                                    "noise": noise,
                                    "prompt_length": len(prompt),
                                    "steps": steps,
                                    "scale": scale,
                                    "width": width,
                                    "height": height,
                                },
                            }
                            logger.warning(f"图生图服务器错误详情: {error_details}")
                            error = NAIErrors.ServerError(
                                response.status_code, response.text
                            )
                        else:
                            # 其他API错误
                            error = NAIErrors.APIError(
                                response.status_code, response.text
                            )

                        # 所有错误都重试，重试次数用尽后再抛出
                        if (
                            attempt < max_attempts - 1
                            and response.status_code != 401
                        ):
                            logger.warning(
                                f"图生图请求失败 ({response.status_code})，"
                                f"将重试: {error}"
                            )
                            continue
                        raise error
                except httpx.TimeoutException as e:
                    # 超时错误
                    raise NAIErrors.NetworkError(f"请求超时 (60秒)", e)

        except NAIErrors.NetworkError as e:
            logger.error(f"图生图网络错误: {e}")
            return None, {"error": str(e), "error_type": "network"}
        except NAIErrors.AuthenticationError as e:
            logger.error(f"图生图认证错误: {e}")
            return None, {"error": str(e), "error_type": "auth"}
        except NAIErrors.QuotaExceededError as e:
            logger.error(f"图生图配额超限: {e}")
            return None, {"error": str(e), "error_type": "quota"}
        except NAIErrors.ServerError as e:
            logger.error(f"图生图服务器错误: {e}")
            return None, {"error": str(e), "error_type": "server"}
        except NAIErrors.APIError as e:
            logger.error(f"图生图API错误: {e}")
            return None, {
                "error": str(e),
                "error_type": "api",
                "status_code": e.status_code,
                "response_text": e.response_text,
            }
        except NAIErrors.ConfigurationError as e:
            logger.error(f"图生图配置错误: {e}")
            return None, {"error": str(e), "error_type": "config"}
        except NAIErrors.ParameterError as e:
            logger.error(f"图生图参数错误: {e}")
            return None, {"error": str(e), "error_type": "param"}
        except Exception as e:
            logger.error(f"图生图未知错误: {e}")
            return None, {"error": f"未知错误: {str(e)}", "error_type": "unknown"}

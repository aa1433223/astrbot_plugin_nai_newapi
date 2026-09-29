import json
from types import SimpleNamespace

import pytest

from data.plugins.astrbot_plugin_novelai.main import Constants, NovelAIPlugin


def make_plugin(backend: str = "newapi") -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin.api_backend = backend
    plugin.api_token = "novelai-token"
    plugin.api_proxy = "https://novelai-proxy.example.com"
    plugin.novelai_proxy = "socks5://127.0.0.1:7890"
    plugin.newapi_base_url = "http://localhost:3000/"
    plugin.newapi_api_key = "newapi-key"
    return plugin


def test_newapi_backend_builds_native_novelai_endpoint():
    plugin = make_plugin()

    assert plugin.get_api_url() == "http://localhost:3000/ai/generate-image"
    assert plugin._get_active_api_token() == "newapi-key"


@pytest.mark.asyncio
async def test_newapi_backend_uses_newapi_key_without_novelai_network_proxy():
    plugin = make_plugin()

    client_config, headers = await plugin._create_http_client_and_headers()

    assert "proxy" not in client_config
    assert headers["Authorization"] == "Bearer newapi-key"


@pytest.mark.asyncio
async def test_request_api_key_overrides_global_newapi_key():
    plugin = make_plugin("novelai")

    client_config, headers = await plugin._create_http_client_and_headers(
        "sk-personal", use_newapi=True
    )

    assert "proxy" not in client_config
    assert headers["Authorization"] == "Bearer sk-personal"
    assert (
        plugin.get_api_url(use_newapi=True) == "http://localhost:3000/ai/generate-image"
    )


@pytest.mark.asyncio
async def test_request_can_force_novelai_for_free_quota():
    plugin = make_plugin()

    client_config, headers = await plugin._create_http_client_and_headers(
        use_newapi=False
    )

    assert client_config["proxy"] == "socks5://127.0.0.1:7890"
    assert headers["Authorization"] == "Bearer novelai-token"
    assert (
        plugin.get_api_url(use_newapi=False)
        == "https://novelai-proxy.example.com/ai/generate-image"
    )


def test_newapi_configuration_requires_url_and_key():
    plugin = make_plugin()

    errors = plugin.validate_config(
        {
            "api_backend": "newapi",
            "newapi_base_url": "localhost:3000",
            "newapi_api_key": "",
        }
    )

    assert "newapi_base_url" in errors
    assert "newapi_api_key" in errors
    assert "api_token" in errors


@pytest.mark.parametrize("group_daily_image_limit", [-1, 1.5, True, "100"])
def test_group_daily_image_limit_must_be_a_non_negative_integer(
    group_daily_image_limit,
):
    plugin = make_plugin("novelai")

    errors = plugin.validate_config(
        {
            "api_backend": "novelai",
            "api_token": "novelai-token",
            "newapi_base_url": "http://localhost:3000",
            "group_daily_image_limit": group_daily_image_limit,
        }
    )

    assert "group_daily_image_limit" in errors


def test_group_daily_image_limit_allows_disabling_free_quota():
    plugin = make_plugin("novelai")

    errors = plugin.validate_config(
        {
            "api_backend": "novelai",
            "api_token": "novelai-token",
            "newapi_base_url": "http://localhost:3000",
            "group_daily_image_limit": 0,
        }
    )

    assert "group_daily_image_limit" not in errors


def test_group_daily_image_limit_update_takes_effect_immediately():
    plugin = make_plugin("novelai")
    plugin.config = {}
    plugin.usage_manager = SimpleNamespace(
        max_daily_usage=50,
        max_group_daily_usage=Constants.GROUP_DAILY_IMAGE_LIMIT,
    )

    plugin.on_config_changed({"group_daily_image_limit": 37, "enable_v5": False})

    assert plugin.group_daily_image_limit == 37
    assert plugin.enable_v5 is False
    assert plugin.usage_manager.max_daily_usage == 0
    assert plugin.usage_manager.max_group_daily_usage == 37


def test_enable_v5_must_be_boolean():
    plugin = make_plugin("novelai")

    errors = plugin.validate_config(
        {
            "api_backend": "novelai",
            "api_token": "novelai-token",
            "newapi_base_url": "http://localhost:3000",
            "enable_v5": "false",
        }
    )

    assert "enable_v5" in errors


def test_ai_video_key_lookup_is_isolated_to_exact_sender(tmp_path):
    plugin = make_plugin()
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps(
            {
                "users": {
                    "10001": {"api_keys": ["sk-owner"]},
                    "10002": {"api_keys": ["sk-sender"]},
                },
                "groups": {
                    "20001": {"enabled": True, "owner_uid": "10001"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert plugin._get_bound_user_api_key("10002") == "sk-sender"
    assert plugin._get_bound_user_api_key("20001") is None
    assert plugin._get_bound_user_api_key("unknown") is None


def test_ai_video_key_lookup_prefers_current_bot_binding(tmp_path):
    plugin = make_plugin()
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps(
            {
                "users": {
                    "10001": {
                        "api_keys": ["sk-old-key", "sk-current-key"],
                    }
                },
                "bot_chat_keys": {
                    "bot-a:10001": {"api_key": "sk-current-key"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert plugin._get_bound_user_api_key("10001", "bot-a") == "sk-current-key"
    assert plugin._get_bound_user_api_key("10001", "bot-b") == "sk-old-key"


def test_free_daily_quota_keeps_global_key_and_selected_model(tmp_path):
    plugin = make_plugin("novelai")
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps({"users": {"10001": {"api_keys": ["sk-personal"]}}}),
        encoding="utf-8",
    )

    class AvailableUsageManager:
        @staticmethod
        def check_group_daily_limit(quota_scope):
            return True

    plugin.usage_manager = AvailableUsageManager()
    settings = {"model": "nai-diffusion-4-5-curated"}

    changed, api_key, use_newapi, personal_key_route = plugin._resolve_daily_generation_route(
        settings, "10001"
    )

    assert changed is False
    assert api_key is None
    assert use_newapi is False
    assert personal_key_route is False
    assert settings["model"] == "nai-diffusion-4-5-curated"
    assert (
        plugin.get_api_url(use_newapi=use_newapi)
        == "https://novelai-proxy.example.com/ai/generate-image"
    )


def test_exhausted_quota_uses_owners_key_and_forces_v5(tmp_path):
    plugin = make_plugin("novelai")
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps({"users": {"10001": {"api_keys": ["sk-personal"]}}}),
        encoding="utf-8",
    )

    class ExhaustedUsageManager:
        @staticmethod
        def check_group_daily_limit(quota_scope):
            return False

    plugin.usage_manager = ExhaustedUsageManager()
    settings = {"model": "nai-diffusion-4-5-curated"}

    changed, api_key, use_newapi, personal_key_route = plugin._resolve_daily_generation_route(
        settings, "10001"
    )

    assert changed is True
    assert api_key == "sk-personal"
    assert use_newapi is True
    assert personal_key_route is True
    assert settings["model"] == Constants.PERSONAL_KEY_MODEL


@pytest.mark.parametrize("quota_available", [True, False])
def test_disabled_v5_forces_v45_without_personal_key_route(tmp_path, quota_available):
    plugin = make_plugin("novelai")
    plugin.enable_v5 = False
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps({"users": {"10001": {"api_keys": ["sk-personal"]}}}),
        encoding="utf-8",
    )
    plugin.usage_manager = SimpleNamespace(
        check_group_daily_limit=lambda quota_scope: quota_available
    )
    settings = {"model": "nai-diffusion-5-curated"}

    changed, api_key, use_newapi, personal_key_route = plugin._resolve_daily_generation_route(
        settings, "10001"
    )

    assert changed is True
    assert api_key is None
    assert use_newapi is False
    assert personal_key_route is False
    assert settings["model"] == Constants.DAILY_FALLBACK_MODEL


def test_exhausted_quota_never_uses_another_users_key(tmp_path):
    plugin = make_plugin("novelai")
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps({"users": {"10001": {"api_keys": ["sk-owner"]}}}),
        encoding="utf-8",
    )

    class ExhaustedUsageManager:
        @staticmethod
        def check_group_daily_limit(quota_scope):
            return False

    plugin.usage_manager = ExhaustedUsageManager()
    settings = {"model": "nai-diffusion-5-curated"}

    changed, api_key, use_newapi, personal_key_route = plugin._resolve_daily_generation_route(
        settings, "10002"
    )

    assert changed is True
    assert api_key is None
    assert use_newapi is False
    assert personal_key_route is False
    assert settings["model"] == Constants.DAILY_FALLBACK_MODEL


def test_personal_v5_access_denial_activates_request_local_v45_fallback():
    plugin = make_plugin("novelai")
    settings = {"model": Constants.PERSONAL_KEY_MODEL}
    meta_info = {
        "error_type": "api",
        "status_code": 403,
        "response_text": json.dumps(
            {
                "error": {
                    "message": (
                        "This token has no access to model "
                        "nai-diffusion-5-full"
                    )
                }
            }
        ),
    }

    activated = plugin._activate_personal_v5_access_fallback(
        settings,
        meta_info,
        personal_key_active=True,
    )

    assert activated is True
    assert settings["model"] == Constants.DAILY_FALLBACK_MODEL
    assert settings[Constants.PERSONAL_V5_FALLBACK_FLAG] is True
    assert plugin._resolve_daily_generation_route(settings, "10001") == (
        False,
        None,
        False,
        False,
    )


@pytest.mark.parametrize(
    ("request_uses_newapi", "status_code", "message"),
    [
        (False, 403, "This token has no access to model nai-diffusion-5-full"),
        (True, 401, "This token has no access to model nai-diffusion-5-full"),
        (True, 403, "request was rejected by content policy"),
    ],
)
def test_personal_v5_fallback_only_handles_newapi_model_permission_denials(
    request_uses_newapi,
    status_code,
    message,
):
    plugin = make_plugin("novelai")
    settings = {"model": Constants.PERSONAL_KEY_MODEL}

    activated = plugin._activate_personal_v5_access_fallback(
        settings,
        {
            "error_type": "api",
            "status_code": status_code,
            "response_text": message,
        },
        personal_key_active=request_uses_newapi,
    )

    assert activated is False
    assert settings == {"model": Constants.PERSONAL_KEY_MODEL}


@pytest.mark.asyncio
async def test_personal_v5_access_denial_retries_once_with_normalized_v45_prompt():
    plugin = make_plugin("novelai")
    settings = {"model": Constants.PERSONAL_KEY_MODEL}
    normalized_models = []
    retried_prompts = []

    async def normalize_prompt(
        prompt,
        event,
        model_name,
        *,
        user_prompt=None,
        drawing_settings=None,
    ):
        normalized_models.append(model_name)
        return "normalized for v4.5"

    async def retry_request(prompt):
        retried_prompts.append(prompt)
        return b"image-bytes", {"success": True}

    plugin.normalize_prompt_for_novelai = normalize_prompt

    image_data, meta_info, prompt, attempted = (
        await plugin._retry_with_daily_v45_on_personal_access_denied(
            None,
            {
                "error_type": "api",
                "status_code": 403,
                "response_text": (
                    "This token has no access to model nai-diffusion-5-full"
                ),
            },
            drawing_settings=settings,
            personal_key_active=True,
            prompt="normalized for v5",
            prompt_before_normalization="original prompt",
            event=SimpleNamespace(),
            scene_user_prompt="original prompt",
            request_callback=retry_request,
        )
    )

    assert image_data == b"image-bytes"
    assert meta_info["fallback_attempted"] is True
    assert meta_info["fallback_model"] == Constants.DAILY_FALLBACK_MODEL
    assert prompt == "normalized for v4.5"
    assert attempted is True
    assert normalized_models == [Constants.DAILY_FALLBACK_MODEL]
    assert retried_prompts == ["normalized for v4.5"]


def test_newapi_only_deployment_draws_with_global_key_without_novelai_token(
    tmp_path,
):
    """A newapi-only deployment (no NovelAI token) can still draw.

    With ``api_backend=newapi`` the free/unbound route must use the operator's
    global New API key, not fall back to an unconfigured NovelAI token.
    """

    plugin = make_plugin("newapi")
    plugin.api_token = ""  # no NovelAI token at all
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps({"users": {}}), encoding="utf-8"
    )
    plugin.usage_manager = SimpleNamespace(
        check_group_daily_limit=lambda quota_scope: True
    )
    settings = {"model": Constants.DEFAULT_MODEL}

    changed, api_key, use_newapi, personal_key_route = (
        plugin._resolve_daily_generation_route(settings, "10002")
    )

    assert changed is False
    assert api_key is None
    assert use_newapi is True
    assert personal_key_route is False
    assert (
        plugin.get_api_url(use_newapi=use_newapi)
        == "http://localhost:3000/ai/generate-image"
    )
    # the request is funded by the global New API key, so no credential error
    plugin._validate_common_api_params(
        prompt="1girl", model_name=Constants.DEFAULT_MODEL, use_newapi=use_newapi
    )

    # the global New API route must never trigger the personal-key V4.5 fallback
    assert (
        plugin._activate_personal_v5_access_fallback(
            {"model": Constants.PERSONAL_KEY_MODEL},
            {
                "error_type": "api",
                "status_code": 403,
                "response_text": (
                    "This token has no access to model nai-diffusion-5-full"
                ),
            },
            personal_key_route,
        )
        is False
    )

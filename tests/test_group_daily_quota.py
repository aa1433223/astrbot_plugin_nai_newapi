import json
from datetime import datetime
from pathlib import Path

from data.plugins.astrbot_plugin_novelai.main import (
    Constants,
    NovelAIPlugin,
    UsageManager,
)


def make_plugin(api_token: str = "novelai-token") -> NovelAIPlugin:
    plugin = object.__new__(NovelAIPlugin)
    plugin.api_backend = "novelai"
    plugin.api_token = api_token
    plugin.api_proxy = ""
    plugin.novelai_proxy = ""
    plugin.newapi_base_url = "http://localhost:3000/"
    plugin.newapi_api_key = "newapi-key"
    return plugin


def make_manager(max_group_daily: int = Constants.GROUP_DAILY_IMAGE_LIMIT):
    manager = object.__new__(UsageManager)
    manager.max_group_daily_usage = max_group_daily
    manager.group_usage_data = {}
    manager.usage_data = {}
    manager.last_use = {}
    manager.queue = []
    manager.processing_request = None
    return manager


class FakeGroupEvent:
    def __init__(self, group_id: str, origin: str = ""):
        self.group_id = group_id
        self.unified_msg_origin = origin or f"aiocqhttp:GroupMessage:{group_id}"

    def get_group_id(self):
        return self.group_id


class FakePrivateEvent:
    def __init__(self, sender_id: str):
        self.unified_msg_origin = f"aiocqhttp:FriendMessage:{sender_id}"


def test_resolve_quota_scope_isolates_groups_and_private_sessions():
    assert NovelAIPlugin._resolve_quota_scope(FakeGroupEvent("20001")) == "group:20001"
    assert (
        NovelAIPlugin._resolve_quota_scope(FakePrivateEvent("10001"))
        == "session:aiocqhttp:FriendMessage:10001"
    )
    assert NovelAIPlugin._resolve_quota_scope(None, "10001") == "user:10001"
    assert NovelAIPlugin._resolve_quota_scope(None) == "global"


def test_record_usage_charges_only_the_given_scope(monkeypatch):
    today = datetime.now().strftime("%Y-%m-%d")
    manager = make_manager()
    monkeypatch.setattr(manager, "_save_usage_data", lambda: None)
    monkeypatch.setattr(manager, "_save_group_usage_data", lambda: None)

    manager.record_usage("10001", "group:20001")
    manager.record_usage("10002", "group:20001")
    manager.record_usage("10001", "group:20002")
    manager.record_usage("10003")

    assert manager.get_group_daily_usage("group:20001") == 2
    assert manager.get_group_daily_usage("group:20002") == 1
    assert manager.get_group_daily_usage("group:20003") == 0
    assert manager.get_group_daily_remaining("group:20001") == 3
    assert manager.usage_data["10001"][today] == 2
    assert manager.usage_data["10003"][today] == 1

    manager.group_usage_data["group:20001"][today] = 5

    assert not manager.check_group_daily_limit("group:20001")
    assert manager.check_group_daily_limit("group:20002")


def test_exhausted_group_falls_back_while_other_groups_keep_v5(tmp_path):
    plugin = make_plugin()
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps({"users": {}}), encoding="utf-8"
    )
    today = datetime.now().strftime("%Y-%m-%d")
    manager = make_manager()
    manager.group_usage_data["group:20001"] = {today: Constants.GROUP_DAILY_IMAGE_LIMIT}
    plugin.usage_manager = manager

    exhausted_settings = {"model": "nai-diffusion-5-full"}
    changed, api_key, use_newapi = plugin._resolve_daily_generation_route(
        exhausted_settings, "10002", FakeGroupEvent("20001")
    )

    assert (changed, api_key, use_newapi) == (True, None, False)
    assert exhausted_settings["model"] == Constants.DAILY_FALLBACK_MODEL

    fresh_settings = {"model": "nai-diffusion-5-full"}
    assert plugin._resolve_daily_generation_route(
        fresh_settings, "10002", FakeGroupEvent("20002")
    ) == (False, None, False)
    assert fresh_settings["model"] == "nai-diffusion-5-full"


def test_group_authorization_never_consumes_the_free_quota(tmp_path):
    plugin = make_plugin()
    plugin.ai_video_user_data_path = tmp_path / "users.json"
    plugin.ai_video_user_data_path.write_text(
        json.dumps(
            {
                "users": {"10001": {"api_keys": ["sk-owner"]}},
                "groups": {"20001": {"enabled": True, "owner_uid": "10001"}},
            }
        ),
        encoding="utf-8",
    )
    manager = make_manager(0)
    plugin.usage_manager = manager

    settings = {"model": "nai-diffusion-5-full"}
    changed, api_key, use_newapi = plugin._resolve_daily_generation_route(
        settings, "10002", FakeGroupEvent("20001")
    )

    assert (changed, api_key, use_newapi) == (False, "sk-owner", True)


def test_group_daily_image_limit_default_matches_the_config_schema():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert Constants.GROUP_DAILY_IMAGE_LIMIT == 5
    assert (
        schema["group_daily_image_limit"]["default"]
        == Constants.GROUP_DAILY_IMAGE_LIMIT
    )

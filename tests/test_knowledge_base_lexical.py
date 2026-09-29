from pathlib import Path

from data.plugins.astrbot_plugin_novelai.knowledge_base_manager import KnowledgeBaseManager


def test_lexical_fallback_works_without_embeddings():
    kb_dir = Path(__file__).resolve().parents[1] / "knowledge_base"
    manager = KnowledgeBaseManager(str(kb_dir))
    manager.load()
    result = manager.resolve_user_request("达妮娅在海滩边玩耍")
    assert "beach" in result["extra_tags"]
    assert "ocean" in result["extra_tags"]

"""
知识库管理器 - KnowledgeBaseManager
供画图命令使用，将用户的自然语言意图解析为 NAI tag。

可选使用 OpenAI 兼容 Embeddings API 构建语义索引；未传入 API 配置时只加载基础数据。
"""

import json
import os
from typing import Optional

try:
    from astrbot.api import logger as _logger
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# 向量后端基类 & 实现
# ---------------------------------------------------------------------------

class _EmbeddingBackend:
    """嵌入向量后端基类"""
    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class _OpenAIEmbeddingBackend(_EmbeddingBackend):
    """调用 OpenAI 兼容 Embeddings API"""

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        import urllib.request
        import urllib.error
        endpoint = f"{self._api_url}/embeddings"
        payload = json.dumps({
            "input": texts,
            "model": self._model_name,
        }).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [item["embedding"] for item in data["data"]]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"[KB-Vector] Embedding API 错误 {e.code}: {body}")


# ---------------------------------------------------------------------------
# 向量索引
# ---------------------------------------------------------------------------

class _VectorIndex:
    """轻量余弦相似度索引（无需 faiss）"""

    def __init__(self):
        self._vectors: list[list[float]] = []   # shape: [N, dim]
        self._items: list[dict] = []             # 对应的知识库条目

    def add(self, item: dict, vector: list[float]):
        self._vectors.append(vector)
        self._items.append(item)

    def search(self, query_vec: list[float], top_k: int, threshold: float) -> list[tuple[float, dict]]:
        """返回 (score, item) 列表，按 score 降序"""
        if not self._vectors:
            return []
        import math
        q_norm = math.sqrt(sum(v * v for v in query_vec)) or 1e-9
        results = []
        for vec, item in zip(self._vectors, self._items):
            dot = sum(a * b for a, b in zip(query_vec, vec))
            v_norm = math.sqrt(sum(v * v for v in vec)) or 1e-9
            score = dot / (q_norm * v_norm)
            if score >= threshold:
                results.append((score, item))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def __len__(self):
        return len(self._items)


# ---------------------------------------------------------------------------
# 主管理器
# ---------------------------------------------------------------------------

class KnowledgeBaseManager:
    _LEXICAL_ALIASES = {
        "海滩": {
            "tags": ["beach", "ocean", "sea"],
            "category": "background",
        },
    }

    def __init__(self, kb_dir: str,
                 embedding_model: str = "",
                 embedding_api_url: str = "",
                 embedding_api_key: str = "",
                 vector_top_k: int = 5,
                 vector_threshold: float = 0.55):
        """
        参数：
            embedding_model:    API 模型名
            embedding_api_url:  OpenAI 兼容 API 的 base url
            embedding_api_key:  API key
            vector_top_k:       向量检索返回 Top-K 数量
            vector_threshold:   余弦相似度最低阈值（0~1）
        """
        self.kb_dir = kb_dir
        self.embedding_model = embedding_model
        self.embedding_api_url = embedding_api_url
        self.embedding_api_key = embedding_api_key
        self.vector_top_k = vector_top_k
        self.vector_threshold = vector_threshold

        self._intent_index: dict = {}       # 中文意图词 → {tags, zh, outfit_category, ...}
        self._outfits: list = []            # 完整换装套装列表
        self._cat_files = {                 # 分类文件名 → 已加载列表
            "action": [],
            "clothing_style": [],
            "expression": [],
            "face": [],
            "body_hair": [],
            "accessories": [],
            "background": [],
            "environment": [],
            "lighting": [],
            "composition": [],
            "art_style": [],
            "character_design": [],
        }
        self._loaded = False

        # 向量模式专用
        self._embed_backend: Optional[_EmbeddingBackend] = None
        self._vec_index: Optional[_VectorIndex] = None     # 全量条目索引
        self._vec_index_built = False

    # ------------------------------------------------------------------
    # 分类 → resolve 结果键的映射
    # ------------------------------------------------------------------
    _CAT_TO_KEY = {
        "action":         "action_tags",
        "expression":     "expression_tags",
        "clothing_style": "clothing_tags",
        "clothing":       "clothing_tags",
        "background":     "extra_tags",
        "environment":    "extra_tags",
        "lighting":       "extra_tags",
        "composition":    "extra_tags",
        "accessories":    "extra_tags",
        "face":           "extra_tags",
        "body_hair":      "extra_tags",
        "art_style":      "extra_tags",
        "character_design": "extra_tags",
    }

    def load(self):
        """加载所有知识库文件并构建向量索引。"""
        if self._loaded:
            return
        try:
            self._intent_index = self._load_json("intent_index.json", {})
            self._outfits = self._load_json("clothing_outfits.json", [])
            for key in self._cat_files:
                self._cat_files[key] = self._load_json(f"{key}.json", [])
            self._loaded = True
            _logger.info(
                f"[KB] 知识库加载完成: {len(self._intent_index)} 意图, "
                f"{len(self._outfits)} 套装, action={len(self._cat_files['action'])}, "
                f"clothing_style={len(self._cat_files['clothing_style'])}"
            )
            if self.embedding_api_url and self.embedding_model:
                self._init_vector_index()
            else:
                _logger.info("[KB] 未配置 Embeddings API，跳过语义索引构建")
        except Exception as e:
            _logger.error(f"[KB] 知识库加载失败: {e}")

    def _load_json(self, filename: str, default):
        path = os.path.join(self.kb_dir, filename)
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # 向量索引构建
    # ------------------------------------------------------------------

    def _init_vector_index(self):
        """初始化 Embedding 后端并为所有知识库条目构建向量索引。"""
        try:
            if not self.embedding_api_url:
                raise ValueError("未提供 Embeddings API 地址")
            if not self.embedding_model:
                raise ValueError("未提供 Embeddings API 模型名")
            self._embed_backend = _OpenAIEmbeddingBackend(
                self.embedding_api_url,
                self.embedding_api_key,
                self.embedding_model,
            )
        except Exception as e:
            _logger.error(f"[KB-Vector] Embedding 后端初始化失败: {e}")
            return

        _logger.info("[KB-Vector] 开始构建向量索引...")
        self._vec_index = _VectorIndex()
        items_to_index = []

        # intent_index 条目
        for kw, info in self._intent_index.items():
            items_to_index.append({
                "text": info.get("zh", kw) + " " + kw,   # 中文描述 + 意图词
                "tags": info.get("tags", []),
                "category": info.get("category", ""),
                "outfit_category": info.get("outfit_category", ""),
                "exposure_level": info.get("exposure_level", ""),
                "source": "intent",
            })

        # 完整换装套装条目
        for outfit in self._outfits:
            tags = outfit.get("tags", "")
            if not tags:
                continue
            category = outfit.get("category", "")
            name = outfit.get("name", "")
            items_to_index.append({
                "text": f"{category} {name}".strip(),
                "tags": tags,
                "category": "outfit",
                "outfit_category": category,
                "source": "outfit",
                "result_key": "outfit_tags",
            })

        # 各分类文件条目（zh + tag 拼成检索文本）
        cat_to_result_key = self._CAT_TO_KEY
        for cat_key, items in self._cat_files.items():
            result_key = cat_to_result_key.get(cat_key, "extra_tags")
            for item in items:
                zh = item.get("zh", "")
                tag = item.get("tag", "")
                if not zh or not tag:
                    continue
                items_to_index.append({
                    "text": zh,
                    "tags": [tag],
                    "category": cat_key,
                    "outfit_category": "",
                    "source": "cat_file",
                    "result_key": result_key,
                })

        # 批量编码（每批 64 条，避免 API 单次限制）
        BATCH = 64
        texts = [it["text"] for it in items_to_index]
        all_vecs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            try:
                vecs = self._embed_backend.encode(batch)
                all_vecs.extend(vecs)
            except Exception as e:
                _logger.error(f"[KB-Vector] 批次 {i//BATCH} 编码失败: {e}")
                # 用零向量占位，保证索引对齐
                all_vecs.extend([[0.0] * 768] * len(batch))

        for item, vec in zip(items_to_index, all_vecs):
            self._vec_index.add(item, vec)

        self._vec_index_built = True
        _logger.info(f"[KB-Vector] 向量索引构建完成，共 {len(self._vec_index)} 条目")

    # ------------------------------------------------------------------
    # 公开查询接口
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict:
        return {
            "clothing_tags": [],
            "outfit_tags": "",
            "action_tags": [],
            "expression_tags": [],
            "extra_tags": [],
            "exposure_hint": "",
        }

    def resolve_user_request(self, text: str) -> dict:
        """
        将用户自然语言文本解析为结构化的 tag 结果。
        返回 dict:
            clothing_tags   (list[str]): 衣服/丝袜相关 tag
            outfit_tags     (str):       完整换装套装 tags 串
            action_tags     (list[str]): 动作/姿势 tag
            expression_tags (list[str]): 表情 tag
            extra_tags      (list[str]): 其他命中的 tag
            exposure_hint   (str):       "conservative"/"revealing"/"" — 服装暴露程度意图
        """
        if not self._loaded:
            self.load()

        lexical = self._resolve_lexical(text)
        if self._vec_index_built:
            vector = self._resolve_vector(text)
            return self._merge_results(lexical, vector)
        _logger.debug("[KB] 语义索引不可用，使用本地词法解析")
        return lexical

    @staticmethod
    def _merge_results(primary: dict, secondary: dict) -> dict:
        result = KnowledgeBaseManager._empty_result()
        for key in ("clothing_tags", "action_tags", "expression_tags", "extra_tags"):
            result[key] = list(dict.fromkeys(
                list(primary.get(key, [])) + list(secondary.get(key, []))
            ))
        result["outfit_tags"] = (
            primary.get("outfit_tags") or secondary.get("outfit_tags") or ""
        )
        result["exposure_hint"] = (
            primary.get("exposure_hint") or secondary.get("exposure_hint") or ""
        )
        return result

    def _resolve_lexical(self, text: str) -> dict:
        """Resolve exact Chinese intent words without an embedding dependency."""
        result = self._empty_result()
        normalized = str(text or "").casefold()
        if not normalized:
            return result

        entries: list[tuple[str, dict]] = list(self._intent_index.items())
        entries.extend(self._LEXICAL_ALIASES.items())
        entries.sort(key=lambda item: len(str(item[0])), reverse=True)

        outfit_category_hint = ""
        for keyword, info in entries:
            keyword_text = str(keyword or "").casefold()
            if not keyword_text or keyword_text not in normalized:
                continue
            category = str(info.get("category", "") or "")
            tags = info.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            result_key = self._CAT_TO_KEY.get(category, "extra_tags")
            result[result_key].extend(tag for tag in tags if tag)
            outfit_category_hint = (
                outfit_category_hint or str(info.get("outfit_category", "") or "")
            )
            if not result["exposure_hint"]:
                result["exposure_hint"] = str(info.get("exposure_level", "") or "")

        # Category files contain many useful exact Chinese phrases that are not
        # duplicated in intent_index. Longest-first matching avoids a short term
        # overriding a more specific phrase.
        for category, items in self._cat_files.items():
            result_key = self._CAT_TO_KEY.get(category, "extra_tags")
            matches = 0
            sorted_items = sorted(
                items,
                key=lambda item: len(str(item.get("zh", "") or "")),
                reverse=True,
            )
            for item in sorted_items:
                zh = str(item.get("zh", "") or "").strip()
                tag = str(item.get("tag", "") or "").strip()
                if len(zh) < 2 or not tag or zh.casefold() not in normalized:
                    continue
                result[result_key].append(tag)
                matches += 1
                if matches >= 5:
                    break

        if outfit_category_hint:
            outfit = self.get_outfit_by_category(outfit_category_hint)
            if outfit:
                result["outfit_tags"] = str(outfit.get("tags", "") or "")

        for key in ("clothing_tags", "action_tags", "expression_tags", "extra_tags"):
            result[key] = list(dict.fromkeys(result[key]))
        return result

    # ------------------------------------------------------------------
    # vector 检索
    # ------------------------------------------------------------------

    def _resolve_vector(self, text: str) -> dict:
        """向量检索：将 text 编码后与索引做余弦相似度，归并为结构化 tag 结果。"""
        result = self._empty_result()
        outfit_category_hint = None

        try:
            query_vecs = self._embed_backend.encode([text])
            query_vec = query_vecs[0]
        except Exception as e:
            _logger.error(f"[KB-Vector] 查询编码失败: {e}")
            return self._empty_result()

        hits = self._vec_index.search(query_vec, self.vector_top_k, self.vector_threshold)
        _logger.debug(f"[KB-Vector] 检索到 {len(hits)} 条结果（threshold={self.vector_threshold}）")

        for score, item in hits:
            tags = item.get("tags", [])
            cat = item.get("category", "")
            outfit_cat = item.get("outfit_category", "")
            if cat == "clothing_intent":
                level = item.get("exposure_level", "")
                if level and not result["exposure_hint"]:
                    result["exposure_hint"] = level
                    _logger.debug(f"[KB-Vector] clothing_intent命中: exposure_hint={level}")
                continue

            # 根据 item 是否携带 result_key 直接路由
            result_key = item.get("result_key") or self._CAT_TO_KEY.get(cat, "extra_tags")

            if result_key == "outfit_tags":
                result["outfit_tags"] = tags if isinstance(tags, str) else ", ".join(tags)
            elif result_key == "action_tags":
                result["action_tags"].extend(tags)
            elif result_key == "expression_tags":
                result["expression_tags"].extend(tags)
            elif result_key == "clothing_tags":
                result["clothing_tags"].extend(tags)
                if outfit_cat:
                    outfit_category_hint = outfit_cat
            else:
                result["extra_tags"].extend(tags)

            _logger.debug(f"[KB-Vector] score={score:.3f} cat={cat} tags={tags}")

        if outfit_category_hint and not result["outfit_tags"]:
            outfit = self.get_outfit_by_category(outfit_category_hint)
            if outfit:
                result["outfit_tags"] = outfit["tags"]

        # 去重
        for key in ("clothing_tags", "action_tags", "expression_tags", "extra_tags"):
            result[key] = list(dict.fromkeys(result[key]))
        return result

    # ------------------------------------------------------------------
    # 公开辅助接口
    # ------------------------------------------------------------------

    def get_outfit_by_category(self, category: str, name_hint: str = "") -> Optional[dict]:
        """按大类获取换装套装，支持名称二次过滤"""
        if not self._loaded:
            self.load()
        candidates = [o for o in self._outfits if o.get("category") == category]
        if not candidates:
            return None
        if name_hint:
            for o in candidates:
                if name_hint in o.get("name", ""):
                    return o
        import random
        return random.choice(candidates)

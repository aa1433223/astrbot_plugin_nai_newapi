# NovelAI 画图知识库

本知识库是供维护者和用户查阅的 NAI tag 资料，不参与普通出图流程。插件执行 `generate_image`
时不会根据自然语言自动检索、翻译、补全或增强提示词；只有用户明确提供或确认的内容才会提交。

## 文件结构

| 文件 | 内容 | 条数 |
|------|------|------|
| `intent_index.json` | 中文意图 → tag 映射（黑丝、校服、环抱等） | 42条 |
| `clothing_outfits.json` | 完整换装套装库（430套，含 tags 串） | 430套 |
| `clothing_style.json` | NAI 服装风格单 tag（从 nai_tags_direct.csv 提取） | 419条 |
| `action.json` | 动作/姿态 tag | 911条 |
| `expression.json` | 表情/情绪 tag | 58条 |
| `face.json` | 脸部/妆容 tag | 42条 |
| `body_hair.json` | 身体/发型 tag | 220条 |
| `art_style.json` | 画风/材质 tag | 61条 |
| `lighting.json` | 画面/光影 tag | 140条 |
| `composition.json` | 构图/天气 tag | 250条 |
| `background.json` | 背景/家具 tag | 131条 |
| `accessories.json` | 配饰/物品 tag | 682条 |
| `environment.json` | 环境/生物 tag | 586条 |
| `character_design.json` | 人设/物种 tag | 167条 |

## 查询流程（仅供人工或外部工具参考）

```
用户自然语言指令
    ↓
1. 将用户指令编码为向量
    ↓
2. 在 intent、套装库和分类 tag 条目中做语义检索
    ↓
3. 将命中结果归并为衣服、套装、动作、表情和附加 tag
```

## 数据来源

- `clothing_outfits.json`：来自 `clothing_database.txt`（430套）
- 其他分类 JSON：来自 `nai_tags_direct.csv`（NAI 官方 tag 数据库）
- `intent_index.json`：手工整理的中文意图映射

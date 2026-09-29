# 贡献指南

感谢你愿意为 `astrbot_plugin_novelai` 做贡献。提交前请阅读以下约定。

## 环境准备

- Python 3.10+
- 安装依赖：`pip install -r requirements.txt`
- 运行测试需要 `pytest`：`pip install pytest`

## 运行测试

测试通过插件运行时的包路径导入被测模块，请在包含 `data/plugins` 的 AstrBot 根目录下运行：

```bash
pytest tests
```

- 渲染 `画图帮助` / 画风库图片的用例需要本机可用的 Chromium；
- 其余用例不依赖浏览器，可在 CI 中直接运行。

## 代码约定

- 保持与现有代码一致的风格，改动尽量聚焦、最小化。
- 新增运行时文件请写入插件数据目录（`resolve_plugin_data_dir()`），不要写相对 `./data/`。
- 不要提交密钥、Token、本地运行数据（`data/plugin_data/`、`*.log` 等已被 `.gitignore` 忽略）。

## 提交 Pull Request

1. 从 `main` 切出分支，命名如 `fix/xxx` 或 `feat/xxx`。
2. 确保 `pytest tests` 通过。
3. 在 PR 描述中说明改动动机、影响范围与验证方式。

## 许可证

本项目以 GPL-3.0 授权（见 `LICENSE`）。提交代码即表示你同意以相同许可证分发你的贡献。

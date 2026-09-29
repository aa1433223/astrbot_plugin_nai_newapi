# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本。

## [1.5.2]

### 新增

- 支持“仅用 New API”部署：`api_backend=newapi` 时，普通出图直接使用 `newapi_base_url` + 全局 `newapi_api_key`，无需 NovelAI Token。
  - 免费/未绑定用户走全局 New API Key，并按每日额度计数；
  - 个人 Key / 群授权仍走各自的 Key，保持超额路由与 V4.5 权限回退语义；
  - 新增回归测试覆盖“无 NovelAI Token 也能出图”。

## [1.5.1]

### 变更

- 许可证由 MIT 改为 GPL-3.0，补充 `LICENSE` 文件（与内置的 GPL-3.0 `nai5-prompting` 保持一致）。
- 运行时状态（`usage_data.json`、`group_usage_data.json`、`last_draw_commands.json`）改写到 AstrBot 插件数据目录，不再依赖进程工作目录，并在升级时自动迁移旧文件。
- 统一测试导入方式为插件包路径导入，新增 `conftest.py` 与 `pyproject.toml`，便于在任意检出位置与 CI 中运行测试。
- 修复若干裸 `except:`，改为捕获 `Exception`。
- 修正 `metadata.yaml` 与注册信息中的仓库地址、作者与许可证字段。
- 清理误提交的运行时文件，完善 `.gitignore`。

## [1.5.0]

- 见此前的仓库历史。

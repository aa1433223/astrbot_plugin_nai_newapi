# AstrBot NovelAI 绘图插件

一个面向 AstrBot 的 NovelAI 绘图插件，支持文生图、图生图、智能重画、角色数据库管理与衣服库调用等功能。

## 功能概览

- 文生图：`/画图`（支持 NovelAI Diffusion V5 / V4.5）
- 图生图：`/图生图`（支持 NovelAI Diffusion V5 / V4.5）
- 智能重画：`/重画`
- 批量生成：`/roll图`
- 每日免费画图额度按群独立计算，每个群/会话每天 `group_daily_image_limit` 张（默认 5 张），固定使用 NovelAI Token；某群用完后，该群已在 ai_video 绑定密钥的用户才临时切到 New API 并使用自己的密钥和 V5，未绑定用户继续使用 NovelAI 的 `nai-diffusion-4-5-full`；如果个人 Key 没有 V5 模型权限，则自动回退到 NovelAI V4.5 Full
- 角色数据库搜索与管理命令
- 衣服编号与随机衣服系统
- 图片审核、自动撤回、自动保存
- 模型、画质参数与画风支持会话级选择
- 每个画风同时包含正面提示词、负面提示词和示例图
- `generate_image` 函数工具：仅接收用户提供或明确确认后的 Prompt、Character Prompt 和 UC，内部复用 `/画图` 的参数、预设、额度、审核、保存和重画流程
- 插件内置 GPL-3.0 授权的 [nai5-prompting](https://github.com/Miint-Sunny/nai5-prompting) Skill 仅在用户主动要求编写或优化提示词时使用
- 用户手写的权重、Randomizer、`|` 多角色结构和其它 NovelAI 语法会保留
- AstrBot 内嵌预设管理与画风库 WebUI

## 项目结构

```text
.
├── main.py
├── metadata.yaml
├── _conf_schema.json
├── requirements.txt
├── clothing_manager.py
├── knowledge_base_manager.py
├── novelai_payload.py
├── scene_prompt.py
├── preset_webui.py
├── pages/
│   └── preset-manager/
│       ├── index.html          # 单入口管理页
│       ├── presets/            # 提示词预设视图
│       └── style-gallery/      # 画风库视图
├── skills/
│   └── nai5-prompting/         # 插件内置的 NAI V5 提示词 Skill
├── scripts/update_character_database.py
├── clothing_database.txt
├── danbooru角色标签.csv
└── knowledge_base/
```

## 安装说明

1. 将本项目放入 AstrBot 插件目录。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

`/画图帮助`、`/画风列表` 和 `/查询画风` 使用本地 Playwright 截图后发送图片，属于可选能力：如果系统没有可用的 Chrome/Chromium，安装 `playwright` 后还需执行 `playwright install chromium`；缺少浏览器时相关命令会退回纯文本结果，不影响出图。

3. 在 AstrBot 中加载插件。
4. 将图像 API 后端保持为 `novelai`，填写承担每群每日免费额度的 `api_token`，并按需设置 `group_daily_image_limit`。
5. 填写 `newapi_base_url`，指向能够识别 ai_video 个人密钥的同一个 New API 服务；此功能不需要配置全局 `newapi_api_key`。
6. 如需在每日免费额度耗尽后自费继续使用 V5，用户可在 `astrbot_plugin_ai_video` 中执行 `/设置API密钥 sk-xxxx`；NovelAI 会直接复用该用户自己的绑定。

## 预设管理 WebUI

插件加载后，在 AstrBot 左侧“插件页面”中打开“NovelAI 绘图与预设”，通过顶部页签切换“提示词预设”和“画风库”。页面直接读写插件配置，修改后立即对聊天画图命令生效，无需单独部署网站或启动额外端口。

管理页提供待分类、角色、衣服、场景和动作资料夹；支持名称与提示词搜索、单条拖放、同分类排序、多选批量移动以及预设增删改。

画风库以图片卡片展示效果，可搜索、新建、编辑和删除画风。每个画风由示例图、正面提示词和负面提示词组成；选择画风后，正面词与负面词会自动进入各自的 NovelAI 请求字段。旧版字符串画风会自动迁移为新结构，原正面提示词保持不变。

## 基础配置

核心配置见 `_conf_schema.json`，常用项包括：

- `api_backend`：保持 `novelai`，每群每日免费额度和无个人密钥时的 V4.5 兜底都使用该后端
- `enable_v5`：V5 画图总开关；关闭后所有请求强制使用 V4.5，并暂停个人 Key 的 V5 超额路由
- `group_daily_image_limit`：每群每日免费图片额度，默认 `5`，各群独立计数；设为 `0` 时所有群直接启用超额路由
- `api_token`：NovelAI Token，承担每群每日免费额度及未绑定用户的 V4.5 兜底
- `account_tokens`：`/账户额度` 查询用的额外 NovelAI 账户列表，每条包含备注名、`pst-` 持久 Token 和启用开关；主账户 `api_token` 会自动一起展示，Token 相同的条目不会重复列出。该列表只参与额度查询，不参与画图请求
- `newapi_base_url`：个人密钥超额请求使用的 New API 根地址，默认 `http://localhost:3000`
- `newapi_api_key`：仅供旧的全局 New API 模式兼容使用；个人密钥超额路由不读取此项
- `default_model`：默认模型，V5 可选 `nai-diffusion-5-full` 或 `nai-diffusion-5-curated`
- `default_resolution`：默认分辨率
- `default_steps`：管理员统一设置的步数，用户不能通过聊天命令覆盖
- `default_scale`：默认 CFG，默认值 `6.5`
- `default_sampler`：默认采样器，默认值 `k_euler_ancestral`（Euler Ancestral），可在管理面板下拉选择
- `img2img_strength`：图生图强度
- `roll_image_count`：批量生成数量
- `enable_image_audit`：是否启用图片审核
- `enable_auto_save`：是否自动保存图片
- `enable_auto_recall`：是否自动撤回图片
- `enable_request_retry`：文生图/图生图请求失败（网络异常、认证失败、参数错误、服务器 5xx 等任意错误）时自动重试，默认开启
- `request_retry_count` / `request_retry_max_delay`：单次请求失败后的最大重试次数（默认 `5`，不含首次请求，设为 `0` 表示失败即返回）与两次重试之间的最长等待秒数（默认 `10`）；实际等待按指数退避计算并叠加随机抖动

## 常用命令

### 自然语言函数工具

插件会注册 `generate_image` 函数工具。工具只接受用户提供或明确确认后的 Prompt、Character Prompt 和 UC；它不会替用户构思画面、翻译自然语言、补全标签、识别角色名或增强提示词。没有明确提示词时，工具不会在插件内部自动生成。

函数工具不会建立第二套出图配置：模型、分辨率、步数、CFG、采样器、画风、额度路由、审核、自动保存及重画记录都沿用当前会话的 `/画图` 流程。原来的命令用法不变。

### 绘图命令

- `/画图帮助`：以图片查看完整的绘图命令、参数、预设/画风和提示词规则
- `/画图 [长图|宽图|方图] 提示词`
- `/图生图 [长图|宽图|方图] 提示词`
- `/重画 附加提示词`
- `/roll图 [长图|宽图|方图] 提示词`
- `/画图统计`：查看近30天用户排行和今日画图排行
- `/账户额度`（别名 `/账号额度`、`/额度查询`）：查询 `api_token` 与 `account_tokens` 中每个账户的剩余额度（含可生成张数估算）、Anlas 余额、恢复速度和订阅到期时间（含剩余天数、宽限期）；需要管理员权限时把命令名加入 `admin_only_commands`
- `/开启NAI画图`：由当前群内已绑定个人 API 密钥的用户执行，将自己的密钥授权给当前群
- `/关闭NAI画图`：关闭当前群的 NAI 画图授权

NAI 群授权不是 AstrBot 管理员权限，也不是插件白名单：命令必须在群聊中由已绑定个人
API 密钥的用户执行。开启后，当前群内没有个人密钥的其他成员也可以使用授权人的密钥
进行 `/画图`、`/图生图`、`/重画` 和 `/roll图`；请求不会把密钥复制到其他用户账户，
也不会影响任何人的个人密钥绑定。授权只记录当前群与授权人，其他群不能使用；授权人的
密钥被解绑后，群授权会自动失效，重新绑定后恢复。关闭授权不会删除或修改个人密钥。

命令中的画幅预设只对本次生成生效：长图为 `832x1216`、宽图为 `1216x832`、方图为 `1024x1024`。例如 `/画图 宽图 海边日落`。

未使用群授权时，每个群每日 `group_daily_image_limit` 张成功出图使用共享 NovelAI Token，并沿用
会话选择的模型；该群额度耗尽后，群内有个 API Key 的用户使用自己的密钥，没有密钥的用户继续
使用 NovelAI Token 和 `nai-diffusion-4-5-full` 兜底，其他群不受影响。开启 NAI 群授权后，当前群内
没有个人密钥的成员会直接使用授权人的密钥，不占用本群免费额度；个人/群授权请求切换到
`newapi_base_url` 并使用 V5（会话选择 V4.5 时自动切到 `nai-diffusion-5-full`）。如果 New API
明确返回个人 Key 无权访问 V5 的 403，插件会按 V4.5 规范化提示词并使用 NovelAI Token 自动
重试一次。次日 00:00 按服务器本地时间自动恢复各群的 NovelAI 免费额度路由。

临时关闭 `enable_v5` 后，默认设置、会话覆盖和个人 Key 超额路由中的 V5 都会被禁用，所有请求统一使用 V4.5；该开关不会按日期自动恢复，需要在限制结束后手动重新开启。

### 账户额度查询

`/账户额度` 会并发查询 `api_token`（显示为“主账户”）与配置项 `account_tokens` 中每个启用账户，逐条输出
剩余额度、Anlas 余额、恢复速度和订阅到期时间；单个账户查询失败不会影响其他账户。

输出字段的含义：

- 剩余额度：`subscription.usage.percent`，即 Opus 生成额度的剩余百分比。Opus 订阅在标准分辨率与
  28 步以内生成 NovelAI Diffusion V5（以及 V4 Full）不消耗 Anlas，用掉的就是这份额度；额度耗尽后
  出图才开始扣 Anlas，因此它不等于 Anlas 余额。官网前端按 `round(17.3 × percent)` 估算可生成张数，
  插件沿用同一系数显示“约 N 张”。
- 每天恢复：`86400 ÷ subscription.usage.timeUntilNextPercent`，与官网“Currently refills at [0]% per day”
  一致；`isNegative` 为真时表示已超额，出图直接扣 Anlas。
- Anlas：`subscription.trainingStepsLeft`。官网购买页把这个字段的
  `fixedTrainingStepsLeft` 标为 “Your Subscription Anlas”、`purchasedTrainingSteps` 标为
  “Your Paid Anlas”，命令显示两者合计与明细。绝对 Anlas 余额的另一个来源是计费接口
  `/user/cb/get-customer`，但该接口明确拒绝持久 Token（`401 Usage of persistent access tokens is not allowed
  for this endpoint`），因此 `trainingStepsLeft` 是用 `pst-` Token 唯一能读到的 Anlas 余额。
- 到期：`subscription.expiresAt`，在订阅未激活时附带“（未激活）”，宽限期会单独标注。

该命令调用 `https://image.novelai.net/user/data`，因为 NovelAI 的持久 Token（`pst-` 开头）在
`api.novelai.net` 上会被拒绝（这些路径在 `image.novelai.net` / `text.novelai.net` 上同样可用）；请求沿用
画图请求的网络代理（`novelai_proxy`）与 AstrBot 的全局代理设置，并按画图请求的重试配置重试（最多 3 次尝试，
401/403 等 4xx 不重试）。`account_tokens` 中的 Token 不会用于画图，仅用于此命令。

### 会话级参数命令

管理面板中的绘图参数是默认值。下列命令只覆盖当前群聊或私聊会话，
不会修改管理面板配置，也不会影响其他会话；插件重载后会话覆盖自动清空：

画图步数例外：`default_steps` 由管理员统一设置，用户不能调整。

- `/画图模型 <v5完整|v5精选|v4.5完整|v4.5精选>`
- `/画图权重 <1-15>`
- `/画图采样器 <简称或采样器ID>`
- `/分辨率 <长图|宽图|方图>`
- `/缩放 <0-1>`：设置 CFG Rescale
- `/图生图强度 <0.1-0.9>`
- `/图生图噪声 <0-1>`
- `/roll次数 <1-50>`
- `/切换画风 <名称>`、`/启用画风`、`/关闭画风`
- `/画风列表 [关键词]`：以示例图卡片查看画风，可按名称筛选
- `/查询画风 <名称>`：查看单个画风名称与示例图
- `/参数设置` 或 `/画图配置`：查看当前会话实际生效值
- `/重置画图参数`：清除当前会话覆盖并恢复管理面板默认值

提示词中的 `-s`、`-n` 仍是单次图生图参数，优先于当前会话的图生图设置。
在提示词中写入已有画风名，可仅对本次生成生效；该画风的正面词与负面词会同时应用。
在提示词末尾使用 `--uc <负面提示词>`，可为本次生成完全覆盖画风负面提示词，例如 `/画图 1girl, smile --uc lowres, blurry, bad hands`。
提示词中的长图、宽图或方图同样是单次画幅参数，优先于当前会话的分辨率设置。

### 审核白名单命令

以下命令仅 AstrBot 管理员可用，添加后会立即生效并写回插件配置：

- `/审核白名单`：查看当前审核白名单
- `/添加审核白名单 [群号]`：添加审核豁免群，不填时使用当前群
- `/添加Bot白名单 [Bot ID]`：添加需要审核的 Bot，不填时使用当前 Bot
- `/移除审核白名单 [群号]`：移除审核豁免群
- `/移除Bot白名单 [Bot ID]`：移除 Bot 白名单

Bot 白名单采用“仅名单内审核”机制：只有列表内 Bot 生成的图片会进入图片审核；
列表为空时不会审核任何 Bot。群白名单仍表示审核豁免，群白名单内的图片会跳过审核。

### 典型用法

```text
/画图 椿, 衣服131, 美少女
/图生图 1girl, smile
/重画 动漫
/roll图 白发, 蓝眼, 制服
/画图 美少女, 清新淡雅, 动漫通用
```

## 特色能力

### 1. 提示词处理

普通出图不会调用 LLM，也不会执行自动提示词增强。插件会把用户提供的提示词交给 NovelAI，
仅做必要的本地格式校验和 API 结构整理：

- 不翻译或扩写自然语言，不自动构思场景
- 不自动识别角色名，不把 `hsin` 替换成 `hsin_(wuthering_waves)`
- 不自动添加角色、画师、质量词、风格、动作、场景细节或 UC
- 用户手写的权重、Randomizer、`|` 多角色结构会保留
- 用户明确输入的预设、画风、衣服编号等本地功能仍可按原有逻辑处理
- `/搜索`、`/添加角色` 等命令仍可独立使用角色数据库，但数据库不会参与普通出图的角色识别

`generate_image` 只接收用户提供或明确确认后的 Prompt、独立 Character Prompt 和 UC。
`character_prompts` 没有明确内容时应传空数组；用户只要求出图但没有提供明确提示词时，
插件不会自行生成增强提示词。

V5 多角色可以手动使用紧凑语法：

```text
/画图 1girl, 1boy, train station, side-by-side | girl, red hair, white coat | boy, black hair, navy jacket
```

单个 `|` 分隔 Base Prompt 与各角色；Prompt Randomizer 的
`||red|blue||` 会被保留，不会错误拆成角色。V5 请求会保留角色书写顺序；没有明确
位置时使用模型的默认排布，不强制开启自定义坐标。本插件按官方已经展示验证的规模，将 V5 限制为 22 个角色段；
切换到 V4.5 时限制为 6 个。

### 2. 角色数据库

- 内置 Danbooru 角色标签数据
- 支持中文名、英文名、作品名检索
- 支持随机角色与重名处理

维护者可从 Danbooru 官方 Tags API 增量更新角色库。该操作只追加新标签，保留已有中文名、作品名和别名：

```bash
python scripts/update_character_database.py --dry-run
python scripts/update_character_database.py
```

默认同步非废弃、使用数不少于 25 的角色标签；可用 `--min-post-count` 调整阈值。同步后重载插件即可让 `/搜索` 使用新数据。

### 3. 衣服系统

- 支持 `衣服131` 这类编号调用
- 支持随机衣服
- 适合快速拼装提示词

## 开发与测试

本插件的测试通过插件运行时的包路径导入被测模块，因此需要从包含 `data/plugins`
的目录运行：

```bash
pip install -r requirements.txt pytest
pytest tests
```

渲染 `画图帮助`、`画风库` 图片的用例需要本机可用的 Chromium；其余用例不需要浏览器。

## 贡献

欢迎提交 Issue 和 Pull Request。请先阅读 `CONTRIBUTING.md`，并确保改动通过
`pytest tests`。

## 许可证

本项目采用 [GPL-3.0](LICENSE)（GNU General Public License v3.0 或更高版本）。

内置的 `skills/nai5-prompting` 来自
[Miint-Sunny/nai5-prompting](https://github.com/Miint-Sunny/nai5-prompting)，
同样以 GPL-3.0 授权，来源信息见 `skills/nai5-prompting/UPSTREAM.md`。

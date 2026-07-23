# AI Horde 图像生成工作流

> 文字驱动 → LoRA 自动匹配 → 提示词生成 → AI Horde 分布式生图 → 自动下载

一个基于 [AI Horde](https://stablehorde.net/) 分布式 GPU 网络的智能图像生成工作流。接收任意中文描述，自动提取场景、匹配 LoRA、生成专业提示词、提交远程生成任务并下载结果。

## 项目结构

```
├── workflow.py              # 主编排器（CLI 入口 + Python API）
├── prompt_builder.py        # 提示词构建（场景提取 + Prompt/Negative 生成）
├── lora_manager.py          # LoRA 白名单管理 + 关键词匹配
├── ai_horde_client.py       # AI Horde API（提交、轮询、下载）
├── civitai_client.py        # CivitAI API（LoRA 元数据获取，可选）
├── llm_client.py            # LLM 抽象层（OpenAI 兼容接口）
├── config.yaml              # 用户配置文件（API Key、模型、LoRA 白名单）
├── .env.example             # 环境变量模板（复制为 .env 填入真实 Key）
├── prompt/
│   └── prompt_template.txt  # 专业提示词工程师模板
├── output/                  # 生成图片输出目录
├── reference/               # 参考文档（需求、开发坑点、LoRA 收藏）
└── docs/
    └── WORKFLOW.md          # 工作流详细文档 + 踩坑记录
```

## 快速开始

### 1. 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入真实 API Key
# AI_HORDE_API_KEY=你的key    （必填，在 https://aihorde.net/register 注册）
# OPENAI_API_KEY=sk-xxx       （必填，或兼容的 LLM API Key）

# 编辑 config.yaml，根据需求调整模型和参数
# 关键配置项: defaults.model, defaults.width/height, lora_whitelist
```

### 2. 运行

```bash
# 命令行模式
python workflow.py "一个少女站在樱花树下，微风拂过她的长发"

# 试运行模式（只生成提示词，不调 AI Horde）
python workflow.py "..." --dry-run
```

### 3. Python API

```python
from workflow import Workflow

wf = Workflow("config.yaml")

# 完整生成
result = wf.run("一个白发少女在月光下弹钢琴")
print(result["images"])       # ["output/20250723_120000_xxx.png"]
print(result["prompt_data"])  # 完整生成参数

# 仅生成提示词
result = wf.run("...", dry_run=True)

# 查看可用模型
print(wf.list_available_models())

# 查看已配置 LoRA
print(wf.list_loras())
```

## 工作流概述

```
用户文本 → [场景提取] → [LoRA匹配] → [提示词生成] → [AI Horde生成+下载]
   │           │              │              │                │
   │      LLM提取/编造    白名单关键词    专业提示词模板     异步提交
   │      最后场景         匹配           Prompt+Negative   10s轮询
   │                                   + 参数 JSON          自动下载
```

详见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

## 依赖

- Python 3.10+
- `pyyaml` — YAML 配置解析
- `python-dotenv` — 自动加载 `.env` 环境变量

无其他外部依赖（HTTP 请求全部使用标准库 `urllib`）。

## 配置 LoRA 白名单

在 `config.yaml` 的 `lora_whitelist` 中添加：

```yaml
lora_whitelist:
  - name: "你的 LoRA 名称"
    model_id: 12345            # CivitAI model_id（可选，用于获取描述）
    version_id: "67890"        # AI Horde 使用的版本 ID（必填，字符串）
    keywords: ["角色名", "character"]  # 用于匹配的中/英文关键词
    trigger_words: ["trigger"]         # 触发词
    base_model: "Illustrious"          # 基础模型类型
    strength_model: 0.8
    strength_clip: 0.8
```

> ⚠️ `version_id` 必须是 CivitAI 的 **版本 ID**（模型页面 URL 中的数字），不是模型 ID。这是发送给 AI Horde 的关键参数。

## 日志

日志同时输出到控制台和 `workflow.log` 文件。关键操作（请求提交、状态轮询、下载完成）均有记录，包括完整的 AI Horde 请求体。

## 踩坑记录

本项目经过实际调试，修复了以下问题（详见 `docs/WORKFLOW.md`）：

- Cloudflare 拦截 Python 默认 User-Agent → 添加 UA 头
- `/generate/check` 不返回图片 URL → 改用 `/generate/status`
- 模型名需与 AI Horde 精确匹配 → 通过 API 查询确认
- LLM 漏字段 → JSON 格式约束 + 降级默认值
- `.env` 手动加载繁琐 → `python-dotenv` 自动加载

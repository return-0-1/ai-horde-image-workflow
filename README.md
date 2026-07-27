# AI Horde 图像生成工作流

> 文字驱动 → LoRA 自动匹配 → 提示词生成 → AI Horde 分布式生图 → 自动下载

一个基于 [AI Horde](https://stablehorde.net/) 分布式 GPU 网络的智能图像生成工作流。接收任意中文描述，自动提取场景、匹配 LoRA、生成专业提示词、提交远程生成任务并下载结果。

## 项目结构

```
├── workflow.py              # 主编排器（CLI 入口 + Python API）
├── batch_runner.py          # 批量生图（TXT 文件解析 + 逐条编排）
├── prompt_builder.py        # 提示词构建（场景提取 + Prompt/Negative 生成）
├── params_parser.py         # 参数解析（从口语指令中提取分辨率/步数等）
├── lora_manager.py          # LoRA 白名单管理 + 关键词匹配
├── lora_searcher.py         # LoRA 自动搜索（CivitAI 并行搜索 + LLM 审核）
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
# 命令行模式（单条）
python workflow.py "一个少女站在樱花树下，微风拂过她的长发"

# 批量模式（从 TXT 文件读取多条场景）
python workflow.py --file prompts.txt

# 批量预览（仅生成提示词，不调 AI Horde）
python workflow.py --file prompts.txt --dry-run

# 批量 + 遇错即停
python workflow.py --file prompts.txt --stop-on-error

# 试运行模式（只生成提示词，不调 AI Horde）
python workflow.py "..." --dry-run

# 查询可用模型 / LoRA
python workflow.py --list-models
python workflow.py --list-loras
```

### 3. 批量文件格式

TXT 文件每行为一个场景描述，**连续的非空行自动合并为一个场景**（用空格连接），空行作为场景分隔符。

如需场景内包含空行（如多段落描述），用 `---` 包裹场景块，块内原样保留换行和特殊字符。

```txt
# ===== 注释行（# 开头，跳过）=====

# 单行场景
一个少女在樱花树下微笑

# 多行场景：连续多行会合并成一个场景
教室靠窗的座位，黑长直少女单手托腮望向窗外，
白色校服短裙，阳光从窗户斜射，忧郁氛围，
仰视构图，柔和的暖色调光影  // 行内注释

# 参数覆盖（-- 开头，作用于下一条场景）
--steps=30 --width=1024 --height=1024
银发女剑士双手持大剑，战斗姿势半蹲，破损披风飘动

# --- 场景块：块内空行保留，不做注释/参数解析 ---
---
一个少女站在樱花树下。

微风拂过她的长发，花瓣飘落。

她伸手接住一片花瓣，微笑。
---

# 恢复默认参数
--reset
回到默认配置的普通场景
```

> **规则速查：** `#`=注释 | 空行=场景分隔 | `---`=场景块标记（块内原样保留） | `--key=value`=覆盖参数 | `--reset`=恢复默认 | `//`=行内注释 | 连续行=合并为同一场景

### 4. Python API

```python
from workflow import Workflow
from batch_runner import BatchRunner

wf = Workflow("config.yaml")

# 单条生成
result = wf.run("一个白发少女在月光下弹钢琴")
print(result["images"])       # ["output/20250723_120000_xxx.png"]
print(result["prompt_data"])  # 完整生成参数

# 仅生成提示词
result = wf.run("...", dry_run=True)

# 批量生成（从 TXT 文件）
runner = BatchRunner(wf)
results = runner.run_batch("prompts.txt")
runner.print_summary(results)

# 查看可用模型 / LoRA
print(wf.list_available_models())
print(wf.list_loras())
```

## 工作流概述

```
用户文本 → [场景提取] → [LoRA获取] → [提示词生成] → [AI Horde生成+下载]
   │           │              │              │                │
   │      LLM提取/编造    白名单匹配       专业提示词模板     异步提交
   │      最后场景         ↓ 无匹配         Prompt+Negative   10s轮询
   │                 CivitAI并行搜索       + 参数 JSON        自动下载
   │                 ↓ LLM审核+优先级
   │                自动LoRA匹配
```

详见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

## 依赖

- Python 3.10+
- `pyyaml` — YAML 配置解析
- `python-dotenv` — 自动加载 `.env` 环境变量

无其他外部依赖（HTTP 请求全部使用标准库 `urllib`）。

## 配置 LoRA

### 白名单（手动配置）

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

### 自动搜索（白名单无匹配时触发）

`lora_searcher` 自动从 CivitAI 搜索 LoRA，经 LLM 审核后使用：

```yaml
lora_search:
  max_results: 10             # 每个关键词搜索结果数
  max_loras: 2                # 最终提交 LoRA 数量上限
```

工作流程：场景 → 提取 3-5 个关键词 → 并行搜索 → 合并去重 → LLM 审核兼容性 → 按优先级截断。

### 黑名单（排除不可用的 LoRA）

```yaml
lora_blacklist:
  - version_id: "133229"
    model_id: "122355"
    reason: "SD1.5 LoRA，与主力 SDXL 模型不兼容"
```

> ⚠️ `version_id` 必须是 CivitAI 的 **版本 ID**（模型页面 URL 中的数字），不是模型 ID。

## 日志

日志同时输出到控制台和 `workflow.log` 文件。关键操作（请求提交、状态轮询、下载完成）均有记录，包括完整的 AI Horde 请求体。

## 踩坑记录

本项目经过实际调试，修复了以下问题（详见 `docs/WORKFLOW.md`）：

- Cloudflare 拦截 Python 默认 User-Agent → 添加 UA 头
- `/generate/check` 不返回图片 URL → 改用 `/generate/status`
- 模型名需与 AI Horde 精确匹配 → 通过 API 查询确认
- LLM 漏字段 → JSON 格式约束 + 降级默认值
- `.env` 手动加载繁琐 → `python-dotenv` 自动加载
- LoRA 提交格式错误（字符串 vs 对象） → `is_version` + 对象格式
- LLM JSON 解析失败（尾逗号/围栏） → 鲁棒解析 `_parse_llm_json`
- Windows subprocess GBK 编码崩溃 → `encoding="utf-8"`
- CivitAI 搜索无 NSFW 结果 → `nsfw=true` 参数
- 搜索关键词过长导致零结果 → 拆分为并行短查询
- Docker 容器相对路径漂移 → 输出目录改为绝对路径 `/AstrBot/data/output`

## AstrBot 插件命令

配套插件 `astrbot_plugin_ai_horde_image` 提供以下 QQ 机器人命令：

| 命令 | 功能 |
|------|------|
| `/draw <描述>` | 生成图片（支持口语化参数如分辨率、采样器等） |
| `/生图 <描述>` | 同上（中文别名） |
| `/排队` | 查询 AI Horde 全局排队状态 + 本用户活跃任务 |
| `/发图` | 手动发送待发的生成图片 |

### `/排队` 输出示例

```
🔍 **AI Horde 状态**
📊 全局: 排队 273 请求 | 12 workers (12 线程)
   吞吐: 997 MP/分钟
🎯 WAI-NSFW-illustrious-SDXL: 7 workers | ETA ~289s | 排队 4745M px

👤 你的任务 (kudos: 370520)
  🖼️ 4919b030...: 队列 #3 | 预计 45s | ⏳ 处理中 (处理1/等待0)
```

任务追踪覆盖两个阶段：
- **Pre-submit**：提示词工程 / LoRA 搜索阶段 → 显示"⏳ 进行中"
- **AI Horde 队列**：已提交到 AI Horde → 显示队列位置、预计时间、处理状态

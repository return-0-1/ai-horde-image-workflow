# 工作流详细文档

## 完整执行流程

### 阶段 1: 场景提取 (`prompt_builder.extract_scene()`)

**输入**: 用户任意中文文本（小说片段、口述描述、角色介绍等）

**处理**:
1. 将专业提示词模板 + 场景提取指令发送给 LLM
2. LLM 分析文本：
   - 多场景 → 提取**最后一个场景**
   - 无场景 → 自行编造合理场景
3. 返回结构化中文场景描述（含角色、动作、环境、光影、构图）

**输出**:
```json
{"scene": "中文场景描述...", "is_fabricated": false}
```

**LLM 调用的系统提示词**（`_build_scene_extraction_system_prompt()`）：
- 要求输出 JSON 格式
- 场景需包含 5 大维度：角色外貌、服饰、光影、构图、背景
- 缺失信息用合理默认值补充

---

### 阶段 2: LoRA 获取（`lora_manager.match()` + `lora_searcher.search()`）

**输入**: 场景文本 + 基础模型名称

**处理（Phase 2a — 白名单优先）**:
1. 遍历 `config.yaml` 中 `lora_whitelist` 的所有条目
2. 关键词匹配：场景文本中出现 `keywords` 中的词 → 得分 +1
3. 名称命中 → 额外 +2
4. 按得分降序返回 top_k 个结果

**处理（Phase 2b — 自动搜索，白名单无匹配时触发）**:
1. LLM 从场景提取 3-5 个关键概念，按重要性排序
2. 多线程并行搜索 CivitAI（`nsfw=true`，走 SOCKS5 代理）
3. 合并去重，按 base model 预筛（SDXL/Illustrious/Pony 兼容）
4. 跳过黑名单中的 model_id
5. 获取每个 version 的完整元数据
6. LLM 审核：过滤不兼容的 base model + 不匹配场景的 LoRA
7. 按优先级排序，截断到 `max_loras` 上限

---

### 阶段 3: 提示词生成 (`prompt_builder.build_prompt()`)

**输入**: 场景描述 + LoRA 列表

**处理**:
1. 加载专业提示词模板（`prompt/prompt_template.txt`）
2. 注入 LoRA 触发词和描述信息
3. 追加严格的 JSON 输出格式约束
4. 发送给 LLM 生成最终提示词

**输出**（LLM 返回 JSON）:
```json
{
  "prompt": "masterpiece, best quality, ... 英文正向提示词",
  "negative": "worst quality, low quality, ... 英文负面提示词",
  "chinese_note": "中文说明..."
}
```

**降级策略** (`_ensure_fields()`):
- `negative` 为空 → 使用内置默认负面提示词
- `chinese_note` 为空 → 根据场景自动生成简略说明

**参数合并** (`_merge_defaults()`):
- LLM 返回的字段 > `config.yaml` 的 `defaults` > 代码内置默认值

---

### 阶段 4: AI Horde 生成 (`ai_horde_client`)

#### 4.1 提交 (`submit()`)

构建完整请求体 → `POST /api/v2/generate/async`

**关键参数处理**:
- `loras[].name` = 字符串化的 `version_id`（不是 model_id）
- 过滤 `None` 值（否则 API 400）
- 不传 `control_type`（非 ControlNet 场景不需要）
- 记录完整请求体到日志

#### 4.2 轮询 (`_poll()`)

`GET /api/v2/generate/status/{id}` → 每 10s 轮询 → 直到 `done=True`

> ⚠️ 使用 `/generate/status` 而非 `/generate/check`：后者不返回 `generations` 字段。

**状态判断**（双重检查）:
```python
done = status.get("done", False)
state = status.get("state", "")
if done or state == "done":  # 兼容两种返回格式
    return status
```

#### 4.3 下载 (`_download()`)

从 `generations[].img` 获取图片 URL → 下载到 `output/` 目录

> ⚠️ 方法末尾必须显式 `return images`（Python 默认返回 None）。

**文件命名**: `{timestamp}_{model}_{seed}.png`

---

## 配置文件说明

### config.yaml

```yaml
ai_horde:
  api_url: "https://aihorde.net/api/v2"
  api_key: ""           # 或环境变量 AI_HORDE_API_KEY
  poll_interval: 10     # 轮询间隔（秒）
  max_wait: 300         # 最大等待（秒）

llm:
  provider: "openai"
  api_base: "https://api.openai.com/v1"  # 也支持 deepseek/openrouter 等
  api_key: ""           # 或环境变量 OPENAI_API_KEY
  model: "gpt-4o"      # 实际使用的模型名

defaults:
  model: "WAI-NSFW-illustrious-SDXL"  # 必须与 AI Horde 模型名完全一致
  width: 512
  height: 512
  steps: 25
  cfg_scale: 7.5
  sampler: "k_euler_a"
  nsfw: true

lora_whitelist: []      # LoRA 白名单
```

### .env

```bash
AI_HORDE_API_KEY=xxx    # AI Horde API Key
OPENAI_API_KEY=sk-xxx   # LLM API Key
```

加载顺序：代码自动调用 `load_dotenv()` → config.yaml 覆盖 → 代码默认值

---

## 模块职责分离

| 模块 | 职责 | 依赖 |
|------|------|------|
| `workflow.py` | 编排 4 阶段流程，日志初始化 | 所有模块 |
| `prompt_builder.py` | LLM 交互、提示词生成、鲁棒 JSON 解析 | `llm_client` |
| `lora_manager.py` | LoRA 白名单 CRUD、关键词匹配 | 无 |
| `lora_searcher.py` | CivitAI 并行搜索、LLM 审核、优先级排序 | `llm_client`, `civitai_client` |
| `ai_horde_client.py` | HTTP 请求、轮询、下载 | 无 |
| `civitai_client.py` | CivitAI API 调用、缓存 | 无 |
| `llm_client.py` | OpenAI 兼容 API 调用 | 无 |

新增采样器、模型预设或 LoRA 时，只需修改 `config.yaml`，无需改动核心代码。

---

## 踩坑记录

### 1. Cloudflare 拦截 → HTTP 403

**现象**: `urllib` 请求 `aihorde.net` 返回 403，`curl` 正常

**根因**: Cloudflare WAF 拦截 Python `urllib` 默认 User-Agent

**修复**: 所有 HTTP 请求添加 `User-Agent: AI-Horde-Workflow/1.0`

### 2. /generate/check 无 generations

**现象**: 任务完成 (`done=True`) 但 `generations` 为空

**根因**: `/api/v2/generate/check/{id}` 只返回状态摘要

**修复**: 改用 `/api/v2/generate/status/{id}` 轮询

### 3. 模型名不匹配

**现象**: `wai-nsfw-illustrious` 无法提交

**根因**: AI Horde 上精确名称为 `WAI-NSFW-illustrious-SDXL`

**修复**: 通过 `GET /api/v2/status/models` 查询确认

### 4. LLM 漏字段

**现象**: DeepSeek 只返回 `prompt`，缺 `negative`/`chinese_note`

**根因**: LLM 忽略了 JSON schema 中的其他字段

**修复**: 
- 系统提示词末尾追加 `⚠️ 最终输出格式要求（必须严格遵守）`
- `_ensure_fields()` 降级填充默认值

### 5. .env 手动加载繁琐

**修复**: `workflow.py` 启动时自动 `load_dotenv(".env")`

### 6. CivitAI API 精确搜索困难

`/api/v1/models?query=` 是**模糊搜索**，按热门度排序。没有 model_id 时可能匹配到错误 LoRA。最佳实践：手动获取 model_id 后批量 API 查询。

---

## AI Horde API 参考

### 关键端点

| 端点 | 用途 |
|------|------|
| `GET /api/v2/status/models?type=image` | 查询可用图像模型 |
| `POST /api/v2/generate/async` | 提交异步生成任务 |
| `GET /api/v2/generate/status/{id}` | 查询任务状态（含 generations） |
| `GET /api/v2/generate/check/{id}` | 查询任务状态（仅摘要，无图片） |

### LoRA 参数格式

```json
{
  "loras": [{
    "name": "67890",            // version_id（字符串）
    "model": 0.8,               // 模型强度
    "clip": 0.8,                // CLIP 强度
    "is_version": true,         // 标识为版本 ID
    "inject_trigger": "any"     // 或具体触发词，不能为 null
  }]
}
```

> ⚠️ `name` 必须是版本 ID（version_id），不是模型 ID（model_id）。传错会导致 Download Failed。
> ⚠️ `is_version: true` 告诉 worker 这是一个 version ID，减少匹配歧义。`inject_trigger` 我们不设，因为 prompt_builder 已手动注入触发词。

### status 端点 vs check 端点

| 字段 | `/generate/status/{id}` | `/generate/check/{id}` |
|------|:---:|:---:|
| `done` | ✅ | ✅ |
| `faulted` | ✅ | ✅ |
| `queue_position` | ✅ | ✅ |
| `generations` | ✅ | ❌ |
| `shared` | ✅ | ❌ |

**结论**: 始终用 `/generate/status/{id}` 做轮询。

### 新增踩坑（V0.2）

| 问题 | 现象 | 修复 |
|------|------|------|
| LoRA 提交格式 | `loras: ["2638973"]` → 400 | 改为对象 `{name, model, clip, is_version}` |
| LLM JSON 尾逗号 | `json.loads` 失败 | `_repair_json` 去尾逗号，4 策略解析 |
| GBK 编码崩溃 | subprocess `UnicodeDecodeError` | `encoding="utf-8", errors="replace"` |
| CivitAI 搜索无 NSFW | 全返回 SFW LoRA | 搜索加 `nsfw=true` |
| 长查询→0 结果 | 10 词查询无匹配 | 拆为 3-5 个短查询并行搜索 |
| CivitAI 503/超时 | 搜索间歇失败 | 3 次重试 + 缓存 |
| AI Horde 无 LoRA 状态 | 任务完成不知 LoRA 是否加载 | 目前无解，只能看效果 |
| CivitAI `baseModel` 过滤无效 | API 忽略该参数 | Python 预筛 `_base_compatible` |

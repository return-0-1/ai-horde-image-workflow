# 工作流详细文档

## 完整执行流程

### 阶段 1: LoRA 匹配 (`lora_manager.match()` + `lora_searcher.search()`)

**输入**: 用户原始文本（不经过 LLM 场景提取，节省一次 LLM 调用）

**处理（Phase 1a — 白名单优先）**:
1. 遍历 `config.yaml` 中 `lora_whitelist` 的所有条目
2. 关键词匹配：用户文本中出现 `keywords` 中的词 → 得分 +1
3. 名称命中 → 额外 +2
4. 按得分降序返回 top_k 个结果

**处理（Phase 1b — 自动搜索，白名单无匹配时触发）**:
1. LLM 从场景提取 3-5 个关键概念，按重要性排序（含排除角色名指引）
2. 多线程并行搜索 CivitAI（`nsfw=true`，走 Cloudflare Pages 反代/urllib，无需 VPN）
3. 合并去重，按 base model 预筛（SDXL/Illustrious/Pony 兼容）
4. 跳过黑名单中的 model_id
5. 获取每个 version 的完整元数据
6. **文本相似度预排序**：用英文搜索词与 LoRA 名称做相似度排序，将语义相关候选推到前面
7. **LLM 评分审核**（1-10 分制）：前 15 个候选送 LLM 打分，≥ `min_score`(默认 3) 入选
8. 按评分降序，截断到 `max_loras` 上限

> v0.3 改进：review_limit 5→15，审核改为评分制（避免角色 LoRA 霸榜导致全拒），新增文本相似度预排序。

---

### 阶段 2: 场景提取 + 提示词生成（合并模式，`prompt_builder.build_prompt_from_user_text()`）

**输入**: 用户原始文本 + LoRA 列表

**处理**:
1. 构建合并系统提示词（场景分析指令 + 专业提示词模板 + LoRA 信息）
2. 一次 LLM 调用完成：场景提取 → prompt 生成
3. 鲁棒 JSON 解析（5 级回退）
4. 降级策略：negative 为空 → 二次 LLM 生成 → 内置默认值
5. 自动面部修复：检测人脸关键词 → facefixer_strength=0.6
6. 参数合并：LLM 返回 > config.yaml defaults > 代码内置默认值

**合并失败 → 自动回退两阶段模式**：extract_scene() → build_prompt()

> v0.3 改进：将原来的阶段 1（场景提取）+ 阶段 3（提示词生成）2 次 LLM 调用合并为 1 次，节省 ~9s 延迟。

---

### 阶段 2.5: 参数覆盖 (`params_parser.parse()`)

从用户口语指令中正则提取 → LLM 兜底 → merge_overrides() → 强制 model 使用配置默认值

---

### 阶段 3: AI Horde 生成 (`ai_horde_client`)

#### 3.1 提交 (`submit()`)

构建完整请求体 → `POST /api/v2/generate/async`

**关键参数处理**:
- `loras[].name` = 字符串化的 `version_id`（不是 model_id）
- 过滤 `None` 值（否则 API 400）
- 不传 `control_type`（非 ControlNet 场景不需要）
- 记录完整请求体到日志

#### 3.2 自适应轮询 (`_poll()`)

`GET /api/v2/generate/status/{id}` → 根据队列位置动态调整间隔：

| 队列状态 | 轮询间隔 |
|----------|----------|
| 正在处理 或 queue_pos ≤ 1 | 5s |
| queue_pos 2-5 | 8s |
| queue_pos 6-20 | 15s |
| queue_pos > 20 | 30s |

> v0.3 改进：从固定 10s 改为自适应间隔，减少远端排队时的无效请求。

#### 3.3 下载 (`_download()`)

从 `generations[].img` 获取图片 URL → 下载到 `output/` 目录

**文件命名**: `{timestamp}_{model}_{seed}.png`

---

## 配置文件说明

### config.yaml

```yaml
ai_horde:
  api_url: "https://aihorde.net/api/v2"
  api_key: ""           # 或环境变量 AI_HORDE_API_KEY
  poll_interval: 10     # 轮询间隔（秒），自适应轮询会自动调整
  max_wait: 300         # 最大等待（秒）

civitai:
  api_url: "https://civitai.com/api/v1"
  timeout: 20
  proxy_url: "https://civitai-proxy-ex1.pages.dev/api/v1"

llm:
  provider: "deepseek"
  api_base: "https://api.deepseek.com/v1"
  api_key: ""           # 或环境变量 OPENAI_API_KEY
  model: "deepseek-v4-pro"

defaults:
  model: "WAI-NSFW-illustrious-SDXL"
  width: 1024
  height: 1024
  steps: 25
  cfg_scale: 7.5
  sampler: "k_euler_a"
  nsfw: true

lora_search:
  max_results: 10       # 每个关键词搜索结果数
  max_loras: 2          # 最终提交 LoRA 数量上限
  min_score: 3          # LLM 审核最低通过分 (1-10)

lora_whitelist: []      # LoRA 白名单
```

---

## 模块职责分离

| 模块 | 职责 | 依赖 |
|------|------|------|
| `workflow.py` | 编排 3 阶段流程，日志初始化 | 所有模块 |
| `prompt_builder.py` | LLM 交互、场景提取+提示词生成（合并模式）、鲁棒 JSON 解析 | `llm_client` |
| `lora_manager.py` | LoRA 白名单 CRUD、关键词匹配 | 无 |
| `lora_searcher.py` | CivitAI 并行搜索、文本相似度预排序、LLM 评分审核 | `llm_client`, `civitai_client` |
| `ai_horde_client.py` | HTTP 请求、自适应轮询、下载 | 无 |
| `civitai_client.py` | CivitAI API 调用、缓存 | 无 |
| `llm_client.py` | OpenAI 兼容 API 调用 | 无 |

---

## V0.3 优化记录

| 优化 | 改动 | 效果 |
|------|------|------|
| 合并场景提取+提示词生成 | `prompt_builder.build_prompt_from_user_text()` | LLM 调用 4→3 次，节省 ~9s |
| 自适应轮询 | `ai_horde_client._poll()` 动态间隔 | 减少远端排队无效请求 |
| lora_searcher urllib 替代 curl | `_http_get()` 反代路径用 urllib | 消除子进程开销 |
| LoRA 审核评分制 | `_llm_review()` 1-10 分 + min_score | 避免角色LoRA全拒 |
| 文本相似度预排序 | `_rank_by_similarity()` 搜索词vs名称 | 相关LoRA推到前面 |
| review_limit 5→15 | `__init__` | 更多候选进入审核 |
| 搜索词排除指引 | `_extract_query()` prompt 优化 | 减少角色名噪声 |

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
**修复**: 系统提示词末尾追加 `⚠️ 最终输出格式要求` + `_ensure_fields()` 降级填充

### 5. .env 手动加载繁琐

**修复**: `workflow.py` 启动时自动 `load_dotenv(".env")`

### 6. CivitAI API 精确搜索困难

`/api/v1/models?query=` 是**模糊搜索**，按热门度排序。没有 model_id 时可能匹配到错误 LoRA。最佳实践：手动获取 model_id 后批量 API 查询。

### 7. CivitAI LoRA 搜索高拒绝率 ← v0.3 修复

**现象**: 搜索 29 个结果 → 前 5 全是角色特化 LoRA → LLM 全部拒绝 → 空手而归
**根因**: CivitAI 热门度排序 + review_limit=5 + 二元 pass/fail 审核
**修复**:
- review_limit 5→15
- 审核改为 1-10 评分制，≥min_score 即可
- 文本相似度预排序（英文搜索词 vs LoRA 名）
- 搜索词提取加入排除角色名指引

### 8. 中文场景 vs 英文 LoRA 名相似度为零 ← v0.3 修复

**现象**: `SequenceMatcher` 比较中文场景和英文 LoRA 名，全部返回 0
**修复**: 改用 LLM 提取的英文搜索词做相似度比较

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
> ⚠️ `is_version: true` 告诉 worker 这是一个 version ID，减少匹配歧义。

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
| CivitAI 国内被墙 | `workers.dev` 域名在国内被封 | 改用 Cloudflare Pages (`pages.dev`) + `_worker.js` 反代 |

"""
LoRA 搜索器 — 从 CivitAI 搜索 LoRA + LLM 审核适配性。

流程:
  场景文本 → 提取搜索关键词 → CivitAI API 搜索 (nsfw=true)
  → 获取每个 version 元数据 → 黑名单过滤 → 文本相似度预排序
  → LLM 评分审核 → 返回候选（必返回至少 0 个最优结果）

LLM 审核失败 → 返回空列表（调用方回退白名单）。
"""

import re
import json
import time
import logging
from typing import Optional
from difflib import SequenceMatcher
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# CivitAI 默认 API 地址（可通过 config.proxy_url 覆盖）
_CIVITAI_API = "https://civitai.com/api/v1"


class LoraSearcher:
    """从 CivitAI 搜索 LoRA + LLM 审核。"""

    def __init__(self, config: dict, llm_client, civitai_client):
        """
        Args:
            config:          完整配置 dict。
            llm_client:      LLMClient 实例（用于审核）。
            civitai_client:  CivitAIClient 实例（用于拉取元数据）。
        """
        self.llm = llm_client
        self.civitai = civitai_client
        self.search_limit = config.get("lora_search", {}).get("max_results", 10)
        self.review_limit = 15  # 最多送给 LLM 审核的候选数（从 5 提升）
        self.max_submit = config.get("lora_search", {}).get("max_loras", 2)  # 最终提交上限
        self.min_score = config.get("lora_search", {}).get("min_score", 3)  # 最低通过分 (1-10)
        self._cache: dict[str, list] = {}  # query → 缓存的搜索结果

        # CivitAI API 地址：优先使用代理 URL，否则直连
        civitai_cfg = config.get("civitai", {})
        proxy_url = civitai_cfg.get("proxy_url", "").strip()
        if proxy_url:
            self._api_base = proxy_url.rstrip("/")
            logger.info("CivitAI 使用代理: %s", self._api_base)
        else:
            self._api_base = _CIVITAI_API

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def search(self, scene_text: str,
               base_model: str = "SDXL",
               blacklist_model_ids: Optional[set] = None) -> list[dict]:
        """搜索并审核 LoRA。

        Args:
            scene_text:           场景描述文本。
            base_model:           目标 base model（用于兼容性检查）。
            blacklist_model_ids:  黑名单 model_id 集合（跳过这些）。

        Returns:
            审核通过的 LoRA 配置列表（格式与 config.yaml whitelist 一致）。
            LLM 审核失败返回空列表。
        """
        blacklist = blacklist_model_ids or set()

        # 1. 从场景文本提取搜索关键词（拆分为独立短查询，含排除指引）
        queries = self._extract_query(scene_text)
        logger.info("LoRA 搜索查询 (%d): %s", len(queries), queries)

        # 2. 并行搜索 CivitAI
        search_results = self._search_civitai(queries)
        if not search_results:
            logger.info("CivitAI 搜索无结果")
            return []

        # 3. 获取每个 version 的元数据 + 过滤黑名单 + 基础模型兼容性预筛
        candidates = []
        for item in search_results:
            model_id = str(item.get("id", ""))
            if model_id in blacklist:
                logger.debug("跳过黑名单 model_id=%s: %s", model_id, item.get("name", ""))
                continue

            # 取第一个 version
            versions = item.get("modelVersions", [])
            if not versions:
                continue
            v = versions[0]
            version_id = str(v.get("id", ""))
            if not version_id:
                continue

            # 获取完整 metadata
            try:
                meta = self._http_get(
                    f"{self._api_base}/model-versions/{version_id}",
                    use_proxy=True,
                )
            except Exception as e:
                logger.warning("元数据获取失败 v%s: %s", version_id, e)
                meta = v  # 降级：用搜索返回的简要信息

            lora_info = self._parse_candidate(item, meta, version_id)
            if lora_info and self._base_compatible(lora_info["base_model"], base_model):
                lora_info["_priority"] = item.get("_priority", 99)
                candidates.append(lora_info)

        if not candidates:
            logger.info("无可用候选（全部被过滤）")
            return []

        # 3.5 文本相似度预排序：用英文搜索词代替中文场景做比较
        candidates = self._rank_by_similarity(candidates, queries)

        # 限制候选数量（扩大到 15）
        candidates = candidates[:self.review_limit]
        logger.info("候选 LoRA (%d 个，相似度排序后): %s", len(candidates),
                    [(c["version_id"], c["name"][:35], round(c.get("_sim", 0), 2))
                     for c in candidates])

        # 4. LLM 评分审核
        try:
            scored = self._llm_review(scene_text, base_model, candidates)
        except Exception as e:
            logger.warning("LLM 审核失败: %s — 降级为空列表", e)
            return []

        if not scored:
            logger.info("LLM 审核后无通过候选")
            return []

        # 日志：输出所有候选的评分
        for s in scored:
            logger.info("  LoRA [%s] %s → 评分 %d/10%s",
                        s["version_id"], s["name"][:40], s["_score"],
                        " ✅" if s["_score"] >= self.min_score else " ❌ (低于阈值)")

        # 按评分降序
        scored.sort(key=lambda a: (a["_score"], -a.get("_sim", 0)), reverse=True)

        # 截断到提交上限
        if len(scored) > self.max_submit:
            logger.info("截断 LoRA 数量 %d → %d（按评分）", len(scored), self.max_submit)
            scored = scored[:self.max_submit]

        logger.info("最终入选 %d 个 LoRA: %s",
                    len(scored), [(a["version_id"], a["name"][:25], f"{a['_score']}/10") for a in scored])

        return scored

    # ------------------------------------------------------------------
    # 内部 — 关键词提取
    # ------------------------------------------------------------------

    def _extract_query(self, scene_text: str) -> list[str]:
        """从场景文本提取搜索关键词，按优先级排序（高→低）。

        改进：引导 LLM 排除角色名称/名人等噪声词。
        """
        system = """You are a keyword extractor for CivitAI LoRA search.
From a scene description, extract 3-5 specific search terms.
Each term should be ONE concept: an object (e.g. "twin tails"), an action ("spread pussy"), or a setting ("toilet stall").
Rank by importance to the scene — the most critical visual element first.

CRITICAL: Choose terms that describe VISUAL ELEMENTS, not character identities.
Prefer generic descriptive terms over specific names.
Good: "black hair", "school uniform", "cherry blossoms"
Bad: "Wonder Woman", "Naruto", "Batman"

Output one term per line, nothing else. No numbering, no punctuation, no explanations."""
        try:
            response = self.llm.chat(system, f"Scene: {scene_text}")
            keywords = [line.strip().lower() for line in response.strip().split("\n") if line.strip()]
            keywords = keywords[:5]  # max 5 queries
            logger.info("LLM 提取查询词 (%d, 按优先级): %s", len(keywords), keywords)
            return keywords if keywords else [scene_text[:40]]
        except Exception:
            return [scene_text[:40]]

    # ------------------------------------------------------------------
    # 内部 — 文本相似度排序
    # ------------------------------------------------------------------

    def _rank_by_similarity(self, candidates: list[dict], queries: list[str]) -> list[dict]:
        """用搜索词-名称文本相似度对候选重新排序。

        把名称与搜索关键词更匹配的候选推到前面，
        减少角色特化 LoRA 因热度高而占位的问题。
        queries: 英文搜索关键词列表（由 _extract_query 产生）。
        """
        # 合并所有搜索词为比较文本
        query_text = " ".join(queries).lower()
        for c in candidates:
            name_lower = c.get("name", "").lower()
            # 序列相似度
            sim = SequenceMatcher(None, query_text[:200], name_lower[:200]).ratio()
            # 关键词命中加分
            name_words = set(name_lower.replace("-", " ").replace("_", " ").split())
            query_words = set(query_text.split())
            overlap = len(name_words & query_words)
            c["_sim"] = sim + overlap * 0.2  # 每个重叠词 +0.2
        # 按 _sim 降序排序
        candidates.sort(key=lambda c: c.get("_sim", 0), reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 内部 — CivitAI 搜索
    # ------------------------------------------------------------------

    def _search_civitai(self, queries: list[str]) -> list[dict]:
        """多线程并行搜索 CivitAI API，合并去重结果。返回结果附带 _priority。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from urllib.parse import quote

        def _search_one(query: str, priority: int) -> list[dict]:
            """搜索单个关键词，最多重试 3 次。"""
            if query in self._cache:
                logger.debug("缓存命中: %s", query[:30])
                return self._cache[query]

            url = f"{self._api_base}/models?query={quote(query)}&type=LORA&nsfw=true&limit={self.search_limit}"
            last_error = None
            for attempt in range(1, 4):
                try:
                    data = self._http_get(url, use_proxy=True)
                    items = data.get("items", [])
                    for item in items:
                        item["_priority"] = priority
                    logger.info("CivitAI 搜索 '%s' → %d results (attempt %d)", query[:40], len(items), attempt)
                    self._cache[query] = items
                    return items
                except Exception as e:
                    last_error = e
                    if attempt < 3:
                        wait = 3 * attempt
                        logger.info("搜索 '%s' 重试 %d/3 (等待 %ds)", query[:40], attempt, wait)
                        time.sleep(wait)
            logger.warning("搜索 '%s' 失败（3次重试）: %s", query[:40], last_error)
            return []

        if not queries:
            return []

        logger.info("并行搜索 %d 个关键词...", len(queries))
        seen_ids = set()
        all_items = []

        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            futures = {executor.submit(_search_one, q, i): q for i, q in enumerate(queries)}
            for future in as_completed(futures):
                items = future.result()
                for item in items:
                    mid = str(item.get("id", ""))
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        all_items.append(item)

        logger.info("合并后 %d 个唯一结果", len(all_items))
        return all_items

    def _http_get(self, url: str, use_proxy: bool = True) -> dict:
        """发送 HTTP GET。优先用 urllib（低开销），SOCKS5 路径回退 curl。

        若配置了 proxy_url 则直连（反代 URL 本身已是代理），
        否则走 SOCKS5 代理（CivitAI 国内被墙）。
        """
        has_proxy_url = self._api_base != _CIVITAI_API

        # 路径 1: 反代 URL 或直连 → urllib（无子进程开销）
        if has_proxy_url or not use_proxy:
            req = Request(url, headers={"User-Agent": "AI-Horde-Workflow/1.0"})
            try:
                with urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                raise RuntimeError(f"urllib 请求失败: {url[:100]} — {e}")

        # 路径 2: SOCKS5 代理 → curl fallback（urllib 不原生支持 SOCKS5）
        import subprocess
        cmd = [
            "curl", "-s", "--max-time", "15",
            "--socks5", "127.0.0.1:10808",
            "-H", "User-Agent: AI-Horde-Workflow/1.0",
            url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=20)
            if result.returncode != 0:
                raise RuntimeError(f"curl exit {result.returncode}: {result.stderr[:100]}")
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("curl timed out")

    # ------------------------------------------------------------------
    # 内部 — 候选解析
    # ------------------------------------------------------------------

    # SDXL 兼容的 base model 集合
    _SDXL_COMPATIBLE = {
        "sdxl", "sdxl 1.0", "illustrious", "pony", "noobai",
        "animagine", "anima", "sdxl lightning",
    }

    def _base_compatible(self, lo_base: str, target: str) -> bool:
        """检查 LoRA 的 base model 是否与目标兼容。"""
        lo_lower = lo_base.lower().strip()
        target_lower = target.lower().strip()
        # SDXL 目标接受 SDXL/Illustrious/Pony/NoobAI/Anima 等
        if "sdxl" in target_lower or target_lower == "sdxl":
            return lo_lower in self._SDXL_COMPATIBLE or "sdxl" in lo_lower
        # SD1.5 目标只接受 SD1.5/Other
        if "sd1.5" in target_lower or "sd 1.5" in target_lower:
            return lo_lower in {"sd 1.5", "sd1.5", "other"}
        # 其他情况宽松处理
        return True

    def _parse_candidate(self, item: dict, meta: dict, version_id: str) -> Optional[dict]:
        """将 CivitAI 搜索结果解析为统一的 LoRA 配置格式。"""
        model_name = item.get("name", "Unknown")
        base_model = meta.get("baseModel", "Unknown")
        trained_words = meta.get("trainedWords", [])

        # 自动生成关键词
        keywords = self._gen_keywords(model_name, trained_words)

        return {
            "name": model_name,
            "version_id": version_id,
            "model_id": str(item.get("id", "")),
            "keywords": [kw.lower() for kw in keywords],
            "trigger_words": trained_words if trained_words else [],
            "base_model": base_model,
            "strength_model": 0.8,
            "strength_clip": 0.8,
            "description": model_name,
        }

    def _gen_keywords(self, name: str, trained_words: list) -> list[str]:
        """从模型名和触发词生成搜索关键词。"""
        keywords = set()

        # 从触发词提取
        for w in trained_words:
            for part in w.lower().split():
                if len(part) >= 3:
                    keywords.add(part)

        # 从名称提取（取英文部分的前几个词）
        name_clean = re.sub(r'[\(\[].*?[\)\]]', '', name)
        name_clean = re.sub(r'[/\\]', ' ', name_clean)
        for word in name_clean.lower().split():
            word = word.strip('.,;:!?"\'')
            if len(word) >= 3 and word not in {"the", "and", "for", "with", "from", "that", "this", "v1", "v10"}:
                keywords.add(word)

        # 限制数量
        return list(keywords)[:15]

    # ------------------------------------------------------------------
    # 内部 — LLM 审核（评分制）
    # ------------------------------------------------------------------

    def _llm_review(self, scene_text: str, base_model: str,
                    candidates: list[dict]) -> list[dict]:
        """LLM 评分审核候选 LoRA。

        改为评分制（1-10 分），而非二元 pass/fail。
        评分 ≥ min_score 的入选，按分数降序排列。

        Returns:
            评分后的 LoRA 列表（含 _score 字段），可能为空。
        """
        if not candidates:
            return []

        # 构建候选列表（含描述，让 LLM 更好地判断）
        cand_lines = []
        for i, c in enumerate(candidates):
            cand_lines.append(
                f"{i}: [{c['version_id']}] {c['name']}\n"
                f"   base={c['base_model']}, trigger={c['trigger_words'][:5] if c['trigger_words'] else 'none'}"
            )
        cand_text = "\n".join(cand_lines)

        system = f"""你是一个 AI 绘画 LoRA 评分助手。根据以下条件为每个候选 LoRA 打分：

目标场景: {scene_text[:200]}
目标基础模型: {base_model}（SDXL/Illustrious/Pony/NoobAI 都视为兼容）

评分标准 (1-10 分):
- 9-10: 完美匹配，LoRA 主题与场景高度相关，触发词可直接增强画面
- 7-8: 良好匹配，LoRA 风格/元素与场景有一定关联
- 5-6: 可接受，LoRA 不冲突但也不直接相关（如通用风格 LoRA）
- 3-4: 弱相关，可能有少量元素重叠
- 1-2: 完全不相关，角色特化或无关主题

注意：
- 角色特化 LoRA（如 "Wonder Girl"、"Batman" 等）若与场景无关，给 1-2 分
- 风格/质感 LoRA（如 "watercolor style"、"detail enhancer"）可给 5-7 分
- 即使没有完美匹配，也要认真评估每个候选，给合理的分数

输出 JSON：
{{"scores": [{{"index": 0, "score": 8, "reason": "简短理由"}}, ...]}}
每个候选都必须打分，只输出 JSON，不要其他内容。"""

        try:
            response = self.llm.chat(system, cand_text, json_mode=True)
            result = json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM 审核解析失败: %s", e)
            return []

        scores = result.get("scores", [])
        # 兼容旧格式：LLM 可能仍返回 approved/rejected
        if not scores and "approved" in result:
            logger.info("LLM 返回旧格式 approved/rejected，自动转换")
            approved_indices = set(result.get("approved", []))
            for i in range(len(candidates)):
                scores.append({
                    "index": i,
                    "score": 8 if i in approved_indices else 2,
                    "reason": "auto-converted from old format",
                })

        # 校验 + 附加分数
        scored = []
        for s in scores:
            idx = s.get("index", -1)
            score = s.get("score", 0)
            if 0 <= idx < len(candidates):
                c = dict(candidates[idx])
                c["_score"] = max(1, min(10, int(score)))  # clamp 1-10
                c["_reason"] = s.get("reason", "")
                scored.append(c)

        # 过滤：只保留评分 ≥ min_score 的
        passed = [s for s in scored if s["_score"] >= self.min_score]

        if not passed and scored:
            logger.info("所有候选评分均低于阈值 %d，选取最高分候选", self.min_score)
            best = max(scored, key=lambda s: s["_score"])
            best["_reason"] = f"自动入选（最高分 {best['_score']}/10，低于阈值但无更好选择）: {best.get('_reason', '')}"
            passed = [best]

        return passed

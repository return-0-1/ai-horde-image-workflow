"""
LoRA 搜索器 — 从 CivitAI 搜索 LoRA + LLM 审核适配性。

流程:
  场景文本 → 提取搜索关键词 → CivitAI API 搜索 (nsfw=true)
  → 获取每个 version 元数据 → 黑名单过滤 → LLM 审核 → 返回候选

LLM 审核失败 → 返回空列表（调用方回退白名单）。
"""
import re
import json
import time
import logging
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# CivitAI 需要代理（国内被墙）
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
        self.review_limit = 5  # 最多送给 LLM 审核的候选数
        self.max_submit = config.get("lora_search", {}).get("max_loras", 2)  # 最终提交上限
        self._cache: dict[str, list] = {}  # query → 缓存的搜索结果

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

        # 1. 从场景文本提取搜索关键词（拆分为独立短查询）
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

            # 获取完整 metadata（用 curl 走代理，不用 civitai_client 的 urllib）
            try:
                meta = self._http_get(
                    f"{_CIVITAI_API}/model-versions/{version_id}",
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

        # 限制候选数量
        candidates = candidates[:self.review_limit]
        logger.info("候选 LoRA (%d 个): %s", len(candidates),
                    [(c["version_id"], c["name"]) for c in candidates])

        # 4. LLM 审核
        try:
            approved = self._llm_review(scene_text, base_model, candidates)
        except Exception as e:
            logger.warning("LLM 审核失败: %s — 降级为空列表", e)
            return []

        logger.info("LLM 审核通过 %d 个 LoRA: %s",
                    len(approved), [(a["version_id"], a.get("_priority", "?")) for a in approved])

        # 按优先级排序（高优先级 = 低 _priority 值）
        approved.sort(key=lambda a: a.get("_priority", 99))

        # 截断到提交上限（避免 AI Horde LoRA 过载）
        if len(approved) > self.max_submit:
            logger.info("截断 LoRA 数量 %d → %d（按优先级）", len(approved), self.max_submit)
            approved = approved[:self.max_submit]

        return approved

    # ------------------------------------------------------------------
    # 内部 — 关键词提取
    # ------------------------------------------------------------------

    def _extract_query(self, scene_text: str) -> list[str]:
        """从场景文本提取搜索关键词，按优先级排序（高→低）。"""
        system = """You are a keyword extractor. From a scene description, extract 3-5 specific search terms for CivitAI LoRA search.
Each term should be ONE concept: an object (e.g. "twin tails"), an action ("spread pussy"), or a setting ("toilet stall").
Rank by importance to the scene — the most critical visual element first.
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

            url = f"{_CIVITAI_API}/models?query={quote(query)}&type=LORA&nsfw=true&limit={self.search_limit}"
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
        """发送 HTTP GET，默认走 SOCKS5 代理（CivitAI 国内被墙）。
        
        使用 curl --socks5 而非 Python urllib ProxyHandler，
        因为 Python 的 socks5 代理在 Windows 上不稳定。
        """
        import subprocess
        
        if use_proxy:
            cmd = [
                "curl", "-s", "--max-time", "15",
                "--socks5", "127.0.0.1:10808",
                "-H", "User-Agent: AI-Horde-Workflow/1.0",
                url,
            ]
        else:
            cmd = [
                "curl", "-s", "--max-time", "10",
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
    # 内部 — LLM 审核
    # ------------------------------------------------------------------

    def _llm_review(self, scene_text: str, base_model: str,
                    candidates: list[dict]) -> list[dict]:
        """LLM 审核候选 LoRA 的兼容性和匹配度。

        Returns:
            审核通过的 LoRA 列表。
        """
        if not candidates:
            return []

        # 构建候选列表
        cand_lines = []
        for i, c in enumerate(candidates):
            cand_lines.append(
                f"{i}: [{c['version_id']}] {c['name']}\n"
                f"   base={c['base_model']}, trigger={c['trigger_words']}"
            )
        cand_text = "\n".join(cand_lines)

        system = f"""你是一个 AI 绘画 LoRA 审核助手。根据以下条件审核候选 LoRA：

目标场景: {scene_text[:200]}
目标基础模型: {base_model}（优先 SDXL/Illustrious/Pony 兼容的）

审核规则:
1. base_model 必须与目标兼容：SDXL/Illustrious/Pony/NoobAI 都视为兼容 SDXL
2. SD1.5 / Other / Flux / Wan Video 的排除
3. 与场景匹配度：LoRA 描述/触发词是否与场景相关
4. 排除明显无关的（如角色特化 LoRA 与场景无关）

输出 JSON：
{{"approved": [索引号, ...], "rejected": [{{"index": 索引号, "reason": "原因"}}, ...]}}
只输出 JSON，不要其他内容。"""

        try:
            response = self.llm.chat(system, cand_text, json_mode=True)
            result = json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM 审核解析失败: %s", e)
            return []

        approved_indices = set(result.get("approved", []))
        rejected = result.get("rejected", [])

        for r in rejected:
            logger.info("LLM 拒绝 → [%s] %s: %s",
                        candidates[r["index"]]["version_id"],
                        candidates[r["index"]]["name"][:40],
                        r.get("reason", "无原因"))

        approved = [candidates[i] for i in approved_indices if 0 <= i < len(candidates)]
        return approved

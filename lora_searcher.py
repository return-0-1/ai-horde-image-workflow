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

import net

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
        # 权重上限：LoRA 权重过高会盖掉 prompt 主体和主模型的画风（实测 0.8 时 prompt 里的
        # "anime style" 与主体词被打包带走），自动搜索的 LoRA 一律按此下限强度提交
        self.strength_model = float(config.get("lora_search", {}).get("strength_model", 0.5))
        self.strength_clip = float(config.get("lora_search", {}).get("strength_clip", 0.45))
        # 角色特化 LoRA（复现某个具体人物）默认不参与：场景没点名它，挂上去只会污染主体
        self.reject_unrequested_characters = bool(
            config.get("lora_search", {}).get("reject_unrequested_characters", True)
        )
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
                # 角色特化 LoRA：场景没点名这个角色时直接排除。
                # 名字字面相似 ≠ 能用在场景里（"Blue hair OC(test)" 靠 "blue hair" 命中，
                # 挂到"DeepSeek 拟人化少女"上纯粹是污染）
                if self.reject_unrequested_characters and self._is_character_lora(
                        lora_info, scene_text, queries):
                    continue
                lora_info["_priority"] = item.get("_priority", 99)
                candidates.append(lora_info)

        if not candidates:
            logger.info("无可用候选（全部被过滤）")
            return []

        # 3.5 文本相似度预排序：用英文搜索词代替中文场景做比较
        proper = self._latin_proper_nouns(scene_text)
        candidates = self._rank_by_similarity(candidates, queries, proper)

        # 截断候选时保送“名字命中专有名词”的结果
        candidates = self._pick_review_candidates(candidates)
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

    _LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#._-]{2,}")

    def _latin_proper_nouns(self, text: str) -> list[str]:
        """抽出场景里的拉丁专有名词（DeepSeek / Genshin / Miku / RWBY）。

        LoRA 大多以 IP、品牌、角色名命名，而这类词恰恰是 LLM 提取查询词时最容易
        被当成"噪声"删掉的。这里不依赖 LLM 单独保一份，保证它们一定会被搜到。
        """
        seen, out = set(), []
        for tok in self._LATIN_TOKEN.findall(text or ""):
            low = tok.lower().strip("._-")
            if len(low) < 3 or low in self._GENERIC_WORDS or low in seen:
                continue
            seen.add(low)
            out.append(low)
        return out

    def _fallback_queries(self, scene_text: str, proper: list[str]) -> list[str]:
        """LLM 提取失败时的兜底：按标点切分，优先含拉丁字母的片段。

        旧实现直接拿 scene_text[:40] 当查询词 —— 中文长句在 CivitAI 上几乎零命中，
        而且 40 字截断经常正好把关键英文名截掉。
        """
        parts = [p.strip() for p in re.split(r"[，,。.；;、\n！!？?｜|]", scene_text or "")
                 if p.strip()]
        latin = [p[:40] for p in parts if re.search(r"[A-Za-z]{2,}", p)]
        out = list(proper)
        for p in latin:
            if all(p.lower().find(q) < 0 and q not in p.lower() for q in proper):
                out.append(p)
        if not out:
            out = [p[:40] for p in parts[:3]] or [(scene_text or "")[:40]]
        return out[:5]

    def _extract_query(self, scene_text: str) -> list[str]:
        """从场景文本提取搜索关键词，按优先级排序（高→低）。

        专有名词（IP / 品牌 / 角色名）无条件优先：CivitAI 的 LoRA 就是以这些名字命名的，
        而 LLM 倾向把它们当"角色身份噪声"删掉，导致 "DeepSeek 拟人化少女" 只搜到
        泛泛的 anime style。
        """
        system = """You are a keyword extractor for CivitAI LoRA search.
From a scene description, extract 3-6 specific search terms, ONE concept each:
an object ("twin tails"), clothing ("black lace lingerie"), an action ("spread pussy"),
a setting ("toilet stall"), or an art style ("flat anime").

ALWAYS include any proper noun / IP / brand / character name that appears in the scene,
verbatim (e.g. "DeepSeek", "Genshin Impact", "Hatsune Miku", "RWBY") — LoRAs are usually
named after them, so they are the highest-value queries. Only drop names of unrelated
real people.

Translate Chinese concepts into short ENGLISH terms: CivitAI search does not match
Chinese sentences. Never emit a whole sentence.

Output one term per line, nothing else. No numbering, no punctuation, no explanations."""
        proper = self._latin_proper_nouns(scene_text)
        try:
            response = self.llm.chat(system, f"Scene: {scene_text}", thinking=False)
            keywords = [line.strip().lower() for line in response.strip().split("\n") if line.strip()]
            keywords = [k for k in keywords if k]
            merged = proper + [k for k in keywords if k not in proper]
            merged = merged[:6]  # max 6 queries
            logger.info("LLM 提取查询词 (%d, 按优先级): %s", len(merged), merged)
            return merged or proper or [scene_text[:40]]
        except Exception as e:
            fallback = self._fallback_queries(scene_text, proper)
            logger.warning("查询词提取失败（%s）→ 规则切分兜底: %s", type(e).__name__, fallback)
            return fallback

    # ------------------------------------------------------------------
    # 内部 — 文本相似度排序
    # ------------------------------------------------------------------

    def _rank_by_similarity(self, candidates: list[dict], queries: list[str],
                            proper: Optional[list[str]] = None) -> list[dict]:
        """用搜索词-名称文本相似度对候选重新排序。

        把名称与搜索关键词更匹配的候选推到前面，
        减少角色特化 LoRA 因热度高而占位的问题。
        queries: 英文搜索关键词列表（由 _extract_query 产生）。
        proper:  场景里的专有名词（DeepSeek / RWBY…），命中名称的候选额外加权 ——
                 这类名字常含中文或符号，纯文本相似度天然吃亏（"deepseek娘-小鲸鱼" vs
                 英文 query 只有 0.22，会被泛泛的 "Anime ink illustration" 挤掉）。
        """
        query_text = " ".join(queries).lower()
        proper_lower = [p.lower() for p in (proper or []) if p]
        for c in candidates:
            name_lower = c.get("name", "").lower()
            # 序列相似度
            sim = SequenceMatcher(None, query_text[:200], name_lower[:200]).ratio()
            # 关键词命中加分
            name_words = set(name_lower.replace("-", " ").replace("_", " ").split())
            query_words = set(query_text.split())
            overlap = len(name_words & query_words)
            c["_sim"] = sim + overlap * 0.2  # 每个重叠词 +0.2
            # 专有名词直接出现在名字里 → 加权（最高价值的信号）
            c["_named_hit"] = bool(proper_lower) and any(p in name_lower for p in proper_lower)
            if c["_named_hit"]:
                c["_sim"] += 0.5
        # 按 _sim 降序排序
        candidates.sort(key=lambda c: c.get("_sim", 0), reverse=True)
        return candidates

    def _pick_review_candidates(self, candidates: list[dict], per_named: int = 3) -> list[dict]:
        """截断到 review_limit，但保送名字命中专有名词的候选（最多 per_named 个）。"""
        if len(candidates) <= self.review_limit:
            return candidates
        named = [c for c in candidates if c.get("_named_hit")][:per_named]
        if not named:
            return candidates[:self.review_limit]
        named_ids = {id(c) for c in named}
        rest = [c for c in candidates if id(c) not in named_ids]
        kept = named + rest[:max(0, self.review_limit - len(named))]
        logger.info("候选截断 %d → %d（保送 %d 个专有名词命中: %s）",
                    len(candidates), len(kept), len(named),
                    [c["name"][:30] for c in named])
        return kept

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
        """发送 HTTP GET。优先用 urllib（低开销），失败回退 curl。

        出网代理由 net.py 统一决定（WORKFLOW_PROXY）；
        若配置了 proxy_url（反代）则直连反代即可。
        """
        has_proxy_url = self._api_base != _CIVITAI_API

        # 路径 1: 反代 URL 或直连 → urllib；代理不可用时自动回退直连
        if has_proxy_url or not use_proxy:
            req = Request(url, headers={"User-Agent": "AI-Horde-Workflow/1.0"})
            try:
                with net.open_with_fallback(req, timeout=10) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                raise RuntimeError(f"urllib 请求失败: {url[:100]} — {e}")

        # 路径 2: curl fallback（urllib 不原生支持 SOCKS5；有项目代理则用它）
        import subprocess
        proxy = net.resolve_proxy()
        cmd = ["curl", "-s", "--max-time", "15",
               "-H", "User-Agent: AI-Horde-Workflow/1.0"]
        cmd += ["-x", proxy] if proxy else ["--socks5", "127.0.0.1:10808"]
        cmd.append(url)
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

    # CivitAI baseModel 取值 → 家族。只有同家族（或同源 CLIP）才能混用：
    #   Illustrious / NoobAI 同属 Illustrious 系；
    #   Pony、SDXL 1.0、Anima、SD1.5 各自独立训练，跨家族加载会让画风被拖走或细节丢失
    #   （实测：Pony 版服装 LoRA 挂到 Illustrious 上，服装细节没出来，画风被拉成半写实）
    _BASE_FAMILY = {
        "illustrious": {"illustrious", "noobai", "noobai xl", "illustrious xl", "illustrious xl 1.0"},
        "pony": {"pony", "pony diffusion", "ponyxl", "pony diffusion v6 xl"},
        "sdxl": {"sdxl", "sdxl 1.0", "sdxl lightning", "sdxl turbo"},
        "sd15": {"sd 1.5", "sd1.5", "sd 1.5 inpainting"},
        "flux": {"flux", "flux.1", "flux.1 dev", "flux.1 schnell"},
    }

    def _family(self, base: str) -> str:
        """把 baseModel 字符串归到家族；未知归 other。"""
        b = (base or "").lower().strip()
        for fam, names in self._BASE_FAMILY.items():
            if b in names:
                return fam
        # 模糊匹配：如 "Illustrious XL v1.0" / "Pony Diffusion V6"
        for fam, names in self._BASE_FAMILY.items():
            if any(n in b for n in names if len(n) > 3):
                return fam
        if "illustrious" in b or "noobai" in b:
            return "illustrious"
        if "pony" in b:
            return "pony"
        if "sdxl" in b:
            return "sdxl"
        return "other"

    def _base_compatible(self, lo_base: str, target: str) -> bool:
        """LoRA 的 base 是否可挂到目标主模型上（同家族才放行）。"""
        lo_fam = self._family(lo_base)
        tgt_fam = self._family(target)
        if lo_fam == "other":
            # 元数据缺 baseModel 时不赌：跨基座硬挂的代价比少挂一个 LoRA 大得多
            return False
        if lo_fam == tgt_fam:
            return True
        # SDXL 主模型可接受 Illustrious/NoobAI（同源 CLIP，衍生关系）
        if tgt_fam == "sdxl" and lo_fam == "illustrious":
            return True
        return False

    # 名字里出现即视为“通用描述词”（小写）。这些词不构成专有名词。
    _GENERIC_WORDS = {
        "lora", "loha", "locon", "lycoris", "style", "styles", "styled", "anime", "manga",
        "cartoon", "illustrious", "noobai", "pony", "ponyxl", "sdxl", "xl", "sd15", "flux",
        "test", "beta", "alpha", "wip", "final", "nsfw", "sfw", "concept", "clothing",
        "clothes", "outfit", "outfits", "pose", "poses", "position", "positions", "action",
        "actions", "expression", "expressions", "gesture", "lingerie", "underwear",
        "innerwear", "bra", "panties", "lace", "harness", "garter", "stockings", "dress",
        "skirt", "uniform", "tactical", "armor", "armour", "weapon", "weapons", "background",
        "scene", "lighting", "detail", "detailed", "realistic", "photo", "photography",
        "photorealistic", "portrait", "camera", "angle", "view", "viewpoint", "perspective",
        "quality", "enhancer", "slider", "helper", "tool", "utility", "bonus", "pack", "set",
        "collection", "series", "art", "artstyle", "illustration", "painting", "watercolor",
        "sketch", "lineart", "flat", "color", "colors", "colour", "shading", "rendering",
        "3d", "2d", "cgi", "cg", "pixel", "pixelart", "ai", "the", "of", "and", "for",
        "with", "by", "in", "on", "dark", "light", "black", "white", "red", "blue", "green",
        "yellow", "pink", "purple", "brown", "grey", "gray", "orange", "midnight", "night", "day",
        "sunset", "dawn", "moon", "moonlight", "glow", "glossy", "shiny", "soft", "hard",
        "cute", "sexy", "hot", "casual", "formal", "summer", "winter", "spring", "autumn",
        "festival", "party", "wedding", "school", "office", "gym", "sport", "swim",
        "swimsuit", "bikini", "nightwear", "sleepwear", "pajamas", "maid", "nurse", "bunny",
        "cat", "dog", "fox", "wolf", "dragon", "angel", "demon", "succubus", "elf", "ghost",
        "monster", "giant", "small", "big", "huge", "long", "short", "hair", "eyes", "eye",
        "face", "body", "skin", "female", "male", "girl", "boy", "woman", "man", "adult",
        "young", "mature", "slim", "thick", "chubby", "muscular", "tall", "solo", "duo",
        "group", "full", "half", "upper", "lower", "close", "wide", "shot", "front", "rear",
        "side", "top", "bottom", "left", "right", "v1", "v2", "v3", "v4", "v5", "version",
        "pdxl", "ilxl", "xl", "pony", "il", "gba", "hd", "4k", "8k",
        # 身体部位 / 性行为 / 常见姿态词：这些出现在名字里同样不算“专有名词”，
        # 否则 "Self Pussy Spread (pose)" 一类的姿势 LoRA 会被当成角色
        "self", "spread", "spreading", "spreaded", "open", "closed", "anal", "sex", "sexual",
        "nude", "naked", "breast", "breasts", "boobs", "ass", "thigh", "thighs", "leg", "legs",
        "arm", "arms", "hand", "hands", "finger", "fingers", "cum", "penetration",
        "masturbation", "fingering", "oral", "kiss", "kissing", "standing", "sitting",
        "lying", "kneeling", "bent", "holding", "grip", "hips", "torso", "waist", "chest",
        "nipple", "nipples", "genital", "genitals", "pussy", "penis", "pov", "view", "crop",
        "closeup", "close-up", "fullbody", "full-body", "halfbody", "upperbody", "lowerbody",
        "frontview", "rearview", "sideview", "from", "behind", "above", "below",
    }

    # 触发词里出现这些身份标注 → 该 LoRA 是训练来复现某个具体人物的
    _CHARACTER_MARKERS = ("1girl", "1boy", "1other", "character sheet")

    # 名字里的专有名词达到这个数量才判定为角色/作品特化（单个大写词太容易是描述词）
    _CHARACTER_PROPER_NOUN_MIN = 2

    def _proper_nouns(self, name: str) -> set[str]:
        """提取模型名里像“专有名词”的词（角色名/作品名）。"""
        out = set()
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.\-]*", name or ""):
            low = t.lower().strip(".-'’")
            if len(low) < 3 or low in self._GENERIC_WORDS:
                continue
            if t[0].isupper():
                out.add(low)
        return out

    def _is_character_lora(self, lora: dict, scene_text: str,
                           queries: list[str]) -> bool:
        """是否“复现某个具体人物”的 LoRA，且场景没有点名它。

        名字字面相似不等于能用：`Blue hair OC(test)` 靠 "blue hair" 命中一个
        "DeepSeek 拟人化少女" 的场景，挂上去只会污染主体。判定后若场景/检索词里
        出现该角色名，则视为用户点名，放行。
        """
        name = lora.get("name", "") or ""
        name_low = name.lower()
        triggers = " ".join((lora.get("trigger_words") or [])[:2]).lower()

        names = self._proper_nouns(name)
        is_char = False

        # R1: 有“角色 - 作品”或括号/斜杠格式，且含 ≥2 个专有名词
        #     （Ruby Rose - RWBY / Ludmilla (Bardock…) / Alice Lenaford/アニス… - Dark Bl）
        #     单独一个专有名词太容易是描述词（Orange Bull / Self Pussy Spread (pose)），不判
        if re.search(r"\s-\s|\(|/", name) and len(names) >= self._CHARACTER_PROPER_NOUN_MIN:
            is_char = True
        # R2: 显式标签
        if not is_char and re.search(r"\b(oc|character|cosplay)\b", name_low):
            is_char = True
        # R3: 触发词以人物标注开头，且名字不是服装/姿势/风格类
        if not is_char and any(m in triggers for m in self._CHARACTER_MARKERS):
            if not re.search(r"(style|clothing|outfit|pose|position|lingerie|underwear|"
                             r"uniform|armor|armour|weapon)", name_low):
                is_char = True

        if not is_char:
            return False

        # 豁免：场景或检索词里点名了这个角色
        blob = f"{scene_text} {' '.join(queries)}".lower()
        hit = sorted(w for w in names if w and w in blob)
        if hit:
            logger.info("角色 LoRA 被场景点名，放行: %s（命中 %s）", name[:40], hit)
            return False
        logger.info("跳过角色特化 LoRA [%s] %s（场景未点名）",
                    lora.get("version_id", "?"), name[:50])
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
            "strength_model": self.strength_model,
            "strength_clip": self.strength_clip,
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
目标基础模型: {base_model}（只有同家族 base 的 LoRA 才会入选；跨基座加载会拖走画风）

评分标准 (1-10 分):
- 9-10: 完美匹配，LoRA 主题与场景高度相关，触发词可直接增强画面
- 7-8: 良好匹配，LoRA 风格/元素与场景有一定关联
- 5-6: 可接受，LoRA 不冲突但也不直接相关（如通用风格 LoRA）
- 3-4: 弱相关，可能有少量元素重叠
- 1-2: 完全不相关，角色特化或无关主题

注意：
- 角色特化 LoRA（如 "Wonder Girl"、"Batman" 等）若与场景无关，给 1-2 分
- 风格/质感 LoRA（如 "watercolor style"、"detail enhancer"）只有在场景明确要那种画风时
  才给 7 分以上；场景是动漫/插画时，写实、3DCG、摄影类风格 LoRA 给 2-3 分
- 场景里已有明确主体（动物/角色/物体）时，会改变画风的风格 LoRA 要扣分，避免主体被覆盖
- 即使没有完美匹配，也要认真评估每个候选，给合理的分数

输出 JSON：
{{"scores": [{{"index": 0, "score": 8, "reason": "简短理由"}}, ...]}}
每个候选都必须打分，只输出 JSON，不要其他内容。"""

        try:
            # thinking=False：评分是「照标准打标签」，不需要长思考。开启思考时
            # reasoning 会与 JSON 答案共用 max_tokens，导致答案被截断（实测过一次
            # 思考烧掉 3236/4096），json.loads 报 "Unterminated string"。
            response = self.llm.chat(system, cand_text, json_mode=True, thinking=False)
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

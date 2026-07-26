"""
CivitAI API 客户端 — 获取 LoRA 元数据。

关键设计:
  - 超时/不可用时不影响核心生成功能 → 降级为本地 LoRA 配置信息
  - 缓存机制避免重复请求
"""

import json
import time
import logging
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


class CivitAIClient:
    """CivitAI API 客户端（带缓存 & 降级）。"""

    def __init__(self, config: dict):
        cfg = config.get("civitai", {})
        self.api_url = cfg.get("api_url", "https://civitai.com/api/v1").rstrip("/")
        self.timeout = cfg.get("timeout", 10)
        self._cache: dict[str, dict] = {}      # model_id → metadata
        self._version_cache: dict[str, dict] = {}  # version_id → metadata

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def get_model_meta(self, model_id: int) -> Optional[dict]:
        """获取模型元数据（通过 CivitAI model_id）。

        Returns:
            模型信息 dict，或 None（请求失败/超时）。
        """
        if model_id in self._cache:
            return self._cache[model_id]

        try:
            data = self._get(f"/models/{model_id}")
            self._cache[model_id] = data
            return data
        except Exception as e:
            logger.warning("CivitAI 模型元数据获取失败 (model_id=%s): %s", model_id, e)
            return None

    def get_version_meta(self, version_id: int) -> Optional[dict]:
        """获取版本元数据（通过 version_id）。

        Returns:
            版本信息 dict，或 None。
        """
        if version_id in self._version_cache:
            return self._version_cache[version_id]

        try:
            data = self._get(f"/model-versions/{version_id}")
            self._version_cache[version_id] = data
            return data
        except Exception as e:
            logger.warning("CivitAI 版本元数据获取失败 (version_id=%s): %s", version_id, e)
            return None

    def enrich_lora(self, lora: dict) -> dict:
        """用 CivitAI 在线元数据增强 LoRA 配置。

        尝试顺序: 1) model_id → 提取描述/触发词
                  2) version_id → 提取基础模型/触发词
        失败则保留原始 lora 信息。

        Args:
            lora: 本地 LoRA 配置 dict。

        Returns:
            增强后的 lora dict（含 description / online_trigger_words 等）。
        """
        enriched = dict(lora)
        enriched.setdefault("online_description", "")
        enriched.setdefault("online_trigger_words", [])

        model_id = lora.get("model_id")
        version_id = lora.get("version_id")

        # 优先用 model_id 拉完整信息
        if model_id:
            meta = self.get_model_meta(int(model_id))
            if meta:
                enriched["online_description"] = meta.get("description", "")[:500]

        # 用 version_id 补充触发词
        if version_id:
            v_meta = self.get_version_meta(int(version_id))
            if v_meta:
                triggers = v_meta.get("trainedWords", [])
                enriched["online_trigger_words"] = triggers
                if not enriched.get("base_model"):
                    enriched["base_model"] = v_meta.get("baseModel", "")

        return enriched

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        url = f"{self.api_url}{path}"
        logger.debug("CivitAI 请求 → %s", url)
        req = Request(url, headers={"User-Agent": "AI-Horde-Workflow/1.0"})

        # 直连（短超时，快速失败）
        try:
            with urlopen(req, timeout=min(self.timeout, 5)) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug("CivitAI 直连失败 → 尝试 curl 代理: %s", e)

        # curl fallback
        import subprocess
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", str(self.timeout),
                 "--socks5", "127.0.0.1:10808",
                 "-H", "User-Agent: AI-Horde-Workflow/1.0", url],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=self.timeout + 5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
        except Exception:
            pass

        raise RuntimeError(f"CivitAI 请求失败（直连+curl均失败）: {url}")

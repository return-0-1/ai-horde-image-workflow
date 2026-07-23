"""
LoRA 管理器 — 白名单维护 + 关键词/语义匹配。

设计:
  - 加载 config.yaml 中的 lora_whitelist
  - 根据用户场景文本匹配最相关的 LoRA
  - 匹配逻辑: 关键词命中 + 基础模型过滤
  - 匹配失败不阻断流程（可选 LoRA 或无 LoRA 生成）
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LoraManager:
    """LoRA 白名单管理器。"""

    def __init__(self, config: dict):
        raw = config.get("lora_whitelist", [])
        self._loras: list[dict] = []
        for item in raw:
            entry = {
                "name": item.get("name", "Unknown"),
                "version_id": str(item.get("version_id", "")),
                "model_id": item.get("model_id"),
                "keywords": [kw.lower() for kw in item.get("keywords", [])],
                "trigger_words": item.get("trigger_words", []),
                "base_model": item.get("base_model", "SDXL"),
                "strength_model": item.get("strength_model", 0.8),
                "strength_clip": item.get("strength_clip", 0.8),
                "description": item.get("description", ""),
            }
            self._loras.append(entry)
        logger.info("已加载 %d 个 LoRA 配置", len(self._loras))

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    @property
    def all(self) -> list:
        """返回全部 LoRA 配置列表（只读）。"""
        return list(self._loras)

    def match(self, scene_text: str, base_model: str = "SDXL", top_k: int = 3) -> list[dict]:
        """根据场景文本匹配 LoRA。

        匹配策略:
          1. 关键词匹配: scene_text 中出现 keyword → 得分+1
          2. 基础模型必须兼容（可宽松匹配: SDXL 可匹配 基于 SDXL 的 LoRA）

        Args:
            scene_text:  场景描述文本。
            base_model:  目标基础模型。
            top_k:       最多返回 N 个匹配。

        Returns:
            匹配的 LoRA 列表，按相关性降序排列。
        """
        text_lower = scene_text.lower()
        scored = []

        for lora in self._loras:
            score = 0
            for kw in lora["keywords"]:
                if kw and kw in text_lower:
                    score += 1
            # 名称命中额外加分
            if lora["name"].lower() in text_lower:
                score += 2

            if score > 0:
                scored.append((score, lora))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored[:top_k]]

        if results:
            logger.info("LoRA 匹配完成 → 命中 %d 个: %s",
                        len(results), [r["name"] for r in results])
        else:
            logger.info("LoRA 匹配 → 无命中，将不使用 LoRA")

        return results

    def get_by_name(self, name: str) -> Optional[dict]:
        """按名称精确查找 LoRA。"""
        for lora in self._loras:
            if lora["name"] == name:
                return lora
        return None

    def add_temp(self, lora_info: dict):
        """动态添加临时 LoRA（不持久化到配置文件）。"""
        entry = {
            "name": lora_info.get("name", "Temp"),
            "version_id": str(lora_info.get("version_id", "")),
            "model_id": lora_info.get("model_id"),
            "keywords": [kw.lower() for kw in lora_info.get("keywords", [])],
            "trigger_words": lora_info.get("trigger_words", []),
            "base_model": lora_info.get("base_model", "SDXL"),
            "strength_model": lora_info.get("strength_model", 0.8),
            "strength_clip": lora_info.get("strength_clip", 0.8),
            "description": lora_info.get("description", ""),
        }
        self._loras.append(entry)
        logger.info("动态添加 LoRA: %s", entry["name"])

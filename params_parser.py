"""
参数解析器 — 从中文/英文自然语言中提取图像生成参数。

支持口语化指令，如：
  "分辨率改成 1024x768"     → {"width": 1024, "height": 768}
  "steps 30 cfg 8"          → {"steps": 30, "cfg_scale": 8}
  "采样器用 DPM++ 2M"       → {"sampler": "k_dpmpp_2m"}
  "宽1024 高768 步数30"     → {"width": 1024, "height": 768, "steps": 30}
  "clip_skip=1 karras=true" → {"clip_skip": 1, "karras": True}
"""

import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 采样器名称映射（口语 → AI Horde 内部名）
# ---------------------------------------------------------------------------
SAMPLER_ALIASES = {
    # Euler 族
    "euler": "k_euler",
    "euler_a": "k_euler_a",
    "euler ancestral": "k_euler_a",
    "euler ancestral": "k_euler_a",
    # DPM 族
    "dpm++": "k_dpmpp_2m",
    "dpm++ 2m": "k_dpmpp_2m",
    "dpm++ 2m karras": "k_dpmpp_2m",
    "dpm++ 2s": "k_dpmpp_2s_a",
    "dpm++ 2s a": "k_dpmpp_2s_a",
    "dpm++ sde": "k_dpmpp_sde",
    "dpm++ sde karras": "k_dpmpp_sde",
    "dpm++ 2m sde": "k_dpmpp_2m_sde",
    "dpm++ 3m sde": "k_dpmpp_3m_sde",
    "dpm adaptive": "k_dpm_adaptive",
    "dpm fast": "k_dpm_fast",
    # DDIM / PLMS
    "ddim": "k_ddim",
    "plms": "k_plms",
    # LMS
    "lms": "k_lms",
    "lms karras": "k_lms",
    # Heun
    "heun": "k_heun",
    # UniPC
    "unipc": "k_unipc",
    # 其他
    "restart": "k_restart",
}

# 尺寸关键词
SIZE_KEYWORDS = [
    # 常见预设
    ("512x512", 512, 512),
    ("512x768", 512, 768),
    ("768x512", 768, 512),
    ("768x768", 768, 768),
    ("1024x1024", 1024, 1024),
    ("832x1216", 832, 1216),
    ("1216x832", 1216, 832),
    # 竖屏常见
    ("1024x1536", 1024, 1536),
    ("1536x1024", 1536, 1024),
    ("1080x1920", 1080, 1920),
    ("1920x1080", 1920, 1080),
]


def parse(user_text: str, llm_client=None) -> dict:
    """从用户文本中提取参数覆盖。

    策略:
      1. 正则快速匹配常见模式（覆盖 90% 场景）
      2. 若传入 llm_client 且有未覆盖的复杂指令，LLM 兜底

    Args:
        user_text:   用户自然语言文本
        llm_client:  可选 LLM 客户端，处理复杂指令

    Returns:
        dict，仅包含用户指定的参数。空 dict 表示无覆盖。
    """
    overrides = {}
    text = user_text

    # ---- 1. 分辨率 ----
    overrides.update(_parse_resolution(text))

    # ---- 2. 独立宽高 ----
    overrides.update(_parse_dimensions(text))

    # ---- 3. Steps / 步数 ----
    overrides.update(_parse_int(text, "steps", [
        r"\b(?:steps|步数|步)\s*(?:改成|换成|用|为|：|:|=|是)?\s*(\d+)",
        r"(\d+)\s*\b(?:steps|步数|步)\b",
    ]))

    # ---- 4. CFG Scale ----
    overrides.update(_parse_float(text, "cfg_scale", [
        r"\b(?:cfg|CFG|引导)\s*(?:改成|换成|用|为|：|:|=|是)?\s*(\d+\.?\d*)",
        r"\bcfg[=: ]*(\d+\.?\d*)",
    ]))

    # ---- 5. 采样器 ----
    overrides.update(_parse_sampler(text))

    # ---- 6. Clip Skip ----
    overrides.update(_parse_int(text, "clip_skip", [r"\bclip[_\s]*skip\s*[：:=]?\s*(\d)", r"跳过层?数?\s*[：:=]?\s*(\d)"]))

    # ---- 7. Karras / HiRes Fix 等布尔 ----
    bool_map = {
        "karras": [r"karras\s*[=：:]*\s*(true|false|是|否|开|关|启用|禁用|yes|no|on|off|1|0)"],
        "hires_fix": [r"(?:hires|高清修复|高分辨率修复)\s*[=：:]*\s*(true|false|是|否|开|关|启用|禁用|yes|no|on|off|1|0)"],
        "nsfw": [r"nsfw\s*[=：:]*\s*(true|false|是|否|开|关|启用|禁用|yes|no|on|off|1|0)"],
    }
    for key, patterns in bool_map.items():
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                overrides[key] = _to_bool(m.group(1))
                break

    # ---- 8. 模型别名 ----
    model_match = re.search(r"(?:模型|model)\s*[：:=]?\s*['\"]?([a-zA-Z0-9_\-\s\.()]+?)['\"]?(?:\s|$|,|，)", text, re.IGNORECASE)
    if model_match:
        overrides["model"] = model_match.group(1).strip()

    # ---- 9. Seed ----
    seed_match = re.search(r"(?:seed|种子)\s*(?:用|为|是|：|:|=)?\s*(-?\d+)", text, re.IGNORECASE)
    if seed_match:
        overrides["seed"] = seed_match.group(1)

    # ---- 10. 生成张数 ----
    overrides.update(_parse_int(text, "n", [r"\bn\s*[：:=]?\s*(\d+)", r"生成张数\s*[：:=]?\s*(\d+)", r"生成\s*(\d+)\s*张", r"(\d+)\s*张"]))

    # ---- 11. LLM 兜底 ----
    if llm_client and not overrides:
        llm_result = _llm_fallback(text, llm_client)
        overrides.update(llm_result)

    if overrides:
        # 过滤不合理的尺寸值（SDXL 要求 >= 256 且为 64 的倍数）
        overrides = _validate_dims(overrides)
        logger.info("参数覆盖: %s", json.dumps(overrides, ensure_ascii=False))

    return overrides


# ======================================================================
# 内部解析函数
# ======================================================================

def _parse_resolution(text: str) -> dict:
    """匹配 分辨率/尺寸/宽高 相关表达。"""
    result = {}

    # 模式: 数字x数字 (如 1024x768, 1024*768, 1024×768)
    m = re.search(r"(\d{3,4})\s*[xX×\*]\s*(\d{3,4})", text)
    if m:
        result["width"] = int(m.group(1))
        result["height"] = int(m.group(2))
        return result

    # 模式: 预设关键词
    for kw, w, h in SIZE_KEYWORDS:
        if kw.replace("x", "") in text.replace("x", "").replace("×", "").replace(" ", ""):
            result["width"] = w
            result["height"] = h
            return result

    return result


def _parse_dimensions(text: str) -> dict:
    """匹配独立的宽/高指定（要求词边界，避免匹配 age/version 等）。"""
    result = {}
    m_w = re.search(r"\b(?:宽|width|w)\s*[：:=]?\s*(\d+)", text, re.IGNORECASE)
    m_h = re.search(r"\b(?:高|height|h)\s*[：:=]?\s*(\d+)", text, re.IGNORECASE)
    if m_w:
        result["width"] = int(m_w.group(1))
    if m_h:
        result["height"] = int(m_h.group(1))
    return result


def _parse_int(text: str, key: str, patterns: list[str]) -> dict:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return {key: int(m.group(1))}
    return {}


def _parse_float(text: str, key: str, patterns: list[str]) -> dict:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return {key: float(m.group(1))}
    return {}


def _parse_sampler(text: str) -> dict:
    """从文本中提取采样器名称。"""
    text_lower = text.lower()

    # 先精确匹配英文关键字
    # 模式: "采样器 k_euler" / "用 DPM++ 采样器" / "sampler euler_a"
    m = re.search(r"(?:采样器|sampler)\s*[：:=]?\s*([a-zA-Z0-9_\+\s]+?)(?:\s|$|,|，)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().lower()
        # 尝试别名映射
        for alias, canonical in SAMPLER_ALIASES.items():
            if raw == alias or raw.replace(" ", "_") == canonical:
                return {"sampler": canonical}
        # 直接匹配内部名
        if raw.startswith("k_"):
            return {"sampler": raw}

    # 模糊匹配：在整段文本中找采样器别名
    for alias, canonical in sorted(SAMPLER_ALIASES.items(), key=lambda x: -len(x[0])):
        # 用词边界匹配避免 "a" 匹配到 "sampler"
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", text_lower):
            # 确认是在讨论采样器的上下文中
            if any(kw in text_lower for kw in ["采样器", "sampler", "采样", "sample"]):
                return {"sampler": canonical}
            # "用 dpm++" 这类口语也算
            if re.search(rf"(?:用|使用|换|改成)\s*{re.escape(alias)}", text_lower):
                return {"sampler": canonical}

    return {}


def _to_bool(val: str) -> bool:
    return val.lower() in ("true", "是", "开", "启用", "yes", "on", "1")


def _llm_fallback(text: str, llm_client) -> dict:
    """LLM 兜底：从复杂自然语言中提取参数。"""
    system = """你是一个参数提取器。从用户文本中提取图像生成参数。

支持的参数: width, height, steps, cfg_scale, sampler, clip_skip, karras, hires_fix, nsfw, seed, n, model

输出纯 JSON，只包含用户明确指定的参数：
{"width": 1024, "height": 768, "steps": 30, "cfg_scale": 8}

如果没有参数覆盖，输出 {}"""

    try:
        response = llm_client.chat(system, text, json_mode=True)
        return json.loads(response)
    except Exception:
        return {}


# ======================================================================
# 工具函数
# ======================================================================

def _validate_dims(overrides: dict) -> dict:
    """过滤不合理的宽高值（SDXL 要求 >= 256 且为 64 倍数）。"""
    MIN_DIM = 256
    MULTIPLE = 64
    for key in ("width", "height"):
        if key in overrides:
            val = overrides[key]
            if val < MIN_DIM or val % MULTIPLE != 0:
                logger.warning(
                    "忽略不合理的 %s=%d（需 >=%d 且为 %d 倍数）", key, val, MIN_DIM, MULTIPLE
                )
                del overrides[key]
    # n 限制在合理范围
    if "n" in overrides and (overrides["n"] < 1 or overrides["n"] > 20):
        logger.warning("忽略不合理的 n=%d", overrides["n"])
        del overrides["n"]
    return overrides


def merge_overrides(base: dict, overrides: dict) -> dict:
    """将参数覆盖合并到基础参数中。overrides 中的非空值覆盖 base。"""
    merged = dict(base)
    for key, val in overrides.items():
        if val is not None and val != "":
            merged[key] = val
    return merged

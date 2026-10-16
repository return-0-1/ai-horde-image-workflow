"""AI Horde 采样器名规范化 —— 只把服务端认的名字发出去。

为什么需要这层：`DPM++ 2M Karras` 这类 **A1111/WebUI 写法** AI Horde 不认，
`POST /generate/async` 会直接 **400**（`params.sampler_name ... is not one of [...]`），
整次出图白费。历史别名表里还残留了几个同样无效的名字（k_unipc / k_ddim /
k_plms / k_restart / k_dpmpp_2m_sde / k_dpmpp_3m_sde），所以「查表」不够，
必须**对着服务端的权威清单做校验**，认不出就回退默认，绝不把 400 打到 API 上。

权威清单来源（公开接口，无需鉴权，41 项，2026-10-16 抓取）：
    GET https://aihorde.net/api/v2/status/sampler_constraints
刷新方式：
    curl -s https://aihorde.net/api/v2/status/sampler_constraints
    → 取 samplers 的键替换下面的 HORDE_SAMPLERS
（新增采样器时清单落后只会导致「该名字被忽略并回退默认」，不会报错。）
"""

import re
import logging

logger = logging.getLogger(__name__)

# 服务端认可的采样器名（大小写敏感，DDIM / lcm 等原样）
HORDE_SAMPLERS = (
    "DDIM", "ddpm", "deis", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde",
    "dpmpp_2m_sde_heun", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_3m_sde",
    "dpmsolver", "er_sde", "euler_ancestral_cfg_pp", "euler_cfg_pp",
    "exp_heun_2_x0", "exp_heun_2_x0_sde", "gradient_estimation",
    "gradient_estimation_cfg_pp", "heunpp2", "ipndm", "ipndm_v",
    "k_dpm_2", "k_dpm_2_a", "k_dpm_adaptive", "k_dpm_fast", "k_dpmpp_2m",
    "k_dpmpp_2s_a", "k_dpmpp_sde", "k_euler", "k_euler_a", "k_heun",
    "k_lms", "lcm", "res_multistep", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "res_multistep_cfg_pp",
    "sa_solver", "sa_solver_pece", "seeds_2", "seeds_3", "uni_pc",
    "uni_pc_bh2",
)

# 出默认值：k_euler_a 是 Horde 推荐档（presentation_tier=recommended）
DEFAULT_SAMPLER = "k_euler_a"

# 小写 → 规范拼写（服务端大小写敏感，用户怎么打都要落回规范形）
_CANONICAL = {s.lower(): s for s in HORDE_SAMPLERS}
# 去掉下划线/空格的写法也认（k_euler_a / K EULER A / keulera 都算同一个）
_CANONICAL_SQUEEZED = {s.lower().replace("_", ""): s for s in HORDE_SAMPLERS}

# A1111 / WebUI 写法（以及中文习惯写法）→ Horde 名
A1111_ALIASES = {
    # Euler
    "euler": "k_euler",
    "euler a": "k_euler_a",
    "euler_a": "k_euler_a",
    "euler ancestral": "k_euler_a",
    "euler cfg ++": "euler_cfg_pp",
    "euler a cfg ++": "euler_ancestral_cfg_pp",
    # DPM++ 家族
    "dpm++": "k_dpmpp_2m",
    "dpm++ 2m": "k_dpmpp_2m",
    "dpm++ 2s": "k_dpmpp_2s_a",
    "dpm++ 2s a": "k_dpmpp_2s_a",
    "dpm++ 2 a": "k_dpmpp_2s_a",
    "dpm++ sde": "k_dpmpp_sde",
    "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpm++ 2m sde heun": "dpmpp_2m_sde_heun",
    "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm++ 2m cfg ++": "dpmpp_2m_cfg_pp",
    "dpm adaptive": "k_dpm_adaptive",
    "dpm++ 2m adaptive": "k_dpm_adaptive",
    "dpm fast": "k_dpm_fast",
    "dpm2": "k_dpm_2",
    "dpm2 a": "k_dpm_2_a",
    # 其它常见
    "ddim": "DDIM",
    "lms": "k_lms",
    "heun": "k_heun",
    "heun++2": "heunpp2",
    "unipc": "uni_pc",
    "uni pc": "uni_pc",
    "unipc bh2": "uni_pc_bh2",
    "lcm": "lcm",
    "deis": "deis",
    "ddpm": "ddpm",
    "res multistep": "res_multistep",
    "res multistep ancestral": "res_multistep_ancestral",
    "sa solver": "sa_solver",
    "sa solver pece": "sa_solver_pece",
    "seeds 2": "seeds_2",
    "seeds 3": "seeds_3",
    # 明确没有对应、故意不映射（会让调用方回退默认）：
    #   plms / restart —— Horde 无此采样器，硬映射成近似的只会产出不对的东西
}

_SPACE_RE = re.compile(r"\s+")
_KARRAS_SUFFIX_RE = re.compile(r"[\s_]*(karras|卡拉斯)$", re.IGNORECASE)


def normalize_sampler(name) -> str:
    """把用户/A1111/LLM 给的采样器名转成 AI Horde 内部名。

    Returns:
        服务端认可的内部名；**认不出时返回空字符串**（调用方应回退
        DEFAULT_SAMPLER 并记日志，而不是把原样字符发出去换来 400）。
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    raw = _SPACE_RE.sub(" ", name.strip())
    if raw in HORDE_SAMPLERS:                 # 已经是规范拼写（DDIM 等大小写敏感）
        return raw

    low = raw.lower().strip()
    spaced = low.replace("_", " ")
    squeezed = spaced.replace(" ", "")
    # 先按原样试，再试去掉 Karras 后缀的形（A1111 的 Karras 只是调度器写法，
    # Horde 的 k_* 采样器本身即 karras）
    forms = [low, spaced, squeezed]
    forms += [_KARRAS_SUFFIX_RE.sub("", f).strip() for f in forms]

    for form in forms:
        if not form:
            continue
        if form in _CANONICAL:
            return _CANONICAL[form]
        if form in A1111_ALIASES:
            return A1111_ALIASES[form]
        if form.replace(" ", "") in _CANONICAL_SQUEEZED:
            return _CANONICAL_SQUEEZED[form.replace(" ", "")]

    return ""


def validate_sampler(name, *, context: str = "") -> str:
    """normalize 的带日志版本：能认出来就用，认不出回退默认并告警。"""
    normalized = normalize_sampler(name)
    if normalized:
        return normalized
    if name:
        logger.warning("采样器 %r 不是 AI Horde 认可的名字，已回退默认 %s%s",
                       name, DEFAULT_SAMPLER, f"（{context}）" if context else "")
    return DEFAULT_SAMPLER

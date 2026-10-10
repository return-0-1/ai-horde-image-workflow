"""HTTP 出网工具 —— 项目的代理设置自己说了算，不受宿主/容器环境影响。

优先级：
  1. WORKFLOW_PROXY   —— 本项目专用（.env 里配置）
  2. AI_HORDE_PROXY   —— 兼容旧名
  3. HTTP_PROXY / HTTPS_PROXY / ALL_PROXY —— 仅作兜底（宿主环境可能顺带设置）

设为 none/off/direct/-/0/false 表示**强制直连**，忽略上面第 3 类通用变量。

为什么不用通用 HTTPS_PROXY 当主设定：python-dotenv 会把它写进整个进程环境，
连带 AstrBot 自身（httpx/openai）以及其他插件都改走代理 —— 宿主 VPN 一开一关，
bot 的对话就会跟着出问题。项目专用变量只有本项目的 HTTP 客户端读，隔离干净。
"""

import os
import logging
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

_PROJECT_KEYS = ("WORKFLOW_PROXY", "AI_HORDE_PROXY")
_GENERIC_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                 "ALL_PROXY", "all_proxy")
_DISABLE_VALUES = {"none", "off", "direct", "-", "0", "false", "no"}

# 直连 opener（显式空代理表，不会被系统代理/环境变量影响）
no_proxy_opener = build_opener(ProxyHandler({}))

_opener_cache: dict = {}


def resolve_proxy() -> str:
    """返回本次生效的代理地址；空字符串 = 直连。"""
    for key in _PROJECT_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return "" if val.lower() in _DISABLE_VALUES else val
    for key in _GENERIC_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return "" if val.lower() in _DISABLE_VALUES else val
    return ""


def get_opener():
    proxy = resolve_proxy()
    opener = _opener_cache.get(proxy)
    if opener is None:
        if proxy:
            logger.info("HTTP 出网走代理: %s", proxy)
            opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))
        else:
            opener = no_proxy_opener
        _opener_cache[proxy] = opener
    return opener


def get_openers() -> list:
    """本次请求的 opener 尝试顺序：设了代理就追加直连兜底。"""
    opener = get_opener()
    return [opener, no_proxy_opener] if resolve_proxy() else [opener]


def is_proxy_dead(err) -> bool:
    """区分「代理本身不可用」（VPN 没开/端口变了）与「目标站点慢」。

    连接被拒说明请求根本没发出去，此时连 POST 也可以安全回退直连。
    """
    reason = getattr(err, "reason", None)
    if isinstance(reason, ConnectionRefusedError):
        return True
    text = f"{err} {reason or ''}".lower()
    return "refused" in text or "10061" in text or "connect to proxy" in text


def open_with_fallback(req: Request, timeout: float):
    """依次尝试（代理 → 直连），返回响应对象。

    只适用于幂等请求（GET / 下载）。POST 请自行判断是否允许回退，
    否则「响应读超时但其实已提交」的情况下会重复提交。
    """
    openers = get_openers()
    last_err = None
    for i, opener in enumerate(openers):
        try:
            return opener.open(req, timeout=timeout)
        except HTTPError:
            raise  # HTTP 状态码错误与代理无关，交给调用方
        except (URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            if i + 1 < len(openers):
                logger.warning("代理请求失败: %s → 改用直连重试同一请求", e)
                continue
            raise
    raise last_err

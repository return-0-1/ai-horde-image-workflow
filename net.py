"""HTTP 出网工具 —— 项目的代理设置自己说了算，不受宿主/容器环境影响。

优先级：
  1. WORKFLOW_PROXY   —— 本项目专用（.env 里配置）
  2. AI_HORDE_PROXY   —— 兼容旧名
  3. HTTP_PROXY / HTTPS_PROXY / ALL_PROXY —— 仅作兜底（宿主环境可能顺带设置）

支持**候选列表**：用逗号/分号/空白分隔多个地址，按顺序尝试，谁先连通就记住谁
（宿主直跑用 127.0.0.1:20881，容器里跑用 host.docker.internal:20881 —— 同一行
写两个，宿主与容器共用一份 .env 即可）。候选全不通时回退直连。

设为 none/off/direct/-/0/false 表示**强制直连**，忽略上面第 3 类通用变量。

为什么不用通用 HTTPS_PROXY 当主设定：python-dotenv 会把它写进整个进程环境，
连带 AstrBot 自身（httpx/openai）以及其他插件都改走代理 —— 宿主 VPN 一开一关，
bot 的对话就会跟着出问题。项目专用变量只有本项目的 HTTP 客户端读，隔离干净。

为什么不把端口写死成常量：v2rayN 的 HTTP 入站是**自动生成**的（= socks 端口 +1，
见 v2rayN-Core/config.json），socks 端口一变（历史：10808 → 20880），写死的
http 端口就静默失效。候选列表 + 「谁通记谁」让换端口只需改 .env 一行。
"""

import os
import re
import logging
import threading
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

_PROJECT_KEYS = ("WORKFLOW_PROXY", "AI_HORDE_PROXY")
_GENERIC_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                 "ALL_PROXY", "all_proxy")
_DISABLE_VALUES = {"none", "off", "direct", "-", "0", "false", "no"}
_SEPARATORS = re.compile(r"[,;\s]+")

# 直连 opener（显式空代理表，不会被系统代理/环境变量影响）
no_proxy_opener = build_opener(ProxyHandler({}))

_opener_cache: dict = {}
_lock = threading.RLock()

# 上一次真正连通的候选代理；为空 = 还没定/全部失败
_sticky_proxy: str = ""
# 「候选全挂、已降级直连」的告警只打一次，避免刷日志
_direct_fallback_warned = False


# ----------------------------------------------------------------------
# 配置解析
# ----------------------------------------------------------------------

def _raw_proxy_setting() -> str:
    for key in _PROJECT_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    for key in _GENERIC_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return ""


def resolve_proxies() -> list:
    """返回候选代理列表（保持 .env 里的书写顺序）；空列表 = 直连。

    列表里剔除 none/off/direct 这类禁用值，并去重（大小写不敏感）。
    """
    raw = _raw_proxy_setting()
    if not raw or raw.lower() in _DISABLE_VALUES:
        return []

    proxies = []
    seen = set()
    for part in _SEPARATORS.split(raw):
        p = part.strip()
        if not p or p.lower() in _DISABLE_VALUES:
            continue
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        proxies.append(p)
    return proxies


def resolve_proxy() -> str:
    """本次生效的代理地址（优先返回上次连通过的那个）；空字符串 = 直连。"""
    proxies = resolve_proxies()
    if not proxies:
        return ""
    with _lock:
        sticky = _sticky_proxy
    if sticky and sticky in proxies:
        return sticky
    return proxies[0]


def describe_chain() -> str:
    """给日志用的一句话描述：本次的尝试顺序。"""
    chain = _chain()
    labels = [p or "直连" for p, _ in chain]
    return " → ".join(labels)


# ----------------------------------------------------------------------
# opener 构建
# ----------------------------------------------------------------------

def _build_proxy_opener(proxy: str):
    """构建走 proxy 的 opener。单独成函数，便于测试替换。"""
    logger.info("HTTP 出网走代理: %s", proxy)
    return build_opener(ProxyHandler({"http": proxy, "https": proxy}))


def _opener_for(proxy: str):
    with _lock:
        opener = _opener_cache.get(proxy)
        if opener is None:
            opener = _build_proxy_opener(proxy) if proxy else no_proxy_opener
            _opener_cache[proxy] = opener
    return _ChainedOpener(proxy, opener)


def get_opener():
    """当前唯一会用的 opener（优先上次成功的候选）。"""
    return _opener_for(resolve_proxy())


class _ChainedOpener:
    """opener 包装：透传 .open()，顺带记住「哪个候选真的能用」。

    调用方（AIHordeClient 等）不需要知道候选顺序，只要照着 get_chain() 依次
    尝试即可 —— 谁成功，下一次请求它就会排到最前。
    """

    __slots__ = ("proxy", "_opener")

    def __init__(self, proxy: str, opener):
        self.proxy = proxy
        self._opener = opener

    def open(self, req, timeout=None):
        try:
            resp = self._opener.open(req, timeout=timeout)
        except HTTPError:
            # 能拿到 HTTP 状态码 = 链路通了（4xx/5xx 是服务端的事），代理是好的
            note_proxy_ok(self.proxy)
            raise
        except (URLError, TimeoutError, ConnectionError, OSError) as e:
            # 只有「代理本身连不上」才把候选踢掉；目标站点慢/抖不算候选的错
            if self.proxy and is_proxy_dead(e):
                note_proxy_bad(self.proxy)
            raise
        note_proxy_ok(self.proxy)
        return resp

    def __getattr__(self, name):
        return getattr(self._opener, name)


def note_proxy_ok(proxy: str) -> None:
    """标记某个候选可用（空串 = 直连，忽略）。"""
    global _sticky_proxy, _direct_fallback_warned
    if not proxy:
        return
    with _lock:
        if _sticky_proxy != proxy:
            if _sticky_proxy:
                logger.info("出网代理切换: %s → %s", _sticky_proxy, proxy)
            _sticky_proxy = proxy
        _direct_fallback_warned = False


def note_proxy_bad(proxy: str) -> None:
    """标记某个候选不可用；若它正是当前 sticky，则清空以便重新探测。"""
    global _sticky_proxy
    if not proxy:
        return
    with _lock:
        if _sticky_proxy == proxy:
            _sticky_proxy = ""
            logger.warning("出网代理 %s 已失效，下次请求重新探测候选: %s",
                           proxy, resolve_proxies() or "(无，将直连)")


def reset_network_state() -> None:
    """清空 opener 缓存与 sticky 记录。

    测试、以及运行时改了 .env 后需要重新探测时用。
    """
    global _sticky_proxy, _direct_fallback_warned
    with _lock:
        _opener_cache.clear()
        _sticky_proxy = ""
        _direct_fallback_warned = False


def _chain() -> list:
    """本次请求的尝试顺序：[(proxy_label, opener), ...]。

    候选代理在上（上次成功的排最前），直连永远垫底当兜底。
    """
    proxies = resolve_proxies()
    if not proxies:
        return [("", _opener_for(""))]

    with _lock:
        sticky = _sticky_proxy
    ordered = ([sticky] if sticky in proxies else []) + [p for p in proxies if p != sticky]
    return [(p, _opener_for(p)) for p in ordered] + [("", _opener_for(""))]


def get_chain() -> list:
    """同 _chain()，公开给业务客户端使用（带候选名，便于日志区分）。"""
    return _chain()


def get_openers() -> list:
    """兼容旧调用：只返回 opener 对象（顺序与 get_chain() 一致）。"""
    return [opener for _, opener in _chain()]


# ----------------------------------------------------------------------
# 错误分类 / 请求复制
# ----------------------------------------------------------------------

def is_proxy_dead(err) -> bool:
    """区分「代理本身不可用」（VPN 没开/端口变了）与「目标站点慢」。

    连接被拒说明请求根本没发出去，此时连 POST 也可以安全回退直连。
    """
    reason = getattr(err, "reason", None)
    if isinstance(reason, ConnectionRefusedError):
        return True
    text = f"{err} {reason or ''}".lower()
    return "refused" in text or "10061" in text or "connect to proxy" in text


def clone_request(req: Request) -> Request:
    """复制一个干净的 Request（代理回退直连时必须换新对象）。

    urllib 的 ProxyHandler 在连接之前就会**就地改写** Request —— proxy_open()
    调用 req.set_proxy()，把 req.host 换成代理地址、selector 换成完整 URL。
    所以「代理失败后复用同一个 Request 改直连重试」实际又打了一次代理，
    报的还是同一个连接被拒 —— 必须用新对象，回退才真正生效。
    """
    return Request(
        req.full_url,
        data=req.data,
        headers=dict(req.headers),
        method=req.get_method(),
    )


def open_with_fallback(req: Request, timeout: float):
    """依次尝试候选代理（→ 直连），返回响应对象。

    只适用于幂等请求（GET / 下载）。POST 请自行判断是否允许回退，
    否则「响应读超时但其实已提交」的情况下会重复提交。
    """
    global _direct_fallback_warned
    chain = _chain()
    last_err = None
    for idx, (proxy, opener) in enumerate(chain):
        try:
            # 每个 opener 一个新 Request：前一个（代理）已把它改写过了
            return opener.open(clone_request(req), timeout=timeout)
        except HTTPError:
            raise  # HTTP 状态码错误与代理无关，交给调用方
        except (URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            if idx + 1 < len(chain):
                nxt = chain[idx + 1][0] or "直连"
                logger.warning("出网候选 %s 失败: %s → 改用 %s 重试同一请求",
                               proxy or "直连", e, nxt)
                if proxy and not chain[idx + 1][0] and not _direct_fallback_warned:
                    _direct_fallback_warned = True
                    logger.warning("所有候选代理均不可用，本次已降级直连"
                                   "（直连路径在国内常被劣化，注意成功率下降）")
                continue
            raise
    raise last_err

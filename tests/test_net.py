"""net.py 候选代理链 + AIHordeClient 回退策略的单元测试。

全部离线：用假的 opener 喂脚本化的异常序列，不碰真网络、不消耗 kudos。
运行：  python -m unittest discover -s tests -v
"""

import os
import sys
import email.message
import unittest
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import net  # noqa: E402


# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------

class FakeResponse:
    def __init__(self, body=b'{"ok": true}', status=200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """按脚本依次抛出异常或返回响应；记录每次拿到的请求。"""

    def __init__(self, label, script):
        self.label = label
        self.script = list(script)
        self.seen = []          # 每次 open 拿到的 Request 副本
        self.calls = 0

    def open(self, req, timeout=None):
        self.calls += 1
        # 记录调用时的 host —— 被 urllib 就地改写过的 Request 这里能看到
        self.seen.append({"host": req.host, "url": req.full_url})
        action = self.script.pop(0) if self.script else FakeResponse()
        if isinstance(action, Exception):
            raise action
        return action


def refuse():
    return URLError(ConnectionRefusedError(10061, "由于目标计算机积极拒绝，无法连接。"))


def read_timeout():
    return URLError(SocketTimeout("The read operation timed out"))


def ssl_timeout():
    return URLError(SocketTimeout("_ssl.c:993: The handshake operation timed out"))


def http_429():
    return HTTPError("https://aihorde.net/api/v2/x", 429, "Too Many Requests",
                     email.message.Message(), None)


class _Fixture:
    """公共夹具：清环境变量、重置 net 模块状态、替换 opener 工厂。

    不继承 TestCase —— 否则子类会把父类的测试重复跑一遍。
    """

    def setUp(self):
        self._saved_env = {k: os.environ.get(k) for k in
                           ("WORKFLOW_PROXY", "AI_HORDE_PROXY", "HTTPS_PROXY",
                            "https_proxy", "HTTP_PROXY", "http_proxy",
                            "ALL_PROXY", "all_proxy")}
        for k in self._saved_env:
            os.environ.pop(k, None)

        self._saved_build = net._build_proxy_opener
        self._saved_direct = net.no_proxy_opener
        self.scripts = {}
        self.fakes = {}

        net._build_proxy_opener = self.fake
        net.no_proxy_opener = self.fake("")
        net.reset_network_state()

    def fake(self, proxy):
        """取（必要时建）某个候选对应的假 opener。"""
        if proxy not in self.fakes:
            self.fakes[proxy] = FakeOpener(proxy, self.scripts.get(proxy, [FakeResponse()]))
        return self.fakes[proxy]

    def tearDown(self):
        net._build_proxy_opener = self._saved_build
        net.no_proxy_opener = self._saved_direct
        net.reset_network_state()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class NetTestCase(_Fixture, unittest.TestCase):
    """net.py 的配置解析、候选链与回退。"""

    # ---- 配置解析 -------------------------------------------------

    def test_no_setting_means_direct(self):
        self.assertEqual(net.resolve_proxies(), [])
        self.assertEqual(net.resolve_proxy(), "")
        self.assertEqual(len(net.get_chain()), 1)
        self.assertEqual(net.get_chain()[0][0], "")

    def test_disable_values_mean_direct(self):
        for val in ("none", "NONE", "direct", "off", "0", "false", "-"):
            os.environ["WORKFLOW_PROXY"] = val
            self.assertEqual(net.resolve_proxies(), [], val)

    def test_candidate_list_order_and_dedupe(self):
        os.environ["WORKFLOW_PROXY"] = ("http://127.0.0.1:20881, "
                                        "http://host.docker.internal:20881;"
                                        "http://127.0.0.1:20881")
        self.assertEqual(net.resolve_proxies(),
                         ["http://127.0.0.1:20881", "http://host.docker.internal:20881"])

    def test_project_key_beats_generic_key(self):
        os.environ["HTTPS_PROXY"] = "http://generic:10809"
        self.assertEqual(net.resolve_proxy(), "http://generic:10809")
        os.environ["WORKFLOW_PROXY"] = "http://mine:20881"
        self.assertEqual(net.resolve_proxy(), "http://mine:20881")

    # ---- 候选链顺序 / sticky --------------------------------------

    def test_chain_order_is_config_order_then_direct(self):
        os.environ["WORKFLOW_PROXY"] = "http://a:1,http://b:2"
        self.assertEqual([p for p, _ in net.get_chain()],
                         ["http://a:1", "http://b:2", ""])

    def test_sticky_promotes_working_candidate_to_front(self):
        os.environ["WORKFLOW_PROXY"] = "http://a:1,http://b:2"
        net.note_proxy_ok("http://b:2")
        self.assertEqual([p for p, _ in net.get_chain()],
                         ["http://b:2", "http://a:1", ""])
        self.assertEqual(net.resolve_proxy(), "http://b:2")

    def test_dead_sticky_is_cleared_and_retried_from_config_order(self):
        os.environ["WORKFLOW_PROXY"] = "http://a:1,http://b:2"
        net.note_proxy_ok("http://b:2")
        net.note_proxy_bad("http://b:2")
        self.assertEqual([p for p, _ in net.get_chain()],
                         ["http://a:1", "http://b:2", ""])

    # ---- 回退 ------------------------------------------------------

    def test_open_with_fallback_rebuilds_request_per_candidate(self):
        """回归：代理 opener 会就地改写 Request，复用同一对象会让回退失效。"""
        os.environ["WORKFLOW_PROXY"] = "http://127.0.0.1:20881"

        def proxied(req, timeout=None):
            proxied.seen.append(req.host)
            # 复刻 urllib ProxyHandler.proxy_open 的就地改写
            req.set_proxy("127.0.0.1:20881", "http")
            raise refuse()

        proxied.seen = []
        self.fake("http://127.0.0.1:20881").open = proxied

        req = Request("https://aihorde.net/api/v2/status/heartbeat")
        with net.open_with_fallback(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)

        self.assertEqual(proxied.seen, ["aihorde.net"])
        # 第二个候选（直连）必须拿到干净的 host，而不是被改写成代理地址
        self.assertEqual(self.fakes[""].seen[-1]["host"], "aihorde.net")

    def test_open_with_fallback_tries_all_candidates_in_order(self):
        os.environ["WORKFLOW_PROXY"] = "http://a:1,http://b:2"
        self.scripts["http://a:1"] = [refuse()]
        self.scripts["http://b:2"] = [FakeResponse()]
        net._opener_cache.clear()  # 让脚本生效（opener 在首次取用时构建）

        with net.open_with_fallback(Request("https://x/y"), timeout=5) as resp:
            self.assertEqual(resp.status, 200)
        self.assertEqual(self.fakes["http://a:1"].calls, 1)
        self.assertEqual(self.fakes["http://b:2"].calls, 1)
        self.assertEqual(self.fakes[""].calls, 0)      # 没用到直连
        self.assertEqual(net.resolve_proxy(), "http://b:2")   # b 被记为可用

    def test_open_with_fallback_raises_when_everything_fails(self):
        os.environ["WORKFLOW_PROXY"] = "http://a:1"
        self.scripts["http://a:1"] = [refuse()]
        # 直连也挂（模拟两条路都不通）
        net.no_proxy_opener.open = lambda req, timeout=None: (_ for _ in ()).throw(refuse())
        net._opener_cache.clear()
        with self.assertRaises(URLError):
            net.open_with_fallback(Request("https://x/y"), timeout=5)

    def test_http_error_is_not_a_proxy_failure(self):
        """代理能返回 429 = 链路是通的，不该切换候选。"""
        os.environ["WORKFLOW_PROXY"] = "http://a:1"
        self.scripts["http://a:1"] = [http_429()]
        net._opener_cache.clear()
        with self.assertRaises(HTTPError):
            net.open_with_fallback(Request("https://x/y"), timeout=5)
        self.assertEqual(self.fakes[""].calls, 0)
        self.assertEqual(net.resolve_proxy(), "http://a:1")

    # ---- 错误分类 --------------------------------------------------

    def test_is_proxy_dead_classification(self):
        self.assertTrue(net.is_proxy_dead(refuse()))
        self.assertTrue(net.is_proxy_dead(URLError("WinError 10061")))
        # 目标站点慢/抖动不是「代理死了」—— POST 靠这个区分才敢不敢回退
        self.assertFalse(net.is_proxy_dead(read_timeout()))
        self.assertFalse(net.is_proxy_dead(ssl_timeout()))


class ClientFallbackTestCase(_Fixture, unittest.TestCase):
    """AIHordeClient._request 在候选链上的行为（GET 可回退，POST 有约束）。"""

    def _client(self, max_retries=0):
        import ai_horde_client
        c = ai_horde_client.AIHordeClient.__new__(ai_horde_client.AIHordeClient)
        c.api_url = "https://aihorde.net/api/v2"
        c.api_key = ""
        c.timeout = 5
        c.max_retries = max_retries
        return ai_horde_client, c

    def test_get_falls_back_to_next_candidate(self):
        mod, client = self._client()
        bad = FakeOpener("http://a:1", [refuse()])
        good = FakeOpener("", [FakeResponse(b'{"message": "OK"}')])
        mod._get_chain = lambda: [("http://a:1", net._ChainedOpener("http://a:1", bad)),
                                  ("", net._ChainedOpener("", good))]
        self.assertEqual(client._request("GET", "/status/heartbeat"), {"message": "OK"})
        self.assertEqual(bad.calls, 1)
        self.assertEqual(good.calls, 1)

    def test_post_does_not_fall_back_on_read_timeout(self):
        """读超时 = 结果未知，回退会重复提交重复扣 kudos。"""
        mod, client = self._client()
        bad = FakeOpener("http://a:1", [read_timeout()])
        good = FakeOpener("", [FakeResponse()])
        mod._get_chain = lambda: [("http://a:1", net._ChainedOpener("http://a:1", bad)),
                                  ("", net._ChainedOpener("", good))]
        with self.assertRaises(RuntimeError):
            client._request("POST", "/generate/async", {"prompt": "x"})
        self.assertEqual(bad.calls, 1)
        self.assertEqual(good.calls, 0)          # 没有偷偷直连重发

    def test_post_falls_back_only_when_request_never_left(self):
        """连接被拒 = 请求没发出去，回退安全。"""
        mod, client = self._client()
        bad = FakeOpener("http://a:1", [refuse()])
        good = FakeOpener("", [FakeResponse(b'{"id": "abc"}')])
        mod._get_chain = lambda: [("http://a:1", net._ChainedOpener("http://a:1", bad)),
                                  ("", net._ChainedOpener("", good))]
        self.assertEqual(client._request("POST", "/generate/async", {"prompt": "x"}),
                         {"id": "abc"})
        self.assertEqual(good.calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

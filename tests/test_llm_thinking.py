"""llm_client 思考模式开关（thinking）的单元测试。

背景：deepseek-flash 默认开启思考，reasoning 与最终答案共用 max_tokens，
思考一次可吃掉 2000-3300 token，答案被截断后 json.loads 报
"Unterminated string"。结构化任务必须显式关闭思考。

全部离线：替换 llm_client 模块级 opener，只断言请求体，不碰真网络。
运行：  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_client  # noqa: E402
import params_parser  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from lora_searcher import LoraSearcher  # noqa: E402

BASE_CFG = {
    "llm": {
        "api_base": "https://example.invalid/v1",
        "api_key": "test-key",
        "model": "deepseek-flash",
        "max_tokens": 4096,
    }
}


class FakeResponse:
    """假 HTTP 响应：body 是一份最小的 ChatCompletion 返回。"""

    def __init__(self, content: str):
        payload = {"choices": [{"message": {"content": content}}]}
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class CapturingOpener:
    """假 opener：记录每个请求体，返回脚本化的 content。"""

    def __init__(self, content: str = "{}"):
        self.content = content
        self.bodies = []

    def open(self, req, timeout=None):
        self.bodies.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse(self.content)

    @property
    def last(self) -> dict:
        return self.bodies[-1]


class ThinkingSwitchTestCase(unittest.TestCase):
    """llm_client.chat 的 thinking / max_tokens 参数落到请求体上的行为。"""

    def setUp(self):
        self._orig = llm_client._no_proxy_opener

    def tearDown(self):
        llm_client._no_proxy_opener = self._orig

    def _client(self, content="{}", **llm_overrides):
        cfg = {"llm": dict(BASE_CFG["llm"], **llm_overrides)}
        client = LLMClient(cfg)
        opener = CapturingOpener(content)
        llm_client._no_proxy_opener = opener
        return client, opener

    def test_thinking_false_sends_disabled(self):
        client, opener = self._client()
        client.chat("系统", "用户", json_mode=True, thinking=False)
        self.assertEqual(opener.last["thinking"], {"type": "disabled"})
        self.assertEqual(opener.last["response_format"], {"type": "json_object"})

    def test_thinking_true_sends_enabled(self):
        client, opener = self._client()
        client.chat("系统", "用户", thinking=True)
        self.assertEqual(opener.last["thinking"], {"type": "enabled"})

    def test_default_sends_no_thinking_field(self):
        """config 未配 thinking 时不下发该字段 —— 交给服务端默认（=开启）。"""
        client, opener = self._client()
        client.chat("系统", "用户")
        self.assertNotIn("thinking", opener.last)

    def test_config_thinking_disabled_string(self):
        """config.yaml 里写 disabled 时，所有调用默认关闭思考。"""
        client, opener = self._client(thinking="disabled")
        client.chat("系统", "用户")
        self.assertEqual(opener.last["thinking"], {"type": "disabled"})

    def test_call_site_overrides_config(self):
        client, opener = self._client(thinking="enabled")
        client.chat("系统", "用户", thinking=False)
        self.assertEqual(opener.last["thinking"], {"type": "disabled"})

    def test_max_tokens_override(self):
        client, opener = self._client()
        client.chat("系统", "用户")
        self.assertEqual(opener.last["max_tokens"], 4096)
        client.chat("系统", "用户", max_tokens=8192)
        self.assertEqual(opener.last["max_tokens"], 8192)


class StructuredCallSitesTestCase(unittest.TestCase):
    """结构化任务的三个调用点必须真的把 thinking=False 传下去。"""

    def setUp(self):
        self._orig = llm_client._no_proxy_opener

    def tearDown(self):
        llm_client._no_proxy_opener = self._orig

    def _wire(self, content: str):
        opener = CapturingOpener(content)
        llm_client._no_proxy_opener = opener
        return LLMClient(BASE_CFG), opener

    def test_lora_review_disables_thinking(self):
        payload = json.dumps({"scores": [{"index": 0, "score": 9, "reason": "主题吻合"}]})
        llm, opener = self._wire(payload)
        searcher = LoraSearcher(BASE_CFG, llm, None)
        cands = [{"version_id": "1", "name": "Orange Cat", "base_model": "SDXL",
                  "trigger_words": ["orange cat"], "keywords": ["cat"]}]

        scored = searcher._llm_review("一只橘猫在窗台晒太阳", "SDXL", cands)

        self.assertEqual(opener.last["thinking"], {"type": "disabled"})
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["_score"], 9)

    def test_query_extraction_disables_thinking(self):
        llm, opener = self._wire("orange cat\nsunlight")
        searcher = LoraSearcher(BASE_CFG, llm, None)

        queries = searcher._extract_query("一只橘猫在窗台晒太阳")

        self.assertEqual(opener.last["thinking"], {"type": "disabled"})
        self.assertNotIn("thinking", opener.last.get("response_format", {}))
        self.assertEqual(queries, ["orange cat", "sunlight"])

    def test_params_fallback_disables_thinking(self):
        llm, opener = self._wire(json.dumps({"steps": 30}))
        # 文本用正则提不出参数，才会走 LLM 兜底
        out = params_parser.parse("来一张随便什么图", llm_client=llm)
        self.assertEqual(opener.last["thinking"], {"type": "disabled"})
        self.assertEqual(out.get("steps"), 30)


if __name__ == "__main__":
    unittest.main()

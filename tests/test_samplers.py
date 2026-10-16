"""采样器规范化 + 400 报错可读化的单元测试。

背景（真实事故）：用户说「DPM++ 2M Karras」，LLM 兜底把它原样写进参数，
POST /generate/async 被 AI Horde 400 拒掉，整次出图失败 —— 而日志里只有
"HTTP Error 400: Bad Request"，看不出原因。

运行：  python -m unittest discover -s tests -v
"""

import json
import io
import os
import sys
import unittest
from urllib.error import HTTPError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import samplers  # noqa: E402
import params_parser  # noqa: E402
import ai_horde_client  # noqa: E402
from test_net import _Fixture, FakeOpener, FakeResponse  # noqa: E402


class SamplerNormalizeTestCase(unittest.TestCase):
    """A1111 / 口语写法 → Horde 内部名。"""

    def test_a1111_forms_map_to_horde_names(self):
        cases = {
            "DPM++ 2M Karras": "k_dpmpp_2m",       # 本次事故的元凶
            "DPM++ 2M": "k_dpmpp_2m",
            "dpm++ 2m karras": "k_dpmpp_2m",
            "DPM++ 2S a Karras": "k_dpmpp_2s_a",
            "DPM++ SDE Karras": "k_dpmpp_sde",
            "DPM++ 2M SDE Karras": "dpmpp_2m_sde",
            "DPM++ 3M SDE": "dpmpp_3m_sde",
            "DPM++ 2M Adaptive": "k_dpm_adaptive",
            "Euler": "k_euler",
            "Euler a": "k_euler_a",
            "Euler Ancestral": "k_euler_a",
            "LMS": "k_lms",
            "LMS Karras": "k_lms",
            "Heun": "k_heun",
            "DDIM": "DDIM",
            "UniPC": "uni_pc",
            "LCM": "lcm",
        }
        for raw, expected in cases.items():
            self.assertEqual(samplers.normalize_sampler(raw), expected, raw)

    def test_internal_names_keep_server_spelling(self):
        # 服务端大小写敏感：DDIM / lcm 是规范拼写，k_* 一律小写
        for name in samplers.HORDE_SAMPLERS:
            self.assertEqual(samplers.normalize_sampler(name), name, name)
            self.assertEqual(samplers.normalize_sampler(name.upper()), name, name)
            self.assertEqual(samplers.normalize_sampler(name.replace("_", " ")), name, name)

    def test_unknown_names_return_empty(self):
        # plms / restart 在 Horde 里没有对应；宁可回退默认也不发出去换 400
        for bad in ["plms", "restart", "k_ddim", "k_unipc", "k_plms",
                    "k_dpmpp_2m_sde", "k_dpmpp_3m_sde", "bogus", "", None, "   "]:
            self.assertEqual(samplers.normalize_sampler(bad), "", repr(bad))

    def test_validate_falls_back_to_default(self):
        self.assertEqual(samplers.validate_sampler("bogus"), samplers.DEFAULT_SAMPLER)
        self.assertEqual(samplers.validate_sampler("DPM++ 2M Karras"), "k_dpmpp_2m")
        self.assertEqual(samplers.validate_sampler(None), samplers.DEFAULT_SAMPLER)
        self.assertIn(samplers.DEFAULT_SAMPLER, samplers.HORDE_SAMPLERS)


class SamplerTableHygieneTestCase(unittest.TestCase):
    """防止「别名表里塞了服务端不认的名字」这类事故复发。"""

    def test_every_alias_target_is_accepted_by_horde(self):
        accepted = set(samplers.HORDE_SAMPLERS)
        bad = {alias: target for alias, target in samplers.A1111_ALIASES.items()
               if target not in accepted}
        self.assertEqual(bad, {}, f"这些别名指向了 Horde 不认的采样器: {bad}")

    def test_params_parser_alias_table_is_clean(self):
        """params_parser 自己的表（历史遗留）也必须只指向有效名字。"""
        accepted = set(samplers.HORDE_SAMPLERS)
        bad = {alias: target for alias, target in params_parser.SAMPLER_ALIASES.items()
               if target not in accepted}
        self.assertEqual(bad, {}, f"params_parser.SAMPLER_ALIASES 有失效目标: {bad}")


class ParamsParserSamplerTestCase(unittest.TestCase):
    def test_explicit_sampler_phrase_is_canonicalized(self):
        self.assertEqual(params_parser.parse("采样器用 k_euler_a"), {"sampler": "k_euler_a"})
        self.assertEqual(params_parser.parse("sampler: Euler a"), {"sampler": "k_euler_a"})

    def test_unrecognized_sampler_is_dropped(self):
        out = params_parser.parse("采样器用 plms, 步数 30")
        self.assertNotIn("sampler", out)          # 丢弃，而不是把 400 交给服务端
        self.assertEqual(out.get("steps"), 30)

    def test_llm_fallback_value_is_validated(self):
        """LLM 兜底写回 A1111 写法 → 必须被规范化/拦掉。"""
        class FakeLLM:
            def chat(self, system, text, json_mode=True, **kwargs):
                return json.dumps({"sampler": "DPM++ 2M Karras", "steps": 30})

        # 文本用正则提不出参数，才会走 LLM 兜底
        out = params_parser.parse("来一张随便什么图", llm_client=FakeLLM())
        self.assertEqual(out.get("sampler"), "k_dpmpp_2m")

        class BadLLM:
            def chat(self, system, text, json_mode=True, **kwargs):
                return json.dumps({"sampler": "完全没听过的采样器"})

        out2 = params_parser.parse("来一张随便什么图", llm_client=BadLLM())
        self.assertNotIn("sampler", out2)


class BuildPayloadGateTestCase(unittest.TestCase):
    """最后一道闸：payload 里绝不会出现服务端不认的采样器。"""

    def _client(self, defaults=None):
        c = ai_horde_client.AIHordeClient.__new__(ai_horde_client.AIHordeClient)
        c.defaults = defaults or {}
        return c

    def test_invalid_sampler_in_params_falls_back_to_default(self):
        payload = self._client()._build_payload({"prompt": "x", "sampler": "DPM++ 2M Karras"})
        self.assertEqual(payload["params"]["sampler_name"], "k_dpmpp_2m")

    def test_invalid_default_from_config_falls_back(self):
        c = self._client(defaults={"sampler": "k_unipc"})   # 旧配置里的失效名字
        self.assertEqual(c._build_payload({"prompt": "x"})["params"]["sampler_name"],
                         samplers.DEFAULT_SAMPLER)

    def test_valid_sampler_passes_through(self):
        c = self._client(defaults={"sampler": "DDIM"})
        self.assertEqual(c._build_payload({"prompt": "x"})["params"]["sampler_name"], "DDIM")


class HttpErrorDetailTestCase(_Fixture, unittest.TestCase):
    """400 的响应体必须进日志和异常消息，否则无法定位是哪个参数。"""

    BODY = json.dumps({"errors": {"params.sampler_name":
                                  "'DPM++ 2M Karras' is not one of ['k_euler', 'DDIM']"}}).encode()

    def _400(self):
        return HTTPError("https://aihorde.net/api/v2/generate/async", 400, "Bad Request",
                         {}, io.BytesIO(self.BODY))

    def test_detail_extracts_errors_field(self):
        detail = ai_horde_client._http_error_detail(self._400())
        self.assertIn("params.sampler_name", detail)
        self.assertIn("is not one of", detail)

    def test_detail_survives_non_json_body(self):
        e = HTTPError("u", 502, "Bad Gateway", {}, io.BytesIO(b"<html>oops</html>"))
        self.assertIn("oops", ai_horde_client._http_error_detail(e))

    def test_request_error_message_carries_server_reason(self):
        mod, client = (ai_horde_client,
                       ai_horde_client.AIHordeClient.__new__(ai_horde_client.AIHordeClient))
        client.api_url = "https://aihorde.net/api/v2"
        client.api_key = ""
        client.timeout = 5
        client.max_retries = 0
        bad = FakeOpener("http://a:1", [self._400()])
        saved = mod._get_chain
        mod._get_chain = lambda: [("http://a:1", _wrap(bad))]
        try:
            with self.assertRaises(RuntimeError) as ctx:
                client._request("POST", "/generate/async", {"prompt": "x"})
        finally:
            mod._get_chain = saved
        msg = str(ctx.exception)
        self.assertIn("HTTP 400", msg)
        self.assertIn("is not one of", msg)       # 直接能看到服务端说的原因


def _wrap(opener):
    import net
    return net._ChainedOpener(getattr(opener, "label", ""), opener)


if __name__ == "__main__":
    unittest.main(verbosity=2)

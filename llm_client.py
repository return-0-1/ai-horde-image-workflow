"""
LLM 客户端抽象层 — 支持 OpenAI 兼容接口。

用于：
  1. 场景提取（从用户文本中提取生成目标）
  2. 提示词生成（根据场景 + LoRA 信息生成最终 Prompt）
"""

import os
import json
import logging
from typing import Optional
from urllib.request import Request, urlopen, ProxyHandler, build_opener
from urllib.error import URLError

# 绕过系统代理直连 LLM API（避免 v2rayN 干扰认证）
_no_proxy_opener = build_opener(ProxyHandler({}))

logger = logging.getLogger(__name__)


def _read_env_key() -> str:
    """直接从项目 .env 文件读取 OPENAI_API_KEY（load_dotenv 失效时的兜底）。"""
    import re as _re
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(_env_path):
        return ""
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _m = _re.match(r"^\s*OPENAI_API_KEY\s*=\s*(.+?)\s*$", _line)
                if _m:
                    return _m.group(1).strip()
    except Exception:
        pass
    return ""


class LLMClient:
    """OpenAI 兼容的 LLM 客户端。"""

    def __init__(self, config: dict):
        cfg = config.get("llm", {})
        self.api_base = cfg.get("api_base", "https://api.openai.com/v1").rstrip("/")
        self.api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "") or _read_env_key()
        self.model = cfg.get("model", "gpt-4o")
        self.temperature = cfg.get("temperature", 0.7)
        self.max_tokens = cfg.get("max_tokens", 2048)
        # 思考模式默认值（config.yaml 的 llm.thinking）：enabled / disabled / 留空=不发该字段。
        # deepseek-flash 服务端默认开启思考，而 reasoning 与答案共用 max_tokens 预算，
        # 所以结构化任务（评分/抽取）由调用方显式传 thinking=False，不依赖这里的默认值。
        self.thinking = cfg.get("thinking")

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def chat(self, system_prompt: str, user_message: str, json_mode: bool = False,
             thinking: Optional[bool] = None,
             max_tokens: Optional[int] = None) -> str:
        """发送 ChatCompletion 请求，返回模型回复文本。

        Args:
            system_prompt: 系统提示词。
            user_message:  用户消息。
            json_mode:     是否要求 JSON 格式输出。
            thinking:      思考模式开关。None = 用 config 的 llm.thinking（默认不发送，
                           交给服务端，而 deepseek-flash 的服务端默认是「开启」）；
                           False = 本次请求关闭思考。reasoning 与最终答案**共用**
                           max_tokens 预算，思考一次可烧掉 2000-3300 token，
                           答案会被截断成非法 JSON —— 结构化任务（评分/抽取）一律传 False。
                           True = 强制开启。
            max_tokens:    本次请求的输出上限，覆盖 config 的 llm.max_tokens。

        Returns:
            模型回复的文本内容。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self._call_api(messages, json_mode=json_mode,
                              thinking=thinking, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    @staticmethod
    def _thinking_flag(value) -> Optional[bool]:
        """把 True/False/'enabled'/'disabled'/'' 归一成 bool 或 None（None = 不发该字段）。"""
        if value is None or isinstance(value, bool):
            return value
        v = str(value).strip().lower()
        if v in ("disabled", "off", "false", "no", "0"):
            return False
        if v in ("enabled", "on", "true", "yes", "1"):
            return True
        return None

    def _call_api(self, messages: list, json_mode: bool = False,
                  thinking: Optional[bool] = None,
                  max_tokens: Optional[int] = None) -> str:
        """调用 OpenAI Chat Completion API。"""
        url = f"{self.api_base}/chat/completions"
        limit = self.max_tokens if max_tokens is None else max_tokens
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": limit,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        flag = self._thinking_flag(self.thinking if thinking is None else thinking)
        if flag is not None:
            body["thinking"] = {"type": "enabled" if flag else "disabled"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        logger.info("LLM 请求 → model=%s, messages=%d, max_tokens=%d, thinking=%s",
                    self.model, len(messages), limit,
                    body.get("thinking", {}).get("type", "服务端默认"))
        logger.debug("LLM system_prompt 前 200 字符: %s...", messages[0]["content"][:200])

        try:
            req = Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
            with _no_proxy_opener.open(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            logger.error("LLM API 调用失败: %s", e)
            raise RuntimeError(f"LLM API 不可用: {e}") from e

        content = data["choices"][0]["message"]["content"]
        logger.debug("LLM 响应（前 300 字符）: %s...", content[:300])
        return content

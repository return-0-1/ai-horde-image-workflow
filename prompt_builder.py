"""
提示词构建器 — 整合专业提示词模板，生成最终的生图 Prompt。

两阶段工作:
  阶段1: 场景提取 — 从用户任意文本中提取生成目标
  阶段2: 提示词生成 — 结合场景 + LoRA 信息输出结构化 Prompt + 参数
"""

import os
import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 模板相对于本文件的位置
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "prompt")


class PromptBuilder:
    """提示词构建器。"""

    def __init__(self, llm_client, config: dict):
        """
        Args:
            llm_client: LLMClient 实例。
            config:     完整配置 dict。
        """
        self.llm = llm_client
        self.config = config
        self.defaults = config.get("defaults", {})

    # ------------------------------------------------------------------
    # 阶段 1: 场景提取
    # ------------------------------------------------------------------

    def extract_scene(self, user_text: str) -> dict:
        """从用户文本中提取/编造生成目标场景。

        Returns:
            {"scene": "...", "is_fabricated": bool}
        """
        system_prompt = self._build_scene_extraction_system_prompt()
        user_msg = f"请分析以下用户输入，提取或编造一个适合图像生成的场景描述：\n\n{user_text}"

        response = self.llm.chat(system_prompt, user_msg, json_mode=True)

        try:
            result = json.loads(response)
            scene = result.get("scene", user_text)
            is_fab = result.get("is_fabricated", False)
        except json.JSONDecodeError:
            logger.warning("LLM 返回非 JSON，回退使用原始文本")
            scene = user_text
            is_fab = False

        logger.info("场景提取完成 → fabricated=%s, text=%s...", is_fab, scene[:80])
        return {"scene": scene, "is_fabricated": is_fab}

    # ------------------------------------------------------------------
    # 阶段 2: 提示词生成
    # ------------------------------------------------------------------

    # 默认负面提示词（当 LLM 未提供时使用）
    DEFAULT_NEGATIVE = (
        "worst quality, low quality, normal quality, lowres, blurry, distorted, "
        "deformed, messy, ugly, extra limbs, bad anatomy, bad proportions, "
        "distorted limbs, watermark, signature, text, jpeg artifacts, sketch, "
        "censorship, bad hands, mutated hands, missing fingers, extra fingers, "
        "poorly drawn face, asymmetrical eyes, deformed iris"
    )

    def build_prompt(self, scene: str, loras: list, user_prefs: Optional[dict] = None) -> dict:
        """生成最终正向/负向提示词和参数。

        Args:
            scene:       场景描述。
            loras:       匹配到的 LoRA 列表（已含 online_description 等）。
            user_prefs:  用户确认/修改的偏好（如 "发色改为红色"）。

        Returns:
            {
                "prompt":         "...",
                "negative":       "...",
                "model":          "...",
                "width":          832,
                "height":         1216,
                "steps":          25,
                "cfg_scale":      7.5,
                "sampler":        "k_euler",
                "loras":         [...],
                "chinese_note":   "...",
            }
        """
        system_prompt = self._load_template()

        # 注入 LoRA 信息
        lora_context = self._format_lora_context(loras)
        system_prompt += f"\n\n## 当前可用的 LoRA 信息\n{lora_context}"

        # 补充 JSON 格式约束（模板已定义输出 schema，此处兜底）
        system_prompt += (
            "\n\n---\n"
            "## ⚠️ 输出约束\n"
            "只输出上述 3 个字段的 JSON 对象，严禁添加 width/height/steps/model/seed 等其他字段。"
            "直接输出 JSON，不要 markdown 代码块标记。"
        )

        user_msg = f"场景描述：{scene}"
        if user_prefs:
            user_msg += f"\n\n用户额外要求：{json.dumps(user_prefs, ensure_ascii=False)}"

        user_msg += "\n\n请输出最终提示词。"

        response = self.llm.chat(system_prompt, user_msg, json_mode=True)
        logger.debug("LLM 原始响应（阶段2）: %s", response[:500])

        result = self._parse_llm_json(response)
        if result is None:
            logger.warning("LLM 未返回有效 JSON，使用原始文本作为 prompt")
            result = {"prompt": response, "negative": "", "chinese_note": ""}

        # 降级：若 LLM 未提供 negative/chinese_note，使用默认值
        result = self._ensure_fields(result, scene)

        # 合并默认参数
        return self._merge_defaults(result, loras)

    def _ensure_fields(self, result: dict, scene: str) -> dict:
        """确保 negative 和 chinese_note 不为空，提供降级默认值。

        Args:
            result: LLM 返回的解析结果。
            scene:  原始场景描述。

        Returns:
            补充了默认值的 result dict。
        """
        # negative 降级：二次 LLM 调用生成针对性 negative，失败则用默认值
        if not result.get("negative", "").strip():
            logger.info("LLM 未提供 negative，尝试二次 LLM 生成...")
            custom_neg = self._generate_negative(result.get("prompt", ""), scene)
            if custom_neg:
                result["negative"] = custom_neg
            else:
                logger.info("二次生成 negative 失败，使用默认负面提示词")
                result["negative"] = self.DEFAULT_NEGATIVE

        # chinese_note 降级：根据场景自动生成简略说明
        if not result.get("chinese_note", "").strip():
            logger.info("LLM 未提供 chinese_note，自动生成简略说明")
            result["chinese_note"] = (
                f"场景凝固：{scene[:60]}。"
                f"质量词前置，动漫风格绑定，视角因果过滤已应用。"
            )

        return result

    def _generate_negative(self, prompt: str, scene: str) -> str:
        """二次 LLM 调用：根据 prompt 和场景生成针对性负面提示词。"""
        if not prompt or not prompt.strip():
            return ""
        system = """You are an AI painting negative prompt expert. Given a positive prompt and scene, generate a concise English negative prompt.
Focus on common SD quality issues AND scene-specific problems to avoid. Output ONLY the negative prompt text, no JSON, no explanations."""
        user = f"Positive prompt: {prompt[:500]}\nScene: {scene[:200]}\n\nGenerate negative prompt:"
        try:
            response = self.llm.chat(system, user, json_mode=False)
            neg = response.strip()
            if neg and len(neg) > 5:
                logger.info("二次 LLM 生成 negative 成功 (%d chars)", len(neg))
                return neg
        except Exception as e:
            logger.warning("二次 LLM 生成 negative 失败: %s", e)
        return ""

    # ------------------------------------------------------------------
    # 内部 — JSON 解析
    # ------------------------------------------------------------------

    def _parse_llm_json(self, text: str) -> dict | None:
        """鲁棒解析 LLM 返回的 JSON，处理常见格式问题。

        尝试顺序:
          1. 直接 json.loads
          2. 修复尾逗号后解析
          3. 去除 markdown 代码围栏后解析
          4. 找到第一个 { 到最后一个 }，解析 JSON 对象

        Returns:
            解析成功的 dict，或 None。
        """
        if not text or not text.strip():
            return None

        # 1. 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. 修复常见 JSON 错误（尾逗号）
        repaired = self._repair_json(text.strip())
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # 3. 去除 markdown 围栏 (```json ... ```)，也修复尾逗号
        cleaned = re.sub(r'^```(?:json)?\s*\n', '', text.strip())
        cleaned = re.sub(r'\n```\s*$', '', cleaned)
        cleaned = self._repair_json(cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 4. 提取花括号之间的内容，修复尾逗号
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start >= 0 and end > start:
            extracted = self._repair_json(cleaned[start:end + 1])
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

        # 5. 兜底：正则提取字段（处理完全无法解析的 JSON）
        return self._regex_extract_fields(text)

    @staticmethod
    def _regex_extract_fields(text: str) -> dict | None:
        """正则兜底：从畸形的 JSON 文本中提取 prompt/negative/chinese_note。"""
        def _extract(key: str) -> str:
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'
            m = re.search(pattern, text, re.DOTALL)
            return m.group(1) if m else ""

        prompt = _extract("prompt")
        if not prompt:
            return None
        return {
            "prompt": prompt,
            "negative": _extract("negative"),
            "chinese_note": _extract("chinese_note"),
        }

    @staticmethod
    def _repair_json(text: str) -> str:
        """修复常见 JSON 格式错误：尾逗号。"""
        # 移除 } 或 ] 前的尾逗号
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return text

    def _build_scene_extraction_system_prompt(self) -> str:
        """构建场景提取阶段的系统提示词。"""
        return """你是一个专业的场景分析助手。你的任务：

1. 分析用户输入的文字（可能是小说片段、口述描述、角色介绍等）。
2. 如果输入包含多个场景，只提取**最后一个场景**作为图像生成目标。
3. 如果输入无明显场景，由你**自行编造一个合理的动漫/漫画风格场景**（包含角色外貌、动作、环境等关键元素）。
4. 输出格式必须是 JSON：
   {"scene": "场景描述（中文，包含角色、动作、环境、氛围）", "is_fabricated": true/false}

重要：
- 场景描述要具体，包含角色外貌、服饰、发型发色、动作姿态、环境背景、光影氛围、构图视角等关键维度。
- 如果角色外貌信息不完整，使用合理的默认值补充（如黑色中长发、深色眼眸、日常便服）。
- 不要直接生成英文 Prompt，只需要中文场景描述。"""

    def _format_lora_context(self, loras: list) -> str:
        """格式化 LoRA 信息供 LLM 参考。"""
        if not loras:
            return "（无可用 LoRA，生成时不使用 LoRA）"

        lines = []
        for lo in loras:
            desc = lo.get("online_description") or lo.get("description", "无描述")
            triggers = lo.get("online_trigger_words") or lo.get("trigger_words", [])
            lines.append(
                f"### {lo['name']}\n"
                f"- 触发词: {', '.join(triggers) if triggers else '无'}\n"
                f"- 基础模型: {lo.get('base_model', 'SDXL')}\n"
                f"- 强度: model={lo.get('strength_model', 0.8)}, clip={lo.get('strength_clip', 0.8)}\n"
                f"- 描述: {desc[:300]}"
            )
        return "\n\n".join(lines)

    def _load_template(self) -> str:
        """加载专业提示词模板。"""
        path = os.path.join(_TEMPLATE_DIR, "prompt_template.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        logger.warning("提示词模板文件不存在: %s，使用内置简化版", path)
        return self._builtin_template()

    def _builtin_template(self) -> str:
        """内置简化版提示词模板（当外部模板缺失时使用）。"""
        return """你是一位专业的AI绘画提示词工程师，专注于将中文场景描述转换为高质量英文Stable Diffusion提示词。

## 核心规则
1. 正向提示词结构: `[质量词] + [主体描述] + [风格] + [动作/姿态] + [环境/背景] + [光影/氛围] + [构图/视角]`
2. 质量词必须前置: `masterpiece, best quality, amazing quality, highly detailed`
3. 风格固定: `anime style, manga style, 2D illustration`
4. 禁止描述画面中不可见的物体或光源本体，只能描述视觉结果
5. 提示词越靠前权重越高

## 负面提示词
`worst quality, low quality, normal quality, lowres, blurry, distorted, deformed, messy, ugly, extra limbs, bad anatomy, bad proportions, distorted limbs, watermark, signature, text, jpeg artifacts, sketch, censorship, bad hands, mutated hands, missing fingers, extra fingers, poorly drawn face, asymmetrical eyes, deformed iris`

## 输出格式 (JSON)
{
  "prompt": "完整英文正向提示词",
  "negative": "完整英文负面提示词",
  "chinese_note": "中文说明（动态→静态的凝固点、权重分配、风格定位）"
}

如果提供了 LoRA 信息，请在 prompt 中包含对应的触发词。"""

    def _merge_defaults(self, result: dict, loras: list) -> dict:
        """将 LLM 输出与默认参数合并，过滤不合理值。"""
        d = self.defaults

        # 宽高校验：必须 >= 64 且为 64 倍数，否则回退默认值
        def _safe_dim(key: str) -> int:
            raw = result.get(key)
            if raw is not None:
                try:
                    val = int(raw)
                    if val >= 64 and val % 64 == 0:
                        return val
                except (ValueError, TypeError):
                    pass
            return int(d.get(key, 512))

        merged = {
            "prompt": result.get("prompt", ""),
            "negative": result.get("negative", ""),
            "chinese_note": result.get("chinese_note", ""),
            "model": result.get("model", d.get("model", "AlbedoBase XL (SDXL)")),
            "width": _safe_dim("width"),
            "height": _safe_dim("height"),
            "steps": int(result.get("steps", d.get("steps", 25))),
            "cfg_scale": float(result.get("cfg_scale", d.get("cfg_scale", 7.5))),
            "sampler": result.get("sampler", d.get("sampler", "k_euler")),
            "clip_skip": int(result.get("clip_skip", d.get("clip_skip", 2))),
            "karras": bool(result.get("karras", d.get("karras", False))),
            "hires_fix": bool(result.get("hires_fix", d.get("hires_fix", False))),
            "nsfw": bool(result.get("nsfw", d.get("nsfw", True))),
            "n": int(result.get("n", d.get("n", 1))),
            "seed": result.get("seed", ""),
            "loras": loras,
            "facefixer_strength": float(result.get("facefixer_strength", d.get("facefixer_strength", 0.0))),
        }
        return merged

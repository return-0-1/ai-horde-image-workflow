"""
AI Horde 图像生成工作流 — 主编排器。

完整流程:
  用户文本 → 场景提取 → LoRA匹配 → 提示词生成 → AI Horde生成 → 下载
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

import yaml

# 自动加载 .env 文件（优先级：.env 当前目录 → config.yaml → 系统环境变量）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv 未安装，依赖手动设置环境变量

from llm_client import LLMClient
from prompt_builder import PromptBuilder
from lora_manager import LoraManager
from lora_searcher import LoraSearcher
from civitai_client import CivitAIClient
from ai_horde_client import AIHordeClient
from params_parser import parse as parse_params, merge_overrides

logger = logging.getLogger(__name__)


class Workflow:
    """AI Horde 图像生成工作流编排器。

    用法:
        wf = Workflow("config.yaml")
        result = wf.run("一个男生坐在电脑桌前，表情专注")
        if result["success"]:
            print(result["images"])      # ["output/20250101_120000_xxx.webp", ...]
            print(result["prompt_data"]) # 完整的生成参数
    """

    def __init__(self, config_path: str = "config.yaml"):
        """初始化工作流。

        Args:
            config_path: YAML 配置文件路径。
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

        # 初始化各模块
        self.llm = LLMClient(self.config)
        self.prompt_builder = PromptBuilder(self.llm, self.config)
        self.lora_manager = LoraManager(self.config)
        self.civitai = CivitAIClient(self.config)
        self.lora_searcher = LoraSearcher(self.config, self.llm, self.civitai)
        self.horde = AIHordeClient(self.config)

        self.base_model = self.config.get("defaults", {}).get("model", "SDXL")
        logger.info("工作流初始化完成 → base_model=%s", self.base_model)

    # ------------------------------------------------------------------
    # 公共 — 主流程
    # ------------------------------------------------------------------

    def run(self, user_text: str,
            user_prefs: Optional[dict] = None,
            dry_run: bool = False) -> dict:
        """执行完整工作流。

        Args:
            user_text:  用户输入文本（小说片段/口述/角色介绍等）。
            user_prefs: 用户对生成参数的额外要求（如发色修改）。
            dry_run:    True=仅生成提示词不调用 AI Horde。

        Returns:
            {
                "success":       bool,
                "images":        ["path/to/img.webp", ...],   # dry_run 时为空
                "prompt_data":   {...},                        # 完整生成参数
                "scene":         "...",                        # 提取的场景
                "loras_matched": [...],                        # 匹配的 LoRA
                "task_id":       "...",                        # AI Horde 任务 ID
                "error":         "...",                        # 出错时的信息
            }
        """
        result = {
            "success": False,
            "images": [],
            "prompt_data": {},
            "scene": "",
            "loras_matched": [],
            "task_id": "",
            "error": "",
        }

        try:
            # ---- 阶段 1: LoRA 匹配（直接用用户文本，不提前提取场景） ----
            logger.info("=" * 50)
            logger.info("阶段 1/3: LoRA 匹配")

            # 1a. 白名单关键词匹配（用原始用户文本，节省一次 LLM 调用）
            loras = self.lora_manager.match(user_text, base_model=self.base_model)

            # 1b. 白名单无结果 → CivitAI 自动搜索 + LLM 审核
            searched_loras = []
            if not loras:
                logger.info("白名单无匹配，启动 CivitAI 自动搜索...")
                blacklist_ids = set()
                for b in self.config.get("lora_blacklist", []):
                    bid = b.get("model_id", "")
                    if bid:
                        blacklist_ids.add(str(bid))
                try:
                    searched_loras = self.lora_searcher.search(
                        user_text,
                        base_model="SDXL",
                        blacklist_model_ids=blacklist_ids,
                    )
                except Exception as e:
                    logger.warning("LoRA 搜索失败（回退白名单）: %s", e)

            loras = loras + searched_loras

            # 用 CivitAI 增强（失败不影响）
            enriched_loras = []
            for lo in loras:
                try:
                    enriched = self.civitai.enrich_lora(lo)
                except Exception as e:
                    logger.warning("LoRA 增强失败 [%s]: %s，使用本地信息", lo["name"], e)
                    enriched = dict(lo)
                enriched_loras.append(enriched)
            result["loras_matched"] = enriched_loras

            # ---- 阶段 2: 场景提取 + 提示词生成（合并为一次 LLM 调用） ----
            logger.info("-" * 40)
            logger.info("阶段 2/3: 场景提取 + 提示词生成（合并模式）")

            prompt_data = self.prompt_builder.build_prompt_from_user_text(
                user_text, enriched_loras, user_prefs,
            )

            if prompt_data is None:
                # 合并模式失败 → 回退到两阶段模式
                logger.warning("合并模式失败，回退到两阶段模式")
                extract_result = self.prompt_builder.extract_scene(user_text)
                scene = extract_result["scene"]
                result["scene"] = scene
                prompt_data = self.prompt_builder.build_prompt(scene, enriched_loras, user_prefs)
            else:
                # 提取 scene 信息（合并模式在 prompt_data 中已包含）
                scene = prompt_data.pop("_scene", user_text.strip()[:200])
                result["scene"] = scene
                # 清理内部字段
                prompt_data.pop("_is_fabricated", None)

            # ---- 阶段 2.5: 参数覆盖（从用户口语指令中提取） ----
            param_overrides = parse_params(user_text, llm_client=self.llm)
            if param_overrides:
                logger.info("从用户指令中解析出参数覆盖: %s", param_overrides)
                prompt_data = merge_overrides(prompt_data, param_overrides)

            # 强制 model 使用配置默认值（防止 LLM fallback 幻觉如 "anime"）
            prompt_data["model"] = self.config.get("defaults", {}).get(
                "model", prompt_data["model"]
            )

            result["prompt_data"] = prompt_data
            logger.info("Prompt 生成完成 → model=%s, size=%dx%d",
                        prompt_data["model"], prompt_data["width"], prompt_data["height"])
            logger.debug("正向 Prompt: %s", prompt_data["prompt"][:200])

            if dry_run:
                result["success"] = True
                logger.info("Dry run 模式 — 跳过图像生成")
                return result

            # ---- 阶段 3: AI Horde 生成 + 下载 ----
            logger.info("-" * 40)
            logger.info("阶段 3/3: AI Horde 图像生成")

            task_id = self.horde.submit(prompt_data)
            result["task_id"] = task_id

            images = self.horde.wait_and_download(task_id)
            result["images"] = images
            result["success"] = len(images) > 0

            logger.info("=" * 50)
            if result["success"]:
                logger.info("✅ 工作流完成 → 生成 %d 张图片", len(images))
                for img in images:
                    logger.info("   📁 %s", img)
            else:
                logger.warning("⚠️ 工作流完成但无图片输出")

        except Exception as e:
            logger.error("❌ 工作流失败: %s", e, exc_info=True)
            result["error"] = str(e)

        return result

    # ------------------------------------------------------------------
    # 公共 — 查看模型 & LoRA
    # ------------------------------------------------------------------

    def list_available_models(self) -> list:
        """列出 AI Horde 当前可用的图像模型。"""
        return self.horde.get_available_models()

    def list_loras(self) -> list[dict]:
        """列出已配置的 LoRA 白名单。"""
        return self.lora_manager.all

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # 控制台
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter(fmt))
            root_logger.addHandler(ch)

        # 文件
        log_file = log_cfg.get("file", "workflow.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        root_logger.addHandler(fh)

        logger.debug("日志系统初始化完成 → level=%s, file=%s", logging.getLevelName(level), log_file)


# ======================================================================
# CLI 入口
# ======================================================================

if __name__ == "__main__":
    import sys
    import argparse

    ap = argparse.ArgumentParser(
        description="AI Horde 图像生成工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单条生成
  python workflow.py "一个少女在樱花树下微笑"

  # Dry run 模式
  python workflow.py "测试场景" --dry-run

  # 指定配置文件
  python workflow.py "测试场景" --config my_config.yaml

  # 批量生成（从 TXT 文件）
  python workflow.py --file prompts.txt

  # 批量 dry run
  python workflow.py --file prompts.txt --dry-run

  # 批量 + 遇错即停
  python workflow.py --file prompts.txt --stop-on-error

  # 查询
  python workflow.py --list-models
  python workflow.py --list-loras
        """,
    )

    # 互斥组：用户文本 / 文件 / 查询
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("user_text", nargs="?", default=None,
                       help="用户场景描述（与 --file 互斥）")
    group.add_argument("-f", "--file", dest="batch_file", default=None,
                       help="批量场景 TXT 文件路径")
    group.add_argument("--list-models", action="store_true",
                       help="列出 AI Horde 当前可用模型")
    group.add_argument("--list-loras", action="store_true",
                       help="列出已配置的 LoRA 白名单")

    ap.add_argument("--dry-run", action="store_true",
                    help="仅生成提示词，不调用 AI Horde")
    ap.add_argument("--config", default="config.yaml",
                    help="配置文件路径（默认 config.yaml）")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="批量模式下遇错即停（默认跳过继续）")

    args = ap.parse_args()

    # --- 执行 ---
    wf = Workflow(args.config)

    # 列表模式
    if args.list_models:
        models = wf.list_available_models()
        print(f"可用模型数: {len(models)}")
        for m in models[:20]:
            name = m.get("name", m) if isinstance(m, dict) else m
            print(f"  - {name}")
        sys.exit(0)

    if args.list_loras:
        loras = wf.list_loras()
        print(f"已配置 LoRA 数: {len(loras)}")
        for lo in loras:
            print(f"  - {lo['name']} (base: {lo.get('base_model', '?')})")
        sys.exit(0)

    # 批量模式
    if args.batch_file:
        from batch_runner import BatchRunner
        runner = BatchRunner(wf)
        results = runner.run_batch(
            filepath=args.batch_file,
            dry_run=args.dry_run,
            continue_on_error=not args.stop_on_error,
        )
        runner.print_summary(results)
        if any(not r["success"] for r in results):
            sys.exit(1)
        sys.exit(0)

    # 单条模式（原逻辑）
    if not args.user_text:
        ap.error("请提供场景描述文本或使用 --file 指定批量文件")

    result = wf.run(args.user_text, dry_run=args.dry_run)

    # 输出结果摘要
    print("\n" + "=" * 50)
    print("📋 工作流结果")
    print("=" * 50)
    print(f"场景: {result['scene'][:100]}...")
    print(f"LoRA: {[l['name'] for l in result['loras_matched']]}")
    print(f"Prompt: {result['prompt_data'].get('prompt', '')[:120]}...")
    if args.dry_run:
        print("[DRY RUN] 未实际生成图片")
        print(json.dumps(result["prompt_data"], ensure_ascii=False, indent=2))
    elif result["success"]:
        print("✅ 生成成功:")
        for img in result["images"]:
            print(f"   📁 {img}")
    else:
        print(f"❌ 失败: {result['error']}")
        sys.exit(1)

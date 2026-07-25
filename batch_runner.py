"""
批量生图运行器 — 从 TXT 文件读取多行场景并逐条生成。

TXT 文件格式:
    # 注释行（整行跳过）
    场景描述行  // 行内注释
    --key=value  参数覆盖行（作用于下一条场景）
    --reset      重置参数覆盖为默认值
    空行跳过
"""

import os
import re
import logging

logger = logging.getLogger(__name__)


class BatchRunner:
    """批量生图运行器。

    用法:
        runner = BatchRunner(workflow)
        results = runner.run_batch("prompts.txt")
        runner.print_summary(results)
    """

    def __init__(self, workflow):
        """
        Args:
            workflow: Workflow 实例。
        """
        self.workflow = workflow

    # ------------------------------------------------------------------
    # 公共 — 批量运行
    # ------------------------------------------------------------------

    def run_batch(self, filepath: str, dry_run: bool = False,
                  continue_on_error: bool = True) -> list[dict]:
        """解析 TXT 文件并逐条执行工作流。

        Args:
            filepath:           TXT 文件路径。
            dry_run:            True=仅生成提示词不调用 AI Horde。
            continue_on_error:  True=单条失败继续下一条，False=遇错即停。

        Returns:
            每条场景的结果列表:
            [
                {
                    "line": 1,
                    "scene": "原始场景描述",
                    "success": bool,
                    "images": [...],
                    "error": "",
                    "param_overrides": {...},
                },
                ...
            ]
        """
        scenes = self._parse_file(filepath)
        results = []

        logger.info("=" * 60)
        logger.info("批量生图开始 → 共 %d 条场景", len(scenes))
        logger.info("=" * 60)

        for idx, scene_info in enumerate(scenes, 1):
            scene = scene_info["text"]
            overrides = scene_info["overrides"]
            line_no = scene_info["line"]

            logger.info("")
            logger.info("[%d/%d] 行 %d: %s", idx, len(scenes), line_no, scene[:80])

            try:
                user_prefs = overrides if overrides else None

                result = self.workflow.run(
                    user_text=scene,
                    user_prefs=user_prefs,
                    dry_run=dry_run,
                )
                results.append({
                    "line": line_no,
                    "scene": scene,
                    "success": result["success"],
                    "images": result["images"],
                    "error": result.get("error", ""),
                    "param_overrides": overrides,
                    "prompt_data": result.get("prompt_data", {}),
                })
            except Exception as e:
                logger.error("[%d/%d] 行 %d 失败: %s", idx, len(scenes), line_no, e)
                results.append({
                    "line": line_no,
                    "scene": scene,
                    "success": False,
                    "images": [],
                    "error": str(e),
                    "param_overrides": overrides,
                })
                if not continue_on_error:
                    logger.warning("遇错即停模式，中断批量处理")
                    break

        # 汇总
        success_count = sum(1 for r in results if r["success"])
        logger.info("")
        logger.info("=" * 60)
        logger.info("批量生图完成 → %d/%d 成功", success_count, len(results))
        logger.info("=" * 60)

        return results

    # ------------------------------------------------------------------
    # 内部 — TXT 解析
    # ------------------------------------------------------------------

    def _parse_file(self, filepath: str) -> list[dict]:
        """解析 TXT 文件，返回场景列表。

        连续的非空行会被合并为一个场景（用空格连接），
        空行 / 注释行 / 参数行作为场景分隔符。

        用 ``---`` 标记的场景块：块内所有内容（包括空行）保留为同一场景，
        块内行用换行符连接，不做注释/参数解析。

        Returns:
            [{"text": "场景描述", "line": 起始行号, "overrides": {...}}, ...]
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"TXT 文件不存在: {filepath}")

        scenes = []
        pending_overrides = {}
        buffer = []       # 当前场景的多行缓冲
        buffer_start = 0  # 缓冲区起始行号

        in_block = False       # 是否在 --- 块内
        block_buffer = []      # 块内行缓冲
        block_start = 0        # 块起始行号

        def _flush():
            """将缓冲区中的多行合并为一个场景。"""
            nonlocal buffer_start, pending_overrides
            if buffer:
                text = " ".join(buffer)
                scenes.append({
                    "text": text,
                    "line": buffer_start,
                    "overrides": dict(pending_overrides),
                })
                buffer.clear()
                buffer_start = 0
                pending_overrides = {}

        def _flush_block():
            """将 --- 块缓冲区合并为一个场景（换行连接）。"""
            nonlocal block_start, pending_overrides
            if block_buffer:
                # 去除首尾空行，内部空行保留
                text = "\n".join(block_buffer).strip()
                if text:
                    scenes.append({
                        "text": text,
                        "line": block_start,
                        "overrides": dict(pending_overrides),
                    })
                block_buffer.clear()
                block_start = 0
                pending_overrides = {}

        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.rstrip("\n\r")

                # 0. --- 场景块标记（最高优先级）
                if line.strip() == "---":
                    if in_block:
                        # 退出块模式
                        _flush_block()
                        in_block = False
                    else:
                        # 进入块模式：先 flush 普通缓冲区
                        _flush()
                        in_block = True
                        block_start = line_no + 1
                    continue

                # 块模式：原样收集所有行
                if in_block:
                    block_buffer.append(line)
                    continue

                # 1. 空行 → 场景分隔符，flush 缓冲区
                if not line.strip():
                    _flush()
                    continue

                # 2. 整行注释 (# 开头) → flush + 跳过
                if line.strip().startswith("#"):
                    _flush()
                    continue

                # 3. 参数覆盖行 (-- 开头) → flush + 记录参数
                if line.strip().startswith("--"):
                    _flush()
                    clean = line.strip()
                    if clean == "--reset":
                        pending_overrides = {}
                    else:
                        pending_overrides = self._parse_param_line(clean)
                    continue

                # 4. 行内注释 (//) → 去除
                comment_pos = line.find("//")
                if comment_pos >= 0:
                    line = line[:comment_pos]

                # 5. 去除前后空白后再检查是否为空
                text = line.strip()
                if not text:
                    _flush()
                    continue

                # 6. 有效场景行 → 加入缓冲区
                if not buffer:
                    buffer_start = line_no
                buffer.append(text)

        # 文件结束时 flush 剩余缓冲区
        _flush()
        # 未闭合的 --- 块也 flush
        if in_block:
            _flush_block()

        if not scenes:
            raise ValueError(f"TXT 文件中未找到有效场景描述: {filepath}")

        logger.info("解析 TXT 完成 → %s → %d 条场景", filepath, len(scenes))
        return scenes

    def _parse_param_line(self, line: str) -> dict:
        """解析参数行 --key=value --key2=value2 → dict。

        Returns:
            dict: 参数字典，如 {"steps": 30, "width": 1024}
        """
        overrides = {}
        pattern = r"--(\w[\w-]*)\s*=\s*(\S+)"
        matches = re.findall(pattern, line)
        for key, val in matches:
            if val.lower() in ("true", "false"):
                overrides[key] = val.lower() == "true"
            elif re.match(r"^-?\d+$", val):
                overrides[key] = int(val)
            elif re.match(r"^-?\d+\.\d+$", val):
                overrides[key] = float(val)
            else:
                overrides[key] = val
        return overrides

    # ------------------------------------------------------------------
    # 统计 & 报告
    # ------------------------------------------------------------------

    def print_summary(self, results: list[dict]):
        """打印批量结果摘要。"""
        total = len(results)
        success = sum(1 for r in results if r["success"])
        failed = total - success

        print("\n" + "=" * 60)
        print("📊 批量生图结果汇总")
        print("=" * 60)
        print(f"  总计: {total}  成功: {success}  失败: {failed}")
        print("-" * 60)

        for r in results:
            status = "✅" if r["success"] else "❌"
            print(f"  {status} 行 {r['line']:>4d}: {r['scene'][:60]}...")
            if r["success"]:
                for img in r.get("images", []):
                    print(f"         📁 {img}")
            else:
                print(f"         原因: {r.get('error', '未知')[:80]}")

        print("=" * 60)

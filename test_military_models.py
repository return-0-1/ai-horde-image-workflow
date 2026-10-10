"""
军事题材模型对比测试 —— 3 模型 × 2 张，并行提交。
"""
import os
import sys
import time
import logging
from datetime import datetime

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from ai_horde_client import AIHordeClient

# --- 配置 ---
MILITARY_PROMPT = (
    "masterpiece, best quality, photorealistic, 1man, special forces soldier, "
    "tactical gear, assault rifle, modern military, battlefield, dust and smoke, "
    "dramatic lighting, sharp focus, detailed equipment, 8k, hyperdetailed"
)

NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, distorted, deformed, ugly, "
    "cartoon, anime, 3d render, painting, illustration, watermark, text"
)

TEST_MODELS = [
    "ICBINP - I Can't Believe It's Not Photography",
    "Realistic Vision",
    "majicMIX realistic",
]

OUTPUT_DIR = "output/test_military"
GEN_COUNT = 2  # 每模型生成张数

# 简易日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("military_test")


def run_test():
    # 构造客户端配置（覆写 output 目录）
    config = {
        "ai_horde": {
            "api_url": "https://aihorde.net/api/v2",
            "api_key": "",
            "poll_interval": 10,
            "max_wait": 600,
        },
        "defaults": {
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "cfg_scale": 7.5,
            "sampler": "k_euler_a",
            "clip_skip": 2,
            "karras": False,
            "hires_fix": False,
            "nsfw": True,
            "n": GEN_COUNT,
            "trusted_workers": False,
            "slow_workers": True,
            "facefixer_strength": 0.0,
        },
        "output": {
            "directory": OUTPUT_DIR,
        },
    }

    client = AIHordeClient(config)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 提交所有任务（并行） ===
    tasks = {}  # model_name -> task_id
    submit_params = {}

    logger.info("=" * 60)
    logger.info("军事模型对比测试 — 提交阶段")
    logger.info("Prompt: %s", MILITARY_PROMPT[:120])
    logger.info("模型: %s", TEST_MODELS)
    logger.info("=" * 60)

    for model in TEST_MODELS:
        params = {
            "prompt": MILITARY_PROMPT,
            "negative": NEGATIVE_PROMPT,
            "model": model,
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "cfg_scale": 7.5,
            "sampler": "k_euler_a",
            "n": GEN_COUNT,
            "nsfw": True,
            "loras": [],
            "seed": "",
        }
        try:
            task_id = client.submit(params)
            tasks[model] = task_id
            submit_params[model] = params
            logger.info("✅ 已提交 [%s] → task_id=%s", model, task_id[:12])
        except Exception as e:
            logger.error("❌ 提交失败 [%s]: %s", model, e)

    if not tasks:
        logger.error("无任务成功提交，退出")
        return

    logger.info("已提交 %d 个任务，开始轮询...", len(tasks))

    # === 轮询 + 下载（串行轮询，每轮检查全部） ===
    results = {}  # model_name -> [image_paths]
    poll_interval = 10
    max_wait = 600
    start_time = time.time()

    pending = dict(tasks)

    while pending and (time.time() - start_time) < max_wait:
        for model, task_id in list(pending.items()):
            try:
                status = client._get(f"/generate/status/{task_id}")
            except Exception as e:
                logger.warning("查询 %s 失败: %s", model, e)
                continue

            done = status.get("done", False)
            faulted = status.get("faulted", False)

            if done and not faulted:
                images = client._download(status)
                results[model] = images
                logger.info("🎯 [%s] 完成 → %d 张图片", model, len(images))
                del pending[model]
            elif faulted:
                logger.error("💀 [%s] 任务失败 (faulted)", model)
                results[model] = []
                del pending[model]
            else:
                qpos = status.get("queue_position", "?")
                wtime = status.get("wait_time", "?")
                logger.debug("[%s] queue=%s, wait=%ss", model, qpos, wtime)

        if pending:
            remaining = list(pending.keys())
            elapsed = int(time.time() - start_time)
            logger.info("等待中... 剩余=%s, 已等待=%ds", remaining, elapsed)
            time.sleep(poll_interval)

    # 超时处理
    for model in list(pending.keys()):
        logger.warning("⏰ [%s] 超时未完成", model)
        results[model] = []
        del pending[model]

    # === 输出结果 ===
    elapsed = int(time.time() - start_time)
    print()
    print("=" * 60)
    print("📊 测试结果 (耗时 %ds)" % elapsed)
    print("=" * 60)
    for model in TEST_MODELS:
        imgs = results.get(model, [])
        status_icon = "✅" if imgs else "❌"
        print(f"  {status_icon} {model}: {len(imgs)} 张")
        for img in imgs:
            print(f"       📁 {img}")

    # 返回路径给调用者
    return results


if __name__ == "__main__":
    run_test()

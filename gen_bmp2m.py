"""单模型快速生成 — Juggernaut XL 俄军 BMP-2M"""
import os, sys, time, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from ai_horde_client import AIHordeClient

PROMPT = (
    "masterpiece, best quality, photorealistic, Russian BMP-2M infantry fighting vehicle, "
    "IFV, 30mm autocannon, Kornet anti-tank missiles on turret, military green camouflage, "
    "modern Russian army, driving through muddy terrain, dust trail behind tracks, "
    "overcast sky, dramatic atmosphere, sharp focus, detailed rivets and armor plating, "
    "realistic military photography, 8k, hyperdetailed"
)

NEGATIVE = (
    "worst quality, low quality, blurry, distorted, deformed, ugly, "
    "cartoon, anime, 3d render, painting, illustration, watermark, text, "
    "tank, T-72, T-90, wrong vehicle, BMP-1, BMP-3"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bmp2m")

config = {
    "ai_horde": {"api_url": "https://aihorde.net/api/v2", "api_key": "", "poll_interval": 10, "max_wait": 600},
    "defaults": {"width": 1024, "height": 768, "steps": 30, "cfg_scale": 7.5, "sampler": "k_euler_a",
                 "clip_skip": 2, "n": 2, "nsfw": False, "trusted_workers": False, "slow_workers": True,
                 "facefixer_strength": 0.0},
    "output": {"directory": "output/test_military"},
}

client = AIHordeClient(config)
os.makedirs("output/test_military", exist_ok=True)

params = {
    "prompt": PROMPT, "negative": NEGATIVE, "model": "Juggernaut XL",
    "width": 1024, "height": 768, "steps": 30, "cfg_scale": 7.5,
    "sampler": "k_euler_a", "n": 2, "nsfw": False, "loras": [], "seed": "",
}

logger.info("提交 BMP-2M 生成 → Juggernaut XL")
task_id = client.submit(params)
logger.info("task_id=%s", task_id)

start = time.time()
while time.time() - start < 600:
    status = client._get(f"/generate/status/{task_id}")
    if status.get("done"):
        if status.get("faulted"):
            logger.error("任务失败")
            break
        images = client._download(status)
        elapsed = int(time.time() - start)
        print(f"\n✅ 完成 ({elapsed}s):")
        for img in images:
            print(f"  📁 {img}")
        break
    qpos = status.get("queue_position", "?")
    wtime = status.get("wait_time", "?")
    logger.info("轮询中... queue=%s, wait=%ss, elapsed=%ds", qpos, wtime, int(time.time()-start))
    time.sleep(10)
else:
    logger.warning("超时")

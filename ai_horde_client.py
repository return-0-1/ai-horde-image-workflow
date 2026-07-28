"""
AI Horde API 客户端 — 异步图像生成、状态轮询、图片下载。

关键坑点 (来自开发总结):
  - 状态判断: 同时检查 state=="done" 和 done==True
  - 参数清理: 过滤 None 值，不传无关参数(如 control_type)
  - 返回值:   download_images 必须显式 return
  - LoRA:     使用 version_id (字符串)，不是 model_id
  - WebP:     AI Horde 返回 WebP，需转 PNG（QQ highway 不支持 WebP）
"""

import os
import io
import json
import time
import logging
from datetime import datetime
from typing import Optional
from urllib.request import Request, urlopen, ProxyHandler, build_opener
from urllib.error import URLError

# 绕过系统代理直连 AI Horde API（避免 v2rayN 干扰认证）
# 图片下载仍走系统代理（Cloudflare R2 国内可能被墙）
_no_proxy_opener = build_opener(ProxyHandler({}))

logger = logging.getLogger(__name__)


def _read_horde_key() -> str:
    """直接从项目 .env 文件读取 AI_HORDE_API_KEY（load_dotenv 失效时的兜底）。"""
    import re as _re
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(_env_path):
        return ""
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _m = _re.match(r"^\s*AI_HORDE_API_KEY\s*=\s*(.+?)\s*$", _line)
                if _m:
                    return _m.group(1).strip()
    except Exception:
        pass
    return ""


class AIHordeClient:
    """AI Horde API 客户端。"""

    def __init__(self, config: dict):
        cfg = config.get("ai_horde", {})
        self.api_url = cfg.get("api_url", "https://aihorde.net/api/v2").rstrip("/")
        self.api_key = cfg.get("api_key") or os.environ.get("AI_HORDE_API_KEY", "") or _read_horde_key()
        self.poll_interval = cfg.get("poll_interval", 10)
        self.max_wait = cfg.get("max_wait", 300)
        self.defaults = config.get("defaults", {})
        self.output_cfg = config.get("output", {})
        self._submitted_ids: list[str] = []  # 追踪已提交的任务 ID

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def submit(self, params: dict) -> str:
        """提交异步生成请求，返回任务 ID。"""
        payload = self._build_payload(params)
        logger.info("提交生成请求 → model=%s, size=%dx%d, steps=%d",
                    payload.get("models"), params.get("width"), params.get("height"),
                    payload["params"].get("steps"))
        logger.info("AI Horde 完整请求体:\n%s",
                    json.dumps(payload, ensure_ascii=False, indent=2))

        data = self._post("/generate/async", payload)
        task_id = data.get("id", "")
        if not task_id:
            raise RuntimeError(f"AI Horde 未返回任务 ID: {data}")

        logger.info("任务已提交 → id=%s, kudos=%s", task_id, data.get("kudos"))
        self._submitted_ids.append(task_id)
        return task_id

    def wait_and_download(self, task_id: str) -> list:
        """轮询任务状态，完成后下载图片。

        Returns:
            下载的图片本地路径列表。
        """
        status = self._poll(task_id)
        return self._download(status)

    def get_available_models(self) -> list:
        """获取当前可用的图像模型列表。"""
        data = self._get("/status/models")
        return data if isinstance(data, list) else []

    def get_performance(self) -> dict:
        """查询 AI Horde 全局性能/排队状态。

        Returns:
            {
                "queued_requests": 395, "worker_count": 10, "thread_count": 10,
                "queued_megapixelsteps": 4063.13, "past_minute_megapixelsteps": 532.11,
            }
        """
        data = self._get("/status/performance")
        return data if isinstance(data, dict) else {}

    def get_models_status(self) -> list:
        """查询所有图像模型的排队/worker 状态。

        Returns:
            模型列表，每项含 name, count, queued, eta, performance 等。
        """
        data = self._get("/status/models?type=image")
        return data if isinstance(data, list) else []

    def get_user_info(self) -> dict:
        """查询当前用户信息（含 kudos）。"""
        data = self._get("/find_user")
        return {
            "kudos": data.get("kudos", 0),
        }

    def get_active_tasks(self) -> list:
        """查询当前已提交但未完成的任务状态。

        逐个检查 _submitted_ids 中的任务，已完成/已失败的移除。

        Returns:
            [{queue_position, wait_time, done, processing, waiting, faulted}, ...]
        """
        active = []
        still_pending = []
        for tid in self._submitted_ids:
            try:
                s = self._get(f"/generate/status/{tid}")
            except Exception as e:
                logger.info("get_active_tasks: 查询 %s 失败: %s", tid[:8], e)
                still_pending.append(tid)
                continue
            if s.get("done") or s.get("faulted"):
                logger.info("get_active_tasks: %s 已完成，移除", tid[:8])
                continue  # 已完成，不再追踪
            still_pending.append(tid)
            active.append({
                "id": tid[:8],
                "queue_position": s.get("queue_position", 0),
                "wait_time": s.get("wait_time", 0),
                "done": s.get("done", False),
                "processing": s.get("processing", 0),
                "waiting": s.get("waiting", 0),
                "faulted": s.get("faulted", False),
            })
        self._submitted_ids = still_pending
        return active

    # ------------------------------------------------------------------
    # 内部 — 提交构建
    # ------------------------------------------------------------------

    def _build_payload(self, params: dict) -> dict:
        d = self.defaults
        loras_raw = params.get("loras", [])
        loras_clean = []
        for lo in loras_raw:
            if isinstance(lo, str):
                loras_clean.append({"name": lo, "model": 1, "clip": 1, "is_version": True})
            elif isinstance(lo, dict):
                vid = str(lo.get("version_id", ""))
                if vid:
                    loras_clean.append({
                        "name": vid,
                        "model": lo.get("strength_model", 1),
                        "clip": lo.get("strength_clip", 1),
                        "is_version": True,
                    })
        loras_clean = [n for n in loras_clean if n.get("name")]

        payload = {
            "prompt": params["prompt"],
            "params": {
                "sampler_name": params.get("sampler", d.get("sampler", "k_euler")),
                "cfg_scale": float(params.get("cfg_scale", d.get("cfg_scale", 7.5))),
                "height": int(params.get("height", d.get("height", 512))),
                "width": int(params.get("width", d.get("width", 512))),
                "steps": int(params.get("steps", d.get("steps", 25))),
                "karras": params.get("karras", d.get("karras", False)),
                "hires_fix": params.get("hires_fix", d.get("hires_fix", False)),
                "clip_skip": params.get("clip_skip", d.get("clip_skip", 2)),
                "n": params.get("n", d.get("n", 1)),
                "loras": loras_clean,
                "seed": str(params["seed"]) if params.get("seed") else "",
            },
            "nsfw": params.get("nsfw", d.get("nsfw", True)),
            "facefixer_strength": float(params.get("facefixer_strength", d.get("facefixer_strength", 0.0))),
            "trusted_workers": params.get("trusted_workers", d.get("trusted_workers", False)),
            "slow_workers": params.get("slow_workers", d.get("slow_workers", True)),
            "censor_nsfw": False,
            "models": [params.get("model", d.get("model", "AlbedoBase XL (SDXL)"))],
            "r2": True,
        }

        if params.get("negative"):
            payload["params"]["noprompt"] = params["negative"]

        clean_params = {}
        for k, v in payload["params"].items():
            if v is not None:
                clean_params[k] = v
        payload["params"] = clean_params

        return payload

    # ------------------------------------------------------------------
    # 内部 — 轮询
    # ------------------------------------------------------------------

    def _poll(self, task_id: str) -> dict:
        start = time.time()
        last_log = start

        while True:
            elapsed = time.time() - start
            if elapsed > self.max_wait:
                raise TimeoutError(f"任务 {task_id} 超时（>{self.max_wait}s），请稍后手动查询")

            status = self._get(f"/generate/status/{task_id}")

            done = status.get("done", False)
            state = status.get("state", "")
            if done or state == "done":
                logger.info("任务完成 → id=%s, 耗时=%.1fs", task_id, elapsed)
                return status

            # 自适应轮询间隔：根据队列位置动态调整
            queue_pos = status.get("queue_position", 0)
            processing = status.get("processing", 0)
            if processing > 0 or queue_pos <= 1:
                interval = 5    # 正在处理或即将处理，加速轮询
            elif queue_pos > 20:
                interval = 30   # 远在队尾，慢轮询
            elif queue_pos > 5:
                interval = 15   # 中等距离
            else:
                interval = 8    # queue_pos 2-5，适中

            if elapsed - last_log >= 30:
                wait_time = status.get("wait_time", "?")
                logger.info("轮询中 → id=%s, queue_pos=%s, wait_time=%s, elapsed=%.0fs, interval=%ds",
                            task_id, queue_pos, wait_time, elapsed, interval)
                last_log = elapsed

            time.sleep(interval)

    # ------------------------------------------------------------------
    # 内部 — 下载
    # ------------------------------------------------------------------

    def _download(self, status: dict) -> list:
        generations = status.get("generations", [])
        if not generations:
            faulted = status.get("faulted", False)
            if faulted:
                logger.error("任务执行失败 (faulted=True)，无图像输出")
            else:
                logger.warning("任务完成但 generations 为空")
            return []

        output_dir = self.output_cfg.get("directory", "output")
        os.makedirs(output_dir, exist_ok=True)

        images = []
        for i, gen in enumerate(generations):
            img_url = gen.get("img")
            if not img_url:
                logger.warning("第 %d 张图片无 URL，跳过", i + 1)
                continue

            seed = gen.get("seed", "unknown")
            model = gen.get("model", "unknown")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"{ts}_{model}_{seed}.png")

            logger.info("下载图片 → %s → %s", img_url[:80], filepath)
            try:
                req = Request(img_url, headers={"User-Agent": "AI-Horde-Workflow/1.0"})
                with urlopen(req, timeout=60) as resp:
                    data = resp.read()

                # AI Horde 返回 WebP → 转 PNG（QQ highway 不支持 WebP）
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(io.BytesIO(data))
                    if img.format == "WEBP":
                        img = img.convert("RGB")
                        img.save(filepath, "PNG")
                        logger.info("WebP → PNG 转换完成 → %s", filepath)
                    else:
                        with open(filepath, "wb") as f:
                            f.write(data)
                except ImportError:
                    with open(filepath, "wb") as f:
                        f.write(data)

                images.append(filepath)
                logger.info("下载完成 → %s (%s)", filepath, gen.get("model"))
            except URLError as e:
                logger.error("下载失败 → %s: %s", img_url[:80], e)

        return images

    # ------------------------------------------------------------------
    # 内部 — HTTP 工具
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        url = f"{self.api_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Horde-Workflow/1.0",
        }
        if self.api_key:
            headers["apikey"] = self.api_key

        data_bytes = None
        if body is not None:
            data_bytes = json.dumps(body).encode("utf-8")

        req = Request(url, data=data_bytes, headers=headers, method=method)

        try:
            with _no_proxy_opener.open(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            logger.error("AI Horde API 请求失败 [%s %s]: %s", method, path, e)
            raise RuntimeError(f"AI Horde API 不可用: {e}") from e

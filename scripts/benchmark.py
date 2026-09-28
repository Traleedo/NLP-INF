from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# /metrics 里我们关心的指标（按子串匹配，兼容不同 vLLM 版本的命名差异）
METRIC_KEYS = (
    "gpu_cache_usage_perc",
    "cpu_cache_usage_perc",
    "num_requests_running",
    "num_requests_waiting",
    "prefix_cache_hit_rate",
    "prefix_cache_hits_total",
    "prefix_cache_queries_total",
)


# ------------------------------------------------------------------ 工具
def http_get(url: str, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def parse_prometheus(text: str) -> dict[str, float]:
    """极简 Prometheus 文本解析：只取我们关心的指标，按系列名聚合。"""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name, value = line.rsplit(" ", 1)
        except ValueError:
            continue
        # 去掉 {labels}
        name = name.split("{", 1)[0]
        if not any(k in name for k in METRIC_KEYS):
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        # 同名系列取最大值：histogram 有 _sum/_count 后缀，会各自成为独立 key
        out[name] = max(out.get(name, float("-inf")), v)
    return out


def vram_used_mb() -> float | None:
    """整卡显存占用（MB）。用 nvidia-smi，不依赖本进程有 CUDA context。"""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


class VramSampler(threading.Thread):
    """后台线程：周期性采样整卡显存，累积到 samples（(相对秒, MB)）。"""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.samples: list[tuple[float, float]] = []

    def run(self) -> None:
        t0 = time.time()
        while not self._stop.is_set():
            v = vram_used_mb()
            if v is not None:
                self.samples.append((round(time.time() - t0, 2), v))
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)


# ------------------------------------------------------------------ 服务端
class VLLMServer:
    def __init__(self, tag: str):
        self.tag = tag
        self.model_path = cfg.FP16_MODEL if tag == "fp16" else str(cfg.AWQ_MODEL)
        self.served_name = cfg.SERVED_NAME[tag]
        self.log_path = cfg.LOG_DIR / f"vllm_{tag}.log"
        self.proc: subprocess.Popen | None = None
        self._log = None
        self.server_flags: set[str] = set()

    def _base_cmd(self) -> list[str]:
        cmd = [
            "vllm", "serve", self.model_path,
            "--host", cfg.HOST,
            "--port", str(cfg.PORT),
            "--served-model-name", self.served_name,
            "--max-model-len", str(cfg.max_model_len()),
            "--gpu-memory-utilization", str(cfg.GPU_MEM_UTIL),
            "--dtype", "auto",
            "--trust-remote-code",
            # 压测时关掉逐请求日志，减少服务端 I/O 干扰
            "--disable-log-requests",
        ]
        if cfg.NO_PREFIX_CACHING:
            # 前缀缓存默认是开的（vLLM V1），要做对比实验必须能关掉
            cmd.append("--no-enable-prefix-caching")
        return cmd

    def start(self) -> None:
        _, gb = cfg.gpu_info()
        print(f"\n🚀 启动 vLLM [{self.tag}] {self.model_path}")
        print(f"   GPU {gb:.1f} GB | max-model-len={cfg.max_model_len()} | "
              f"prefix-caching={'off' if cfg.NO_PREFIX_CACHING else 'on'}")

        self._log = open(self.log_path, "w", encoding="utf-8")
        env = {**os.environ, "VLLM_LOGGING_LEVEL": "INFO"}
        self.proc = subprocess.Popen(
            self._base_cmd(), stdout=self._log, stderr=subprocess.STDOUT, env=env
        )
        print(f"   PID {self.proc.pid}，日志 {self.log_path}")

    def wait_ready(self, timeout: int | None = None) -> None:
        """
        轮询 /health 直到就绪。
        关键：同时检测进程是否已经死掉 —— 原版脚本在这点上会永久挂起。
        """
        timeout = timeout or cfg.SERVER_READY_TIMEOUT
        url = f"http://{cfg.HOST}:{cfg.PORT}/health"
        t0 = time.time()

        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                self._log.flush()
                tail = self._tail_log(40)
                raise RuntimeError(
                    f"❌ vLLM [{self.tag}] 启动失败，退出码 {self.proc.returncode}。\n"
                    f"--- 日志尾部 ---\n{tail}\n"
                    f"常见原因：显存不足（FP16 7B 需要约 15 GB）。"
                )
            if http_get(url, timeout=3) is not None:
                print(f"   ✅ 就绪，耗时 {time.time() - t0:.0f}s")
                return
            time.sleep(3)

        self.stop()
        raise TimeoutError(
            f"❌ vLLM [{self.tag}] 等待 {timeout}s 仍未就绪。\n--- 日志尾部 ---\n{self._tail_log(40)}"
        )

    def metrics(self) -> dict[str, float]:
        text = http_get(f"http://{cfg.HOST}:{cfg.PORT}/metrics", timeout=10)
        return parse_prometheus(text) if text else {}

    def _tail_log(self, n: int) -> str:
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except Exception:
            return "(日志不可读)"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        if self._log:
            self._log.close()
        # 等显存真正归还，否则下一个模型加载会撞上残留占用
        for _ in range(60):
            if (vram_used_mb() or 0) < 1000:
                break
            time.sleep(2)


# ------------------------------------------------------------------ 客户端
def bench_help() -> str:
    try:
        return subprocess.run(
            ["vllm", "bench", "serve", "--help"], capture_output=True, text=True, timeout=120
        ).stdout
    except Exception:
        return ""


def build_bench_cmd(help_text: str, tag: str, rate: float, out_name: str) -> list[str]:

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in help_text:
                return c
        return None

    model = cfg.FP16_MODEL if tag == "fp16" else str(cfg.AWQ_MODEL)
    cmd = [
        "vllm", "bench", "serve",
        "--model", model,
        "--backend", "vllm",
        "--host", cfg.HOST,
        "--port", str(cfg.PORT),
        "--num-prompts", str(cfg.NUM_PROMPTS),
        "--request-rate", "inf" if rate == float("inf") else str(rate),
    ]

    # 服务端是用 --served-model-name 起的，请求里的 model 字段要对上
    if "--served-model-name" in help_text:
        cmd += ["--served-model-name", cfg.SERVED_NAME[tag]]

    ds_flag = pick("--dataset-name", "--dataset")
    if ds_flag:
        cmd += [ds_flag, "sharegpt"]
    cmd += ["--dataset-path", str(cfg.SHAREGPT_PATH)]

    warmup_flag = pick("--num-warmup", "--num-warm-up")
    if warmup_flag:
        cmd += [warmup_flag, str(cfg.NUM_WARMUP)]
    if "--percentile-metrics" in help_text:
        cmd += ["--percentile-metrics", "ttft,tpot,itl,e2el"]
    if "--metric-percentiles" in help_text:
        cmd += ["--metric-percentiles", "50,90,99"]
    if "--save-result" in help_text:
        cmd += ["--save-result", "--result-dir", str(cfg.RAW_DIR), "--result-filename", out_name]
    if "--disable-tqdm" in help_text:
        cmd.append("--disable-tqdm")
    if cfg.MAX_CONCURRENCY and "--max-concurrency" in help_text:
        cmd += ["--max-concurrency", str(cfg.MAX_CONCURRENCY)]
    return [c for c in cmd if c is not None]


def run_one_rate(tag: str, rate: float, help_text: str) -> dict:
    label = "inf" if rate == float("inf") else str(rate)
    out_name = f"{tag}_rate{label}.json"
    print(f"\n   ▶ {cfg.SERVED_NAME[tag]} @ request_rate={label} "
          f"({cfg.NUM_PROMPTS} prompts, warmup {cfg.NUM_WARMUP})")

    cmd = build_bench_cmd(help_text, tag, rate, out_name)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"     ⚠️ 压测失败（退出码 {proc.returncode}）")
        print("     " + "\n     ".join(proc.stderr.strip().splitlines()[-15:]))
        return {"tag": tag, "request_rate": label, "error": proc.stderr[-2000:]}

    raw = cfg.RAW_DIR / out_name
    if raw.exists():
        result = json.loads(raw.read_text(encoding="utf-8"))
    else:
        result = {}

    result["tag"] = tag
    result["request_rate"] = label
    result["wall_seconds"] = round(elapsed, 1)
    print(f"     ✓ 完成，用时 {elapsed:.0f}s，"
          f"output {result.get('output_throughput', float('nan')):.1f} tok/s")
    return result


def summarize_extras(result: dict, samples: list[tuple[float, float]], metrics: dict) -> None:
    """
    把显存 / metrics 合并进结果。

    字段名在不同 vLLM 版本间有差异，所以每个都做多候选回退，
    取不到就跳过而不是 KeyError。
    """
    def first(d: dict, *keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    peak = max((v for _, v in samples), default=None)
    result["vram_peak_mb"] = peak
    result["vram_samples"] = samples
    result["vllm_metrics"] = metrics

    if metrics:
        result["kv_cache_usage_peak"] = first(metrics, "vllm:gpu_cache_usage_perc")
        result["prefix_cache_hit_rate"] = first(metrics, "vllm:prefix_cache_hit_rate")
        result["requests_running_peak"] = first(metrics, "vllm:num_requests_running")
        result["requests_waiting_peak"] = first(metrics, "vllm:num_requests_waiting")


# ------------------------------------------------------------------ 主流程
def run_model(tag: str, rates: list[float], help_text: str) -> list[dict]:
    server = VLLMServer(tag)
    results: list[dict] = []
    sampler = VramSampler()

    try:
        server.start()
        server.wait_ready()

        print("   预热 / 记录空载指标...")
        time.sleep(5)
        idle_metrics = server.metrics()
        idle_vram = vram_used_mb()
        print(f"   空载显存 {idle_vram:.0f} MB" if idle_vram else "   空载显存 N/A")

        sampler.start()

        for rate in rates:
            # 只取本速率窗口内的显存样本。采样器是累积的，
            # 不切片的话后一档会把前面几档的峰值也算进来，越往后越虚高。
            mark = len(sampler.samples)
            r = run_one_rate(tag, rate, help_text)
            metrics = server.metrics()
            summarize_extras(r, list(sampler.samples[mark:]), metrics)
            r["idle_vram_mb"] = idle_vram
            r["idle_metrics"] = idle_metrics
            results.append(r)
            print(f"     显存峰值 {r['vram_peak_mb']:.0f} MB | "
                  f"KV cache 占用 {r.get('kv_cache_usage_peak')}")
    finally:
        sampler.stop()
        server.stop()

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM FP16 vs AWQ 压测")
    ap.add_argument("--models", default="fp16,awq", help="要压测的模型，逗号分隔")
    ap.add_argument("--rates", default=None, help="请求速率，逗号分隔，如 1,8,inf")
    ap.add_argument("--prompts", type=int, default=None, help="每个速率跑多少条请求")
    args = ap.parse_args()

    if args.prompts:
        cfg.NUM_PROMPTS = args.prompts

    if args.rates:
        rates = [float("inf") if r.strip() == "inf" else float(r) for r in args.rates.split(",")]
    else:
        rates = cfg.REQUEST_RATES

    tags = [t.strip() for t in args.models.split(",") if t.strip()]
    if "awq" in tags and not Path(cfg.AWQ_MODEL).exists():
        sys.exit(f"❌ 找不到量化模型 {cfg.AWQ_MODEL}，先运行 scripts/quantize_awq.py")
    if not cfg.SHAREGPT_PATH.exists():
        sys.exit(f"❌ 找不到数据集 {cfg.SHAREGPT_PATH}")

    name, gb = cfg.gpu_info()
    if gb == 0:
        sys.exit("❌ 没检测到 CUDA GPU。")
    if "fp16" in tags and gb < 18:
        print(f"⚠️  显存 {gb:.1f} GB 放不下 FP16 7B（约 15 GB），自动跳过 FP16 基线。")
        print("   没有 FP16 基线就只能报「AWQ 能跑」，报不出「AWQ 快多少 / 掉多少精度」。")
        tags = [t for t in tags if t != "fp16"]

    help_text = bench_help()
    if not help_text:
        print("⚠️  读不到 `vllm bench serve --help`，按默认 flag 拼接。")

    print(f"{'=' * 60}\nvLLM 压测\n"
          f"  模型   : {tags}\n  速率   : {rates}\n"
          f"  请求数 : {cfg.NUM_PROMPTS}  预热: {cfg.NUM_WARMUP}\n"
          f"  GPU    : {name} ({gb:.1f} GB)\n{'=' * 60}")

    all_results = {}
    for tag in tags:
        all_results[tag] = run_model(tag, rates, help_text)

    payload = {
        "env": cfg.env_snapshot(),
        "config": {
            "num_prompts": cfg.NUM_PROMPTS,
            "num_warmup": cfg.NUM_WARMUP,
            "max_model_len": cfg.max_model_len(),
            "gpu_mem_util": cfg.GPU_MEM_UTIL,
            "prefix_caching": not cfg.NO_PREFIX_CACHING,
            "dataset": str(cfg.SHAREGPT_PATH),
            "fp16_model": cfg.FP16_MODEL,
            "awq_model": str(cfg.AWQ_MODEL),
        },
        "results": all_results,
    }
    (cfg.RESULTS_DIR / "bench.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n🎉 压测完成 → {cfg.RESULTS_DIR / 'bench.json'}")
    print("   下一步：python scripts/report.py  生成对比报告与图表")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass  
# ---------------------------------------------------------------- 目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "model"
FILES_DIR = PROJECT_ROOT / "files"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"        # vllm bench serve 的原始 json
LOG_DIR = RESULTS_DIR / "logs"       # vllm 服务端日志
FIG_DIR = RESULTS_DIR / "figures"    # 报告用图

# ---------------------------------------------------------------- 模型
# 原始 FP16 模型：HF repo id 或本地路径
FP16_MODEL = os.environ.get("FP16_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# AWQ 量化产物
AWQ_MODEL = Path(os.environ.get("AWQ_MODEL", MODELS_DIR / "Qwen2.5-7B-Instruct-AWQ-4bit"))

# vLLM 的 --served-model-name，给两个模型起短名，避免命令行里塞长路径
SERVED_NAME = {"fp16": "qwen-fp16", "awq": "qwen-awq"}

# ---------------------------------------------------------------- 数据
SHAREGPT_PATH = Path(os.environ.get("SHAREGPT_PATH", FILES_DIR / "sharegpt-1000.json"))

# ---------------------------------------------------------------- 量化
# 校准样本数。原版取 128 条「首轮 human 提问」，序列极短，
# 会让 AWQ 的 per-channel scale 搜索方差过大 —— 这里改成拼完整多轮对话。
CALIB_NUM = int(os.environ.get("CALIB_NUM", 128))
# 单条校准样本的 token 上限（超过则「保头也保尾」截断）
CALIB_MAX_LEN = int(os.environ.get("CALIB_MAX_LEN", 512))
# 短于这个 token 数的样本直接丢弃：对 scale 估计没贡献，纯噪声
CALIB_MIN_LEN = int(os.environ.get("CALIB_MIN_LEN", 32))

QUANT_CONFIG = {
    "zero_point": bool(int(os.environ.get("ZERO_POINT", 1))),
    "q_group_size": int(os.environ.get("Q_GROUP_SIZE", 128)),
    "w_bit": int(os.environ.get("W_BIT", 4)),
    "version": os.environ.get("AWQ_VERSION", "GEMM"),  # GEMM：vLLM 支持最好
}

# AWQ 的两个可选优化开关，暴露出来是为了做消融（默认都开）
DUO_SCALING = bool(int(os.environ.get("DUO_SCALING", 1)))
APPLY_CLIP = bool(int(os.environ.get("APPLY_CLIP", 1)))

# ---------------------------------------------------------------- 精度评估
PPL_MAX_LEN = int(os.environ.get("PPL_MAX_LEN", 2048))
PPL_STRIDE = int(os.environ.get("PPL_STRIDE", 512))
# KL 散度评估的规模：CHUNKS 段 × CHUNK_LEN token。
# 每段都要缓存 FP16 的完整 logits（vocab≈152k），规模别开太大：
# 内存 ≈ CHUNKS × CHUNK_LEN × vocab × 2B，默认约 1.2 GB。
KL_CHUNKS = int(os.environ.get("KL_CHUNKS", 8))
KL_CHUNK_LEN = int(os.environ.get("KL_CHUNK_LEN", 512))

# ---------------------------------------------------------------- 压测
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", 1000))   # 原版 100 太轻，压不出饱和点
NUM_WARMUP = int(os.environ.get("NUM_WARMUP", 20))
# 请求速率扫描：找出吞吐饱和点和延迟拐点。inf = 一次性全部发出（极限压测）
REQUEST_RATES = [1, 2, 4, 8, 16, 32, float("inf")]
# vllm 服务端口（两个模型串行跑，用同一个端口）
PORT = int(os.environ.get("VLLM_PORT", 8000))
HOST = "127.0.0.1"
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", 0.90))
# 服务起不来时的等待上限（秒）。原版脚本是 while 死循环，服务 OOM 就永远卡住
SERVER_READY_TIMEOUT = int(os.environ.get("SERVER_READY_TIMEOUT", 1800))
# 前缀缓存：vLLM V1 默认开启。要测它在多轮对话里的收益，必须能显式关掉做对照
NO_PREFIX_CACHING = bool(int(os.environ.get("NO_PREFIX_CACHING", 0)))
# 并发上限。request_rate=inf 时建议设一个，否则 1000 条请求瞬间涌入
MAX_CONCURRENCY = int(os.environ["MAX_CONCURRENCY"]) if os.environ.get("MAX_CONCURRENCY") else None


def max_model_len() -> int:
    """
    按显存给一个保守的 max_model_len，可用环境变量覆盖。

    Qwen2.5-7B 的 KV cache 约 56 KB/token（28 层 × 4 KV head × 128 dim × 2 字节 × 2）。
    FP16 权重约 15 GB，AWQ W4A16 约 4.5 GB —— 6 GB 卡只能跑 AWQ，且上下文要压得很低。
    """
    if os.environ.get("MAX_MODEL_LEN"):
        return int(os.environ["MAX_MODEL_LEN"])
    _, gb = gpu_info()
    if gb >= 70:
        return 32768
    if gb >= 38:
        return 16384
    if gb >= 20:
        return 8192
    if gb >= 12:
        return 4096
    return 2048


# ---------------------------------------------------------------- 硬件
def gpu_info() -> tuple[str | None, float]:
    """返回 (GPU 型号, 显存 GB)。无 GPU 时返回 (None, 0.0)。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, 0.0
        p = torch.cuda.get_device_properties(0)
        return p.name, p.total_memory / 1024**3
    except Exception:
        return None, 0.0


def env_snapshot() -> dict:
    """
    实验元数据。压测结果没有硬件/版本信息就没有可比性 ——
    这是实验报告的基本盘，写进每个结果 json 里。
    """
    import platform

    snap: dict = {"platform": platform.platform(), "python": platform.python_version()}
    name, gb = gpu_info()
    snap["gpu"] = name
    snap["gpu_mem_gb"] = round(gb, 2) if gb else None

    for mod in ("torch", "transformers", "vllm", "awq", "datasets"):
        try:
            import importlib

            snap[mod] = getattr(importlib.import_module(mod), "__version__", "unknown")
        except Exception:
            snap[mod] = None

    try:
        import torch

        snap["cuda"] = torch.version.cuda
    except Exception:
        snap["cuda"] = None

    return snap


def ensure_dirs() -> None:
    for d in (MODELS_DIR, RESULTS_DIR, RAW_DIR, LOG_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

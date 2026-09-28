from __future__ import annotations
import gc
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

def model_device(model):
    """HF model / AutoAWQ 包装 都能拿到 device。"""
    for attr in ("device", "model"):
        obj = getattr(model, attr, None)
        if obj is not None and hasattr(obj, "device"):
            return obj.device
    return next(model.parameters()).device


def load_model(tag: str):
    """tag ∈ {'fp16', 'awq'}"""
    from transformers import AutoTokenizer

    path = cfg.FP16_MODEL if tag == "fp16" else str(cfg.AWQ_MODEL)
    if tag == "awq" and not Path(path).exists():
        sys.exit(f"❌ 找不到量化模型 {path}，先运行 scripts/quantize_awq.py")

    print(f"   加载模型: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tag == "awq":
        from awq import AutoAWQForCausalLM

        wrapper = AutoAWQForCausalLM.from_quantized(path, fuse_layers=True, trust_remote_code=True)
        model = wrapper.model  # 直接用底层 HF model 跑 forward
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
    model.eval()
    return model, tokenizer


def free_model(model) -> None:
    """显式释放，否则下一个模型加载时必然 OOM。"""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
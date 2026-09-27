"""
WikiText-2 困惑度（PPL）评估，对比 FP16 原始模型与 AWQ 量化模型。

相对原版的修改：

1. **loss 按 token 数加权**。原版对每个窗口取 `loss.mean()` 再平均。每个窗口的
   loss 是「该窗口内非 mask token 的平均 NLL」，而各个窗口参与计算的 token 数
   并不相同（最后一个窗口更短），直接平均会让短窗口权重偏高。标准做法是
   `Σ(loss_i × n_i) / Σ n_i`，其中 n_i 是窗口 i 实际参与 loss 的 token 数
   = trg_len - 1（模型内部把 label 左移一位，详见下方注释）。

2. **修正 `model.device` 报错**。AutoAWQForCausalLM 是对 HF model 的包装，
   本身没有 `.device` 属性，原版 eval_ppl.py:34 会 AttributeError。

3. **一次只评估一个模型**。原版连续加载 FP16(15GB) 和 AWQ 两个模型且中间不释放，
   必然 OOM。现在改成命令行驱动，跑完一个显式释放再跑下一个。

4. 结果写 json，供 report.py 汇总出报告。

用法：
    python scripts/eval_ppl.py --fp16
    python scripts/eval_ppl.py --awq
    python scripts/eval_ppl.py            # 自动：显存放得下哪个就跑哪个
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def build_test_ids(tokenizer) -> torch.Tensor:
    from datasets import load_dataset

    print("   加载 WikiText-2 测试集...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"   共 {ids.size(1):,} 个 token")
    return ids


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


@torch.no_grad()
def compute_ppl(model, input_ids, max_len: int, stride: int) -> dict:
    device = model_device(model)
    seq_len = input_ids.size(1)

    nll_sum = 0.0   # Σ(loss × 参与 token 数)
    n_tokens = 0    # Σ 参与 token 数
    prev_end = 0
    windows = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end  # 本窗口新出现的 token 数，只对它算 loss

        ids = input_ids[:, begin:end].to(device)
        targets = ids.clone()
        targets[:, :-trg_len] = -100  # 前 (窗口长度 - trg_len) 个位置是上文，不计 loss

        loss = model(ids, labels=targets).loss

        # 模型内部把 label 左移一位（logits[t] 预测 target[t+1]），
        # 所以实际参与 loss 的 token 数是 trg_len - 1，不是 trg_len。
        n = trg_len - 1
        nll_sum += loss.item() * n
        n_tokens += n
        windows += 1
        prev_end = end

        if end == seq_len:
            break

    return {
        "ppl": math.exp(nll_sum / n_tokens),
        "nll_mean": nll_sum / n_tokens,
        "tokens_scored": n_tokens,
        "windows": windows,
        "max_len": max_len,
        "stride": stride,
    }


def evaluate(tag: str) -> dict:
    label = "FP16 原始模型" if tag == "fp16" else "AWQ W4A16 量化模型"
    print(f"\n{'=' * 60}\n评估 {label} ({tag})\n{'=' * 60}")

    model, tokenizer = load_model(tag)
    test_ids = build_test_ids(tokenizer)
    print("   计算 PPL...")
    result = compute_ppl(model, test_ids, cfg.PPL_MAX_LEN, cfg.PPL_STRIDE)
    free_model(model)

    print(f"   ✅ PPL = {result['ppl']:.4f}  "
          f"(评分 token {result['tokens_scored']:,}, {result['windows']} 个窗口)")

    out = {
        "tag": tag,
        "label": label,
        "model": cfg.FP16_MODEL if tag == "fp16" else str(cfg.AWQ_MODEL),
        "dataset": "wikitext-2-raw-v1/test",
        "env": cfg.env_snapshot(),
        **result,
    }
    cfg.ensure_dirs()
    (cfg.RESULTS_DIR / f"ppl_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def auto_plan() -> list[str]:
    """
    按显存决定能跑哪些模型。FP16 7B 约 15 GB，AWQ 约 4.5 GB。
    盲目两个都跑是原版 OOM 的根因，这里给出明确提示而不是崩掉。
    """
    name, gb = cfg.gpu_info()
    if gb == 0:
        sys.exit("❌ 没检测到 CUDA GPU。PPL 评估必须在 GPU 机器上跑。")
    print(f"GPU: {name} ({gb:.1f} GB)")

    plan = []
    if gb >= 18:
        plan.append("fp16")
    else:
        print("⚠️  显存不足以加载 FP16 7B（约 15 GB），跳过 FP16 基线。")
        print("   结果将只有 AWQ 一项，无法给出精度损失 —— 建议换更大显存的机器跑对比。")
    if Path(cfg.AWQ_MODEL).exists():
        plan.append("awq")
    else:
        print(f"⚠️  未找到量化模型 {cfg.AWQ_MODEL}，跳过。")
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description="WikiText-2 PPL 评估")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fp16", action="store_true", help="只评估 FP16 原始模型")
    g.add_argument("--awq", action="store_true", help="只评估 AWQ 量化模型")
    args = ap.parse_args()

    if args.fp16:
        tags = ["fp16"]
    elif args.awq:
        tags = ["awq"]
    else:
        tags = auto_plan()

    if not tags:
        sys.exit("没有可评估的模型。")

    results = [evaluate(t) for t in tags]  # 串行，每个跑完就释放显存

    if len(results) == 2:
        fp16, awq = results
        diff = awq["ppl"] - fp16["ppl"]
        rel = diff / fp16["ppl"] * 100
        print(f"\n{'=' * 60}\n📈 精度对比\n{'=' * 60}")
        print(f"  FP16  PPL : {fp16['ppl']:.4f}")
        print(f"  AWQ   PPL : {awq['ppl']:.4f}")
        print(f"  绝对差    : {diff:+.4f}   相对差: {rel:+.2f}%")
        if diff < 0.5:
            print("  🎉 量化精度损失很小。")
        else:
            print("  ⚠️  精度损失偏大，检查校准集构造（见 quantize_awq.py 注释）。")
        print("  提示：PPL 只是一项指标，指令跟随能力还需 eval_kl.py / 任务指标佐证。")
        print("=" * 60)


if __name__ == "__main__":
    main()

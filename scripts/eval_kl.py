"""
FP16 与 AWQ 模型输出分布的 KL 散度评估。

为什么需要它：PPL 是对整段文本平均后的单一标量，对「少数 token 上分布被量化
严重扭曲」这种情况很不敏感 —— 大部分 token 预测得很准，均值就把问题抹平了。
KL 散度直接度量逐位置的分布偏移，是量化误差更敏感的探针。

做法（受显存限制，分两趟）：
  第 1 趟：加载 FP16，对 N 段文本做 forward，把 logits 存到磁盘
  第 2 趟：释放 FP16，加载 AWQ，对同样的文本做 forward，逐段读回 FP16 logits 算 KL

之所以不一次性加载两个模型：FP16 权重 15 GB + AWQ 4.5 GB，单卡放不下。
之所以缓存到磁盘而不是内存：见 config.KL_CHUNKS 注释。

指标：
  - KL(FP16 || AWQ)  逐位置 KL 的均值 / p95 / 最大值
  - Top-1 一致率     两个模型 argmax 相同的比例（最好解释的一个数）
  - 平均 logit 绝对差

注意：FP16 本身也是近似，所以这个 KL 度量的是「相对偏移」而非绝对误差。

用法：
    python scripts/eval_kl.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

CACHE_PATH = cfg.RESULTS_DIR / "kl_fp16_logits.pt"


def build_chunks(tokenizer) -> torch.Tensor:
    """从 WikiText-2 测试集切出 N 段定长文本，返回 [N, CHUNK_LEN] 的 input_ids。"""
    from datasets import load_dataset

    print("   加载 WikiText-2 测试集...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]

    need = cfg.KL_CHUNKS * cfg.KL_CHUNK_LEN
    if ids.numel() < need:
        sys.exit(f"❌ 测试集只有 {ids.numel()} token，不足 {need}，调小 KL_CHUNKS。")

    # 从中间往后的位置取，避开开头（开头文本质量较差，且没有上下文）
    start = max(0, ids.numel() // 4)
    chunks = ids[start : start + need].view(cfg.KL_CHUNKS, cfg.KL_CHUNK_LEN)
    print(f"   切出 {cfg.KL_CHUNKS} 段 × {cfg.KL_CHUNK_LEN} token = {need:,} 个位置")
    return chunks


def dump_fp16_logits(chunks: torch.Tensor) -> None:
    """第 1 趟：跑 FP16，logits 存盘。"""
    from eval_ppl import free_model, load_model, model_device

    model, _ = load_model("fp16")
    device = model_device(model)

    out = []
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            logits = model(chunk.unsqueeze(0).to(device)).logits[0]
            # 存 fp16 到 CPU：vocab≈152k，N 段 × 512 token 约 1.2 GB
            out.append(logits.to(torch.float16).cpu())
            print(f"   [{i + 1}/{len(chunks)}] logits {tuple(out[-1].shape)}")

    free_model(model)
    torch.save({"chunks": chunks.cpu(), "logits": out}, CACHE_PATH)
    print(f"   FP16 logits 已缓存到 {CACHE_PATH}")


def kl_stats(fp_logits: torch.Tensor, awq_logits: torch.Tensor) -> dict:
    """
    逐位置计算 KL(FP16 || AWQ)。

    在 fp32 下算，并按位置分块，避免 152k 词表上一次性做 softmax 撑爆显存。
    不用 F.kl_div 是为了避开它 reduction='batchmean' 在不同输入维度下的歧义。
    """
    n = fp_logits.size(0)
    kl_list, top1_match, logit_diff = [], 0, []

    for s in range(0, n, 128):
        fp = fp_logits[s : s + 128].to(torch.float32)
        q = awq_logits[s : s + 128].to(torch.float32)

        logp = F.log_softmax(fp, dim=-1)
        logq = F.log_softmax(q, dim=-1)
        # KL(p||q) = Σ p·(log p - log q)，等价形式，数值上比 p·log(p/q) 稳定
        kl = (logp.exp() * (logp - logq)).sum(dim=-1)
        kl_list.append(kl.cpu())

        top1_match += int((fp.argmax(-1) == q.argmax(-1)).sum().item())
        logit_diff.append((fp - q).abs().mean().item())

    kl = torch.cat(kl_list)
    return {
        "kl_mean": kl.mean().item(),
        "kl_p50": kl.median().item(),
        "kl_p95": kl.quantile(0.95).item(),
        "kl_max": kl.max().item(),
        "top1_agreement": top1_match / n,
        "mean_abs_logit_diff": sum(logit_diff) / len(logit_diff),
        "positions": n,
    }


def main() -> None:
    from eval_ppl import free_model, load_model

    if not Path(cfg.AWQ_MODEL).exists():
        sys.exit(f"❌ 找不到量化模型 {cfg.AWQ_MODEL}，先运行 scripts/quantize_awq.py")
    _, gb = cfg.gpu_info()
    if gb < 18:
        sys.exit(
            f"❌ KL 评估需要同时处理 FP16 模型，显存不足（当前 {gb:.1f} GB，建议 ≥18 GB）。\n"
            f"   这个对比在放不下 FP16 的卡上做不了 —— 这也正是量化的意义所在。"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.FP16_MODEL, trust_remote_code=True)
    chunks = build_chunks(tokenizer)

    print(f"\n{'=' * 60}\n第 1 趟：FP16 logits\n{'=' * 60}")
    dump_fp16_logits(chunks)

    print(f"\n{'=' * 60}\n第 2 趟：AWQ logits + 计算 KL\n{'=' * 60}")
    cache = torch.load(CACHE_PATH, map_location="cpu")
    model, _ = load_model("awq")
    from eval_ppl import model_device

    device = model_device(model)

    all_stats = []
    with torch.no_grad():
        for i, chunk in enumerate(cache["chunks"]):
            logits = model(chunk.unsqueeze(0).to(device)).logits[0].to(torch.float16).cpu()
            all_stats.append(kl_stats(cache["logits"][i], logits))
            print(f"   [{i + 1}/{len(cache['chunks'])}] KL={all_stats[-1]['kl_mean']:.5f} "
                  f"top1一致={all_stats[-1]['top1_agreement']:.4%}")

    free_model(model)

    # 按总位置数加权汇总各段
    total = sum(s["positions"] for s in all_stats)
    merged = {
        k: sum(s[k] * s["positions"] for s in all_stats) / total
        for k in ("kl_mean", "kl_p50", "kl_p95", "top1_agreement", "mean_abs_logit_diff")
    }
    merged["kl_max"] = max(s["kl_max"] for s in all_stats)
    merged["positions"] = total

    print(f"\n{'=' * 60}\n📉 KL 散度结果（FP16 vs AWQ W4A16）\n{'=' * 60}")
    print(f"  KL 均值   : {merged['kl_mean']:.5f}")
    print(f"  KL p95    : {merged['kl_p95']:.5f}")
    print(f"  Top-1 一致: {merged['top1_agreement']:.4%}")
    print(f"  平均 logit 绝对差: {merged['mean_abs_logit_diff']:.4f}")
    print("=" * 60)

    (cfg.RESULTS_DIR / "kl.json").write_text(
        json.dumps(
            {
                "dataset": "wikitext-2-raw-v1/test",
                "fp16_model": cfg.FP16_MODEL,
                "awq_model": str(cfg.AWQ_MODEL),
                "chunks": cfg.KL_CHUNKS,
                "chunk_len": cfg.KL_CHUNK_LEN,
                "env": cfg.env_snapshot(),
                "overall": merged,
                "per_chunk": all_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    CACHE_PATH.unlink(missing_ok=True)  # 1.2 GB 的中间产物，别留在盘上
    print("✅ 已写入 results/kl.json")

    gc.collect()


if __name__ == "__main__":
    main()

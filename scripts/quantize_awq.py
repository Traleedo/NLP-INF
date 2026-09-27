"""
AWQ W4A16 量化 Qwen2.5-7B-Instruct。

相对原版的三处关键修改：

1. 校准集路径修正（原为 files/sharegpt_1000.json，实际是 files/sharegpt-1000.json），
   且路径统一收敛到 config.py。

2. 校准文本构造方式重写。原版只取每条对话的第一个 human turn —— 那是
   「帮我写个 X」这种一句指令，token 数常常不到 30。AWQ 是靠校准数据上的
   activation 分布来搜索 per-channel 的缩放因子，序列过短会让这个估计方差
   极大，量化误差被显著放大。现在改成：按 Qwen 的 chat template 拼完整多轮
   对话，用 token 数截断到 CALIB_MAX_LEN。

3. 输出目录写 meta.json 记录实验元数据（校准集来源、样本数、配置、耗时），
   否则跑完一堆量化模型之后根本分不清哪个是哪个配置出来的。

用法：
    python scripts/quantize_awq.py
    W_BIT=4 Q_GROUP_SIZE=64 CALIB_NUM=256 python scripts/quantize_awq.py
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def load_sharegpt() -> list[dict]:
    if not cfg.SHAREGPT_PATH.exists():
        sys.exit(
            f"❌ 找不到校准集 {cfg.SHAREGPT_PATH}\n"
            f"   先运行 scripts/download_sharegpt.py，或用 SHAREGPT_PATH 环境变量指定路径。"
        )
    with open(cfg.SHAREGPT_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_calib_texts(tokenizer, data: list[dict], n: int, max_len: int, min_len: int) -> list[str]:
    """
    把 ShareGPT 的多轮对话拼成校准文本。

    注意这里**不区分 human/gpt**、保留 role 标记后用 chat template 拼接：
    AWQ 校准关心的是模型实际会看到的输入分布，而真实推理时模型看到的正是
    带 role 标记的完整上下文。只喂 user 提问反而偏离了部署分布。
    """
    role_map = {"human": "user", "gpt": "assistant"}
    texts: list[str] = []
    n_short = n_bad = 0

    for item in data:
        msgs = []
        for conv in item.get("conversations", []):
            role = role_map.get(conv.get("from"))
            value = conv.get("value")
            if role is None or not value:
                continue
            msgs.append({"role": role, "content": value})
        if not msgs:
            n_bad += 1
            continue

        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            # 少数样本 role 不严格交替会让 template 报错，退化处理，别让它中断整个量化
            n_bad += 1
            continue

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < min_len:
            n_short += 1
            continue
        if len(ids) > max_len:
            # 保头也保尾：尾部常含结论/代码，只截头会丢掉一半信息
            half = max_len // 2
            ids = ids[:half] + ids[-half:]
            text = tokenizer.decode(ids, skip_special_tokens=False)

        texts.append(text)
        if len(texts) >= n:
            break

    if not texts:
        sys.exit("❌ 没能从 ShareGPT 里提取出任何有效校准样本，检查数据格式。")

    lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    print(
        f"   校准样本 {len(texts)} 条 | token 数 min/mean/max = "
        f"{min(lens)}/{sum(lens) // len(lens)}/{max(lens)}"
    )
    print(f"   丢弃：过短 {n_short} 条，格式异常 {n_bad} 条")
    return texts


def main() -> None:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    cfg.ensure_dirs()
    out_dir = Path(cfg.AWQ_MODEL)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"⚠️  {out_dir} 已存在且非空，将被覆盖。")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"1. 读取 ShareGPT 校准集: {cfg.SHAREGPT_PATH}")
    data = load_sharegpt()
    print(f"   共 {len(data)} 条对话")

    print(f"2. 加载 tokenizer: {cfg.FP16_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.FP16_MODEL, trust_remote_code=True)

    print("3. 按 chat template 构造校准文本...")
    calib_texts = build_calib_texts(
        tokenizer, data, cfg.CALIB_NUM, cfg.CALIB_MAX_LEN, cfg.CALIB_MIN_LEN
    )

    print(f"4. 加载 FP16 模型（约 15 GB 显存）: {cfg.FP16_MODEL}")
    model = AutoAWQForCausalLM.from_pretrained(cfg.FP16_MODEL, trust_remote_code=True)

    print(f"5. 开始 AWQ 量化: {cfg.QUANT_CONFIG}")
    # max_calib_seq_len 在旧版 autoawq 里没有，用签名探测避免 TypeError
    kwargs = {"quant_config": cfg.QUANT_CONFIG, "calib_data": calib_texts}
    supported = inspect.signature(model.quantize).parameters
    if "max_calib_seq_len" in supported:
        kwargs["max_calib_seq_len"] = cfg.CALIB_MAX_LEN
    for opt in ("duo_scaling", "apply_clip"):
        if opt in supported:
            kwargs[opt] = getattr(cfg, opt.upper())

    t0 = time.time()
    model.quantize(tokenizer, **kwargs)
    elapsed = time.time() - t0
    print(f"   量化完成，耗时 {elapsed / 60:.1f} 分钟")

    print(f"6. 保存到 {out_dir}")
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    (out_dir / "quant_meta.json").write_text(
        json.dumps(
            {
                "source_model": cfg.FP16_MODEL,
                "quant_config": cfg.QUANT_CONFIG,
                "calib": {
                    "path": str(cfg.SHAREGPT_PATH),
                    "num_samples": len(calib_texts),
                    "max_len": cfg.CALIB_MAX_LEN,
                    "builder": "chat_template_multiturn",
                },
                "quantize_seconds": round(elapsed, 1),
                "env": cfg.env_snapshot(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("✅ 量化模型与 quant_meta.json 已保存")


if __name__ == "__main__":
    main()

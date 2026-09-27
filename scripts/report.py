"""
汇总所有实验结果，生成 results/REPORT.md + 图表。

设计前提：在云端 GPU 机器上跑完实验后，把 results/ 目录拿回本地也能出报告 ——
这个脚本只依赖已落盘的 json，不需要 GPU、vllm、awq。

输入：
    results/ppl_fp16.json, results/ppl_awq.json   (eval_ppl.py)
    results/kl.json                                (eval_kl.py)
    results/bench.json                             (benchmark.py)
输出：
    results/REPORT.md
    results/figures/*.png

图表配色取自 dataviz 规范的分类色板，已通过校验脚本（明暗双模式全部 PASS）：
  FP16 = 蓝 #2a78d6    AWQ = 橙 #eb6834
两根线的色觉障碍可分辨度 ΔE 24.7（阈值 8），远超要求。
另外用「圆点 vs 方块」做二次编码，不让颜色单独承载身份信息。

图表标注一律用英文：云端机器通常没装中文字体，中文会渲染成方块。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# ------------------------------------------------------------------ 设计令牌
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = {
    "fp16": {"color": "#2a78d6", "label": "FP16", "marker": "o"},
    "awq": {"color": "#eb6834", "label": "AWQ W4A16", "marker": "s"},
}

plt.rcParams.update({
    "figure.facecolor": PAGE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE,
    "font.family": ["DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "text.color": INK,
})


def style_axes(ax) -> None:
    """细网格 + 去掉上/右边框，让数据成为视觉焦点。"""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


# ------------------------------------------------------------------ 数据读取
def load(name: str) -> dict | None:
    p = cfg.RESULTS_DIR / name
    if not p.exists():
        print(f"   跳过 {name}（不存在）")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def first(d: dict, *keys, default=None):
    """vLLM 各版本字段名有差异，多候选回退。"""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def latency_ms(d: dict, metric: str) -> float | None:
    """取延迟指标（毫秒）。老版本 vLLM 用的是秒，做一次换算。"""
    v = first(d, f"{metric}_ms", f"{metric}_mean_ms")
    if v is not None:
        return v
    v = first(d, metric)  # 老版本：单位秒
    return v * 1000 if v is not None else None


def rate_sort_key(label: str) -> float:
    return math.inf if label == "inf" else float(label)


def rate_axis(bench: dict) -> tuple[list[str], dict[str, int]]:
    """
    全局速率刻度：返回 (刻度标签, 标签 → x 坐标 的映射)。

    必须用全局映射而不是每个模型各自的 enumerate 序号 ——
    两个模型跑的速率集合不一致时（比如某个速率失败了），
    局部序号会让数据点落到错误的刻度上。
    """
    labels = sorted({r["request_rate"] for runs in bench["results"].values()
                     for r in runs if "error" not in r}, key=rate_sort_key)
    disp = ["∞" if x == "inf" else x for x in labels]
    return disp, {lab: i for i, lab in enumerate(labels)}


def series_points(bench: dict, tag: str, getter) -> tuple[list[int], list[float]]:
    """按全局 x 坐标取一条曲线上的点，跳过取不到值的速率。"""
    _, xmap = rate_axis(bench)
    xs, ys = [], []
    for r in bench["results"].get(tag, []):
        if "error" in r:
            continue
        v = getter(r)
        if v is None:
            continue
        xs.append(xmap[r["request_rate"]])
        ys.append(v)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


# ------------------------------------------------------------------ 图表
def plot_throughput(bench: dict, out: Path) -> bool:
    """吞吐量 vs 请求速率：找饱和点。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    plotted = False

    for tag in bench["results"]:
        # 刻度按 log₂ 等距排布（1,2,4,8,16,32），∞ 接在最后一格：
        # 它代表「再翻一倍且不限速」，不是等比延长
        xs, ys = series_points(
            bench, tag, lambda r: first(r, "output_throughput", "output_token_throughput")
        )
        if not xs:
            continue

        s = SERIES[tag]
        ax.plot(xs, ys, color=s["color"], linewidth=2, marker=s["marker"],
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=s["label"], zorder=3)
        # 只给端点直接标注，不是每个点都标
        ax.annotate(f"{ys[-1]:,.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=INK_2, fontsize=9, va="center")
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    labels, _ = rate_axis(bench)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Request rate (req/s, log₂ spacing)")
    ax.set_ylabel("Output throughput (tokens/s)")
    ax.set_title("Throughput vs request rate", color=INK, fontsize=12, pad=12, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def plot_latency(bench: dict, out: Path) -> bool:
    """
    延迟 vs 请求速率。TTFT 和 TPOT 量纲不同 —— 分成两张子图，
    绝不画双 y 轴（那会凭空造出一种并不存在的相关性）。
    """
    panels = [
        ("ttft", "Time to first token (ms, p99)", "TTFT"),
        ("tpot", "Time per output token (ms, mean)", "TPOT"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
    plotted = False

    for ax, (key, ylabel, name) in zip(axes, panels):
        # TTFT 看 p99（尾延迟），TPOT 看均值（生成阶段的稳定速度）
        metric = f"p99_{key}" if key == "ttft" else f"mean_{key}"
        for tag in bench["results"]:
            xs, ys = series_points(bench, tag, lambda r, m=metric: latency_ms(r, m))
            if not xs:
                continue
            s = SERIES[tag]
            ax.plot(xs, ys, color=s["color"], linewidth=2, marker=s["marker"],
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=s["label"], zorder=3)
            ax.annotate(f"{ys[-1]:,.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(8, 0), color=INK_2, fontsize=9, va="center")
            plotted = True

        labels, _ = rate_axis(bench)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(name, color=INK, fontsize=11, pad=10, loc="left")
        style_axes(ax)

    if not plotted:
        plt.close(fig)
        return False

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Latency vs request rate (lower is better)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    return True


def plot_vram(bench: dict, out: Path) -> bool:
    """显存占用随时间变化（取请求速率最高的那次运行）。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    plotted = False

    for tag, runs in bench["results"].items():
        runs = [r for r in runs if "error" not in r and r.get("vram_samples")]
        if not runs:
            continue
        r = max(runs, key=lambda x: rate_sort_key(x["request_rate"]))
        xs = [t for t, _ in r["vram_samples"]]
        ys = [v / 1024 for _, v in r["vram_samples"]]  # MB → GB
        s = SERIES[tag]
        ax.plot(xs, ys, color=s["color"], linewidth=2, label=s["label"], zorder=3)
        ax.annotate(f"{ys[-1]:.1f} GB", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=INK_2, fontsize=9, va="center")
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("GPU memory in use (GB)")
    ax.set_title("GPU memory during peak-load run", color=INK, fontsize=12, pad=12, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ Markdown
def fmt(v, spec: str = ",.1f", dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def md_precision(ppl: dict, kl: dict) -> str:
    out = ["## 一、精度评估\n"]

    if ppl:
        f, a = ppl.get("fp16"), ppl.get("awq")
        if f and a:
            diff = a["ppl"] - f["ppl"]
            out += [
                "### 1.1 WikiText-2 困惑度（PPL，越低越好）\n",
                "| 模型 | PPL | 相对 FP16 | 评分 token 数 |",
                "|---|---|---|---|",
                f"| FP16 | {f['ppl']:.4f} | 基线 | {f['tokens_scored']:,} |",
                f"| AWQ W4A16 | {a['ppl']:.4f} | {diff / f['ppl'] * 100:+.2f}% | {a['tokens_scored']:,} |",
                "",
                f"绝对差值 **{diff:+.4f}**。",
                "",
                "> 说明：两边用同一套滑动窗口代码（`max_len=2048, stride=512`），"
                "loss 按实际参与计算的 token 数加权，保证可比性。",
                "",
            ]
        else:
            tag = "FP16" if f else "AWQ"
            only = f or a
            out += [f"仅测到 **{tag}** 一项，PPL = {only['ppl']:.4f}。",
                    "缺少对照组，无法给出精度损失 —— 需要在能放下 FP16 7B 的卡上补测。\n"]

    if kl:
        o = kl["overall"]
        out += [
            "### 1.2 输出分布偏移（FP16 vs AWQ）\n",
            "PPL 是整段文本平均后的单一标量，对「少数位置上分布被扭曲」不敏感。"
            "KL 散度逐位置度量分布偏移，是更灵敏的探针。\n",
            "| 指标 | 数值 | 含义 |",
            "|---|---|---|",
            f"| KL(FP16 ‖ AWQ) 均值 | {o['kl_mean']:.5f} | 平均分布偏移 |",
            f"| KL p95 | {o['kl_p95']:.5f} | 最差 5% 位置 |",
            f"| **Top-1 一致率** | **{o['top1_agreement']:.2%}** | 两模型预测同一个 token 的比例 |",
            f"| 平均 logit 绝对差 | {o['mean_abs_logit_diff']:.4f} | — |",
            f"| 统计位置数 | {o['positions']:,} | {kl['chunks']} 段 × {kl['chunk_len']} token |",
            "",
            f"Top-1 一致率 {o['top1_agreement']:.2%} 是这里最好解释的一个数："
            "**量化后绝大多数位置的最高概率 token 没有改变**，"
            "偏移主要体现在长尾分布上。\n",
            "> 注意：FP16 本身也是近似，此处度量的是「相对偏移」而非绝对误差。\n",
        ]
    return "\n".join(out)


def md_performance(bench: dict) -> str:
    out = ["## 二、性能评估\n"]

    for tag, runs in bench["results"].items():
        runs = sorted([r for r in runs if "error" not in r],
                      key=lambda r: rate_sort_key(r["request_rate"]))
        if not runs:
            continue
        s = SERIES[tag]
        out += [
            f"### {s['label']}\n",
            "| 请求速率 | 输出吞吐 (tok/s) | TTFT 均值 (ms) | TTFT p99 (ms) | TPOT 均值 (ms) | 显存峰值 (GB) | KV cache 峰值 |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in runs:
            kv = r.get("kv_cache_usage_peak")
            out.append(
                f"| {r['request_rate']} "
                f"| {fmt(first(r, 'output_throughput'), ',.1f')} "
                f"| {fmt(latency_ms(r, 'mean_ttft'))} "
                f"| {fmt(latency_ms(r, 'p99_ttft'))} "
                f"| {fmt(latency_ms(r, 'mean_tpot'))} "
                f"| {fmt((r.get('vram_peak_mb') or 0) / 1024, '.2f') if r.get('vram_peak_mb') else '—'} "
                f"| {fmt(kv * 100, '.1f') + '%' if kv is not None else '—'} |"
            )
        out.append("")

        idle, peak = runs[0].get("idle_vram_mb"), max(
            (r.get("vram_peak_mb") or 0) for r in runs
        )
        if idle and peak:
            out.append(f"空载显存 {idle / 1024:.2f} GB → 峰值 {peak / 1024:.2f} GB"
                       f"（模型 + KV cache 约 {(peak - idle) / 1024:.2f} GB）。\n")

    # 对比小结
    both = [t for t in bench["results"] if bench["results"][t]]
    if len(both) == 2:
        out += ["### 对比小结\n", "| 对比项 | FP16 | AWQ | 变化 |", "|---|---|---|---|"]
        for metric, key, spec in [
            ("输出吞吐（峰值）", "output_throughput", ",.1f"),
            ("显存峰值", "vram_peak_mb", ",.0f"),
        ]:
            vals = {}
            for tag, runs in bench["results"].items():
                rs = [r for r in runs if "error" not in r]
                if metric.startswith("显存"):
                    vals[tag] = max((r.get("vram_peak_mb") or 0) for r in rs) if rs else None
                else:
                    vals[tag] = max((first(r, key) or 0) for r in rs) if rs else None
            f, a = vals.get("fp16"), vals.get("awq")
            if f and a:
                unit = " GB" if metric.startswith("显存") else " tok/s"
                out.append(f"| {metric} | {fmt(f / 1024 if unit == ' GB' else f, spec)}{unit} "
                           f"| {fmt(a / 1024 if unit == ' GB' else a, spec)}{unit} "
                           f"| {(a - f) / f * 100:+.1f}% |")
        out.append("")
    return "\n".join(out)


def md_env(bench: dict | None, ppl: dict | None, kl: dict | None) -> str:
    env = None
    for src in (bench or {}, ppl or {}, kl or {}):
        if src.get("env"):
            env = src["env"]
            break
    if not env:
        return ""
    rows = [
        ("GPU", env.get("gpu")),
        ("显存", f"{env.get('gpu_mem_gb')} GB" if env.get("gpu_mem_gb") else None),
        ("CUDA", env.get("cuda")),
        ("PyTorch", env.get("torch")),
        ("vLLM", env.get("vllm")),
        ("transformers", env.get("transformers")),
        ("autoawq", env.get("awq")),
        ("Python", env.get("python")),
    ]
    out = ["## 实验环境\n", "| 项 | 值 |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows if v]
    out.append(
        "\n> 这组参数必须随结果一起报告 —— 换了硬件或版本，数字就没有可比性。\n"
    )
    return "\n".join(out)


def main() -> None:
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("读取实验结果...")
    bench = load("bench.json")
    fp16_ppl, awq_ppl = load("ppl_fp16.json"), load("ppl_awq.json")
    kl = load("kl.json")
    ppl = {"fp16": fp16_ppl, "awq": awq_ppl} if (fp16_ppl or awq_ppl) else None

    if not any([bench, ppl, kl]):
        sys.exit("❌ results/ 下没有任何结果文件。先跑 eval_ppl.py / benchmark.py。")

    print("生成图表...")
    figs = {}
    if bench:
        for name, fn in [
            ("throughput_vs_rate", plot_throughput),
            ("latency_vs_rate", plot_latency),
            ("vram_over_time", plot_vram),
        ]:
            path = cfg.FIG_DIR / f"{name}.png"
            if fn(bench, path):
                figs[name] = path
                print(f"   ✓ {path.name}")

    print("生成 REPORT.md...")
    md = ["# Qwen2.5-7B-Instruct AWQ 量化 × vLLM 推理加速 实验报告\n"]
    md.append(md_env(bench, fp16_ppl, kl))
    md.append(md_precision(ppl, kl))
    if bench:
        parts = [md_performance(bench)]
        if figs:
            parts.append("## 三、图表\n")
            for name, path in figs.items():
                rel = path.relative_to(cfg.RESULTS_DIR).as_posix()
                parts.append(f"![{name}]({rel})\n")
        md.append("\n".join(parts))

    md.append(
        "## 实验配置\n\n```json\n"
        + json.dumps((bench or {}).get("config", {}), ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    md.append(
        "## 局限与后续\n\n"
        "- PPL 与 KL 都基于 WikiText-2，与 ShareGPT 的对话分布有差异；"
        "指令跟随能力的退化还需要 IFEval 一类的任务指标佐证。\n"
        "- 压测是单卡单实例，未涉及张量并行与多副本调度。\n"
        "- 请求速率用 log₂ 等距刻度，`∞` 一格代表「再翻一倍且不限速」，"
        "不是等比延长。\n"
    )

    out = cfg.RESULTS_DIR / "REPORT.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n✅ 报告已生成：{out}")
    if figs:
        print(f"   图表目录：{cfg.FIG_DIR}")


if __name__ == "__main__":
    main()

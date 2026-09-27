#!/usr/bin/env bash
#
# 一条命令跑完整条链路：量化 → 精度评估 → 压测 → 出报告。
#
# 每一步的产物都在 results/ 下，中断了可以从任意一步重跑：
#   ./scripts/run_all.sh --skip-quant          # 量化模型已存在，跳过量化
#   ./scripts/run_all.sh --skip-quant --skip-eval
#
# 注意显存：FP16 7B 约需 15 GB，AWQ 约 4.5 GB。
# 显存不够时脚本会自动跳过 FP16 基线并明确提示（但没有基线就没有对比结论）。

set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
SKIP_QUANT=0
SKIP_EVAL=0
SKIP_BENCH=0

for arg in "$@"; do
    case "$arg" in
        --skip-quant) SKIP_QUANT=1 ;;
        --skip-eval)  SKIP_EVAL=1 ;;
        --skip-bench) SKIP_BENCH=1 ;;
        -h|--help)    sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "未知参数: $arg"; exit 1 ;;
    esac
done

step() { echo; echo "════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════"; }

# ---------------------------------------------------------------- 0. 数据
step "0/4  准备数据集"
if [ ! -f files/sharegpt-1000.json ]; then
    "$PY" scripts/download_sharegpt.py
else
    echo "files/sharegpt-1000.json 已存在，跳过下载。"
fi

# ---------------------------------------------------------------- 1. 量化
if [ "$SKIP_QUANT" -eq 0 ]; then
    step "1/4  AWQ W4A16 量化"
    "$PY" scripts/quantize_awq.py
else
    echo "跳过量化。"
fi

# ---------------------------------------------------------------- 2. 精度
if [ "$SKIP_EVAL" -eq 0 ]; then
    step "2/4  精度评估（PPL + KL 散度）"
    # 串行执行：eval_ppl.py 内部自己决定跑哪些模型并逐个释放显存
    "$PY" scripts/eval_ppl.py || echo "⚠️  PPL 评估未完整通过"
    "$PY" scripts/eval_kl.py  || echo "⚠️  KL 评估跳过（通常是因为显存放在不下 FP16）"
else
    echo "跳过精度评估。"
fi

# ---------------------------------------------------------------- 3. 压测
if [ "$SKIP_BENCH" -eq 0 ]; then
    step "3/4  vLLM 压测（FP16 vs AWQ）"
    "$PY" scripts/benchmark.py
else
    echo "跳过压测。"
fi

# ---------------------------------------------------------------- 4. 报告
step "4/4  生成报告"
"$PY" scripts/report.py

echo
echo "🎉 全部完成。报告：results/REPORT.md"

# Qwen2.5-7B-Instruct 量化 × vLLM 推理加速

对 Qwen2.5-7B-Instruct 做 AWQ W4A16 量化，在 vLLM 上测「精度损失 ↔ 吞吐/显存收益」的权衡。

覆盖完整链路：**数据准备 → 量化 → 精度评估 → 压测 → 自动生成报告**。

---

## 快速开始

```bash
pip install -r requirements.txt

# 一条命令跑完（量化 → PPL/KL → 压测 → 报告）
./scripts/run_all.sh

# 已经量化过，只补跑后面的
./scripts/run_all.sh --skip-quant
```

产物：

| 路径 | 内容 |
|---|---|
| `results/REPORT.md` | **实验报告**（表格 + 图表，最终交付物） |
| `results/figures/*.png` | 报告的配图 |
| `results/bench.json` | 压测原始结果 |
| `results/raw/*.json` | `vllm bench serve` 每次运行的原始输出 |
| `results/logs/*.log` | vLLM 服务端日志 |

报告脚本只依赖落盘的 json，**不需要 GPU** —— 在云端跑完实验后，把 `results/` 拿回本地也能重新出报告。

---

## 脚本

| 脚本 | 作用 |
|---|---|
| [config.py](scripts/config.py) | 所有路径与参数的唯一来源，支持环境变量覆盖 |
| [download_sharegpt.py](scripts/download_sharegpt.py) | 下载 ShareGPT 并截取，供校准与压测使用 |
| [quantize_awq.py](scripts/quantize_awq.py) | AWQ W4A16 量化，输出模型 + 实验元数据 |
| [eval_ppl.py](scripts/eval_ppl.py) | WikiText-2 困惑度，FP16 vs AWQ |
| [eval_kl.py](scripts/eval_kl.py) | FP16 与 AWQ 的输出分布 KL 散度、Top-1 一致率 |
| [benchmark.py](scripts/benchmark.py) | 启动 vLLM 服务并扫描请求速率，采集吞吐/延迟/显存 |
| [report.py](scripts/report.py) | 汇总所有结果，生成报告与图表 |

---

## 实验设计

### 精度：为什么不止看 PPL

PPL 是整段文本平均后的单一标量，对「少数 token 上分布被严重扭曲」不敏感 ——
大部分 token 预测得很准，均值就把问题抹平了。所以这里用三个层次的指标：

| 指标 | 回答的问题 | 脚本 |
|---|---|---|
| PPL | 整体语言建模能力掉了多少 | `eval_ppl.py` |
| KL 散度 | 逐位置的输出分布偏移有多大 | `eval_kl.py` |
| Top-1 一致率 | 量化后有多少比例的预测 token 变了 | `eval_kl.py` |

### 性能：扫请求速率而不是单点

只跑一个请求速率只能得到一个点，看不出**吞吐饱和点**和**延迟拐点**在哪。
默认扫描 `request_rate ∈ {1, 2, 4, 8, 16, 32, inf}`，在 `inf`（不限速）下压出系统的实际容量上限。

同时采集**显存峰值**和 **KV cache 使用率** —— 量化最大的收益是显存，
省下的显存会转化为更大的 KV cache、更高的并发，不测显存等于漏掉一半结论。

### 做消融实验

所有参数都走环境变量，不用改代码：

```bash
# 扫描量化配置
W_BIT=4 Q_GROUP_SIZE=32  python scripts/quantize_awq.py
W_BIT=4 Q_GROUP_SIZE=128 python scripts/quantize_awq.py
CALIB_NUM=32  python scripts/quantize_awq.py
CALIB_NUM=256 python scripts/quantize_awq.py

# 前缀缓存对照（多轮对话场景下收益明显，ShareGPT 里 77% 是多轮）
NO_PREFIX_CACHING=1 python scripts/benchmark.py --models awq
```

建议的消融维度：
- **量化配置**：`W_BIT` × `Q_GROUP_SIZE` × 校准样本数 → 画精度-吞吐 Pareto 前沿
- **校准集**：ShareGPT（对话）vs WikiText（书面语），看分布匹配重不重要
- **前缀缓存**：多轮场景下的 TTFT 收益
- **请求速率**：吞吐-延迟曲线

---

## 实验环境

结果必须连同环境一起报告，否则换个硬件/版本数字就没有可比性。
`config.py` 会自动采集 GPU 型号、显存、CUDA、PyTorch、vLLM、transformers 版本，
写进每个结果 json，并汇总到报告开头。

在云端机器上跑之前请确认：

| 项目 | 要求 |
|---|---|
| 显存 | FP16 7B 约 **15 GB**，AWQ 约 **4.5 GB**；做对比实验建议 ≥24 GB |
| CUDA | 与 vLLM/torch 版本匹配 |

> 只有 6 GB 级别的卡放不下 FP16 基线，脚本会自动跳过并提示。
> 没有基线就只能报「AWQ 能跑」，报不出「AWQ 快多少、掉多少精度」——
> 那就不构成对比实验。

---

## 实现中修掉的问题

早期版本是从官方示例直接拼起来的，有几个会让实验结论不成立的问题：

| 问题 | 后果 |
|---|---|
| 压测启动的是 FP16 原始模型，量化模型从未被压测 | **「量化 → 加速」链路根本没闭环**，数据无法支撑结论 |
| 校准集只取每条对话的首轮提问（常不足 30 token） | AWQ 的 scale 搜索方差过大，量化误差被系统性放大 |
| PPL 对每个窗口的 loss 直接取平均 | 各窗口参与计算的 token 数不同，短窗口权重偏高 |
| 三个脚本写了三个不同的数据集路径（下划线/连字符不一致） | 必然 FileNotFoundError |
| 健康检查是 `while ! curl; do sleep 2; done` 无超时 | 服务 OOM 起不来时脚本永久挂起 |
| 连续加载 FP16 与 AWQ 两个模型且中间不释放显存 | 必然 OOM |
| `AutoAWQForCausalLM` 上取 `.device` | AttributeError（它是包装类，没有这个属性） |

---

## 局限

- 精度评估基于 WikiText-2，与 ShareGPT 的对话分布有差异。指令跟随能力的退化
  还需要 IFEval / MMLU 一类的任务指标佐证，PPL 和 KL 都看不出来。
- 压测是单卡单实例，未涉及张量并行、多副本调度与真实线上流量特征。
- 尚未与 GPTQ / SmoothQuant / FP8 等其他量化方案横向对比。

---

## 参考

- [vLLM 文档](https://docs.vllm.ai/)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [AWQ 论文 (MLSys 2024)](https://arxiv.org/abs/2306.00978)

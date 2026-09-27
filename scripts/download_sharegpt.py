"""
下载 ShareGPT 数据集并截取前 N 条，供 AWQ 校准和压测使用。

替代原 download_scripts/share-gpt_download.py：那个版本下载部分整段被注释掉了，
只剩一个「读文件→截前 1000 条」的片段，且输入输出路径写死在代码里。
现在下载和截取都在这里，路径统一走 config。

用法：
    python scripts/download_sharegpt.py
    python scripts/download_sharegpt.py --num 2000
    python scripts/download_sharegpt.py --input /path/already/downloaded.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="下载并截取 ShareGPT 数据集")
    ap.add_argument("--num", type=int, default=1000, help="截取多少条（默认 1000）")
    ap.add_argument("--input", default=None, help="已有原始 json 的路径，给了就跳过下载")
    args = ap.parse_args()

    # 国内机器走镜像，否则 huggingface.co 大概率连不上
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if args.input:
        src = Path(args.input)
        if not src.exists():
            sys.exit(f"❌ 找不到 {src}")
    else:
        from huggingface_hub import hf_hub_download

        print(f"下载 {REPO_ID}/{FILENAME} ...")
        try:
            src = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset"))
        except Exception as e:
            sys.exit(
                f"❌ 下载失败：{e}\n"
                f"   可以手动下载后传给 --input 参数，或用 HF_ENDPOINT 指定镜像。"
            )
        print(f"   已下载到 {src}")

    print(f"读取 {src} ...")
    data = json.loads(src.read_text(encoding="utf-8"))
    print(f"   原始 {len(data)} 条对话，截取前 {args.num} 条")

    subset = data[: args.num]
    # 只保留需要的字段，文件小一半以上
    keep = [{"id": d["id"], "conversations": d["conversations"]}
            for d in subset if d.get("conversations")]

    cfg.SHAREGPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.SHAREGPT_PATH.write_text(
        json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 已写入 {cfg.SHAREGPT_PATH}（{len(keep)} 条，"
          f"{cfg.SHAREGPT_PATH.stat().st_size / 1024**2:.1f} MB）")


if __name__ == "__main__":
    main()

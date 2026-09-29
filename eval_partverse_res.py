"""
PartVerse RES（指代分割）评测入口脚本。

⚠️ 与现有 test.py / batch_test.py 完全独立。这两个脚本不会被本入口触发，反之亦然。

典型用法：

    # 评测单个 instance（所有 part × 所有 caption_type）
    python eval_partverse_res.py --uid <UID>

    # 评测整个采样子集（来自 scripts/sample_partverse_subset.py 输出）
    python eval_partverse_res.py --uids-file scripts/partverse_subset.json

    # 只跑 brief caption，每 instance 限制 3 条 query
    python eval_partverse_res.py --uid <UID> --caption-types brief --max-queries 3
"""

import argparse
import json
import os
import sys
import time

# 把项目根目录加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from res_pipeline import evaluate_uids


def parse_args():
    p = argparse.ArgumentParser(description="PartVerse RES Evaluation Entry")
    p.add_argument("--config", default="config/config.yaml",
                   help="主配置文件路径（含 res: 段落）")
    p.add_argument("--uid", default=None,
                   help="单个 PartVerse instance UID")
    p.add_argument("--uids-file", default=None,
                   help="包含 uid 列表的 JSON 文件（数组 or {\"uids\": [...]}）")
    p.add_argument("--caption-types", default=None,
                   help="逗号分隔，例如 'brief' 或 'brief,detailed'。默认从 config 读取")
    p.add_argument("--max-queries", type=int, default=None,
                   help="每个 instance 最大 query 数（用于快速 smoke test）")
    p.add_argument("--gpu", type=int, default=-1,
                   help="GPU 索引；-1 表示自动选 cuda:0")
    p.add_argument("--out-root", default=None,
                   help="结果输出根目录；默认使用 config.res.results_dir")
    p.add_argument("--limit", type=int, default=None,
                   help="只评测前 N 个 uid（与 --uids-file 配合用于 smoke test）")
    return p.parse_args()


def load_uids(args) -> list:
    if args.uid:
        return [args.uid]

    if args.uids_file:
        with open(args.uids_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            uids = data
        elif isinstance(data, dict):
            uids = data.get("uids") or data.get("instances") or list(data.keys())
        else:
            raise ValueError(f"不识别的 uids_file 格式: {args.uids_file}")

        # 兼容 [{"uid": "...", ...}, ...] 形式
        if uids and isinstance(uids[0], dict):
            uids = [d.get("uid") for d in uids if d.get("uid")]
        return [str(u) for u in uids if u]

    raise ValueError("必须传 --uid 或 --uids-file 之一")


def main():
    args = parse_args()
    uids = load_uids(args)
    if args.limit:
        uids = uids[:args.limit]
    print(f"[eval_partverse_res] 待评测 instance 数: {len(uids)}")

    caption_types = None
    if args.caption_types:
        caption_types = [c.strip() for c in args.caption_types.split(",") if c.strip()]

    t0 = time.time()
    agg = evaluate_uids(
        config_path=args.config,
        uids=uids,
        gpu_index=args.gpu,
        caption_types=caption_types,
        max_queries=args.max_queries,
        out_root=args.out_root,
    )
    elapsed = time.time() - t0
    print(f"\n[eval_partverse_res] 总耗时 {elapsed/60:.2f} min "
          f"({elapsed/max(1, agg.num_queries):.2f} s/query)")


if __name__ == "__main__":
    main()

"""
PartVerse 评测子集采样脚本（双桶 stratified sampling）。

目的：从完整 PartVerse 集合中按"part 数量"和"caption 长度"两个维度做分桶，
确保子集覆盖各种复杂度，便于做 RES 消融实验时少量 instance 即可代表整体。

桶定义：
- part 数桶（num_parts_buckets）：[1-3, 4-7, 8-12, 13+]
- 平均 detailed caption 长度桶（caption_length_buckets）：[short, medium, long]

每个 (part_bucket × caption_bucket) 抽 N 个 instance。

输出 JSON 形如：
    {
        "metadata": {...},
        "uids": ["uid_a", "uid_b", ...],
        "details": [
            {"uid": "uid_a", "num_parts": 5, "avg_detailed_chars": 137,
             "part_bucket": "4-7", "caption_bucket": "medium"},
            ...
        ]
    }

⚠️ 不修改任何已有文件。
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# 将项目根加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_THIS_DIR)
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)


# ----------------------------------------------------------------------
# 桶定义
# ----------------------------------------------------------------------
PART_COUNT_BUCKETS: List[Tuple[str, int, int]] = [
    ("1-3", 1, 3),
    ("4-7", 4, 7),
    ("8-12", 8, 12),
    ("13+", 13, 10**6),
]

CAPTION_LENGTH_BUCKETS: List[Tuple[str, int, int]] = [
    ("short", 0, 80),
    ("medium", 81, 200),
    ("long", 201, 10**6),
]


def parse_args():
    p = argparse.ArgumentParser(description="PartVerse RES Subset Sampler")
    p.add_argument("--text-captions", required=True,
                   help="text_captions.json 路径（PartVerse 数据集根 /text_captions.json）")
    p.add_argument("--per-bucket", type=int, default=2,
                   help="每个 (part_bucket × caption_bucket) 采样数")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    p.add_argument("--out", default="scripts/partverse_subset.json",
                   help="输出文件路径")
    p.add_argument("--exclude-uids-file", default=None,
                   help="可选：包含已使用 uid 的 JSON 列表，避免重采")
    p.add_argument("--require-mesh-dir", default=None,
                   help="可选：仅保留 normalized_glbs/<uid>.glb 真实存在的 uid")
    return p.parse_args()


def assign_part_bucket(num_parts: int) -> str:
    for name, lo, hi in PART_COUNT_BUCKETS:
        if lo <= num_parts <= hi:
            return name
    return PART_COUNT_BUCKETS[-1][0]


def assign_caption_bucket(avg_chars: float) -> str:
    for name, lo, hi in CAPTION_LENGTH_BUCKETS:
        if lo <= avg_chars <= hi:
            return name
    return CAPTION_LENGTH_BUCKETS[-1][0]


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if not os.path.exists(args.text_captions):
        print(f"[sample_partverse_subset] 找不到 {args.text_captions}")
        sys.exit(1)

    with open(args.text_captions, "r", encoding="utf-8") as f:
        all_captions: Dict[str, Dict[str, List[str]]] = json.load(f)

    excluded = set()
    if args.exclude_uids_file and os.path.exists(args.exclude_uids_file):
        with open(args.exclude_uids_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            excluded = set(data)
        elif isinstance(data, dict):
            excluded = set(data.get("uids", []))

    # ---------- 第一步：统计每个 instance 的 num_parts 与 avg_detailed_chars ----------
    instance_stats: Dict[str, Dict] = {}
    for uid, parts in all_captions.items():
        if uid in excluded:
            continue
        if args.require_mesh_dir:
            glb_path = os.path.join(args.require_mesh_dir, f"{uid}.glb")
            if not os.path.exists(glb_path):
                # 兼容嵌套布局
                alt = os.path.join(args.require_mesh_dir, uid, f"{uid}.glb")
                if not os.path.exists(alt):
                    continue
        if not isinstance(parts, dict) or not parts:
            continue
        num_parts = len(parts)
        detailed_lens: List[int] = []
        brief_lens: List[int] = []
        for part_id, cap_pair in parts.items():
            if not isinstance(cap_pair, list) or len(cap_pair) < 2:
                continue
            brief, detailed = str(cap_pair[0]), str(cap_pair[1])
            brief_lens.append(len(brief))
            detailed_lens.append(len(detailed))
        if not detailed_lens:
            continue
        avg_d = sum(detailed_lens) / len(detailed_lens)
        avg_b = sum(brief_lens) / max(1, len(brief_lens))

        instance_stats[uid] = {
            "uid": uid,
            "num_parts": num_parts,
            "avg_detailed_chars": avg_d,
            "avg_brief_chars": avg_b,
            "part_bucket": assign_part_bucket(num_parts),
            "caption_bucket": assign_caption_bucket(avg_d),
        }

    print(f"[sample_partverse_subset] 候选 instance: {len(instance_stats)}")

    # ---------- 第二步：分桶 ----------
    buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for uid, info in instance_stats.items():
        buckets[(info["part_bucket"], info["caption_bucket"])].append(uid)

    # 打印桶分布
    print("\n[bucket distribution]")
    for pb_name, _, _ in PART_COUNT_BUCKETS:
        for cb_name, _, _ in CAPTION_LENGTH_BUCKETS:
            n = len(buckets.get((pb_name, cb_name), []))
            print(f"  parts={pb_name:5s}  cap={cb_name:6s}  n={n}")

    # ---------- 第三步：按桶采样 ----------
    selected_uids: List[str] = []
    for pb_name, _, _ in PART_COUNT_BUCKETS:
        for cb_name, _, _ in CAPTION_LENGTH_BUCKETS:
            pool = list(buckets.get((pb_name, cb_name), []))
            if not pool:
                continue
            rng.shuffle(pool)
            picked = pool[:args.per_bucket]
            selected_uids.extend(picked)

    print(f"\n[sample_partverse_subset] 采样到 {len(selected_uids)} 个 instance")

    # ---------- 第四步：写文件 ----------
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "metadata": {
            "source": args.text_captions,
            "seed": args.seed,
            "per_bucket": args.per_bucket,
            "part_count_buckets": [[name, lo, hi] for name, lo, hi in PART_COUNT_BUCKETS],
            "caption_length_buckets": [[name, lo, hi] for name, lo, hi in CAPTION_LENGTH_BUCKETS],
            "total_candidates": len(instance_stats),
            "total_selected": len(selected_uids),
        },
        "uids": selected_uids,
        "details": [instance_stats[u] for u in selected_uids],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[sample_partverse_subset] 已写入 {args.out}")


if __name__ == "__main__":
    main()

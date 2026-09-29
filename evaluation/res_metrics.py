"""
RES（指代分割）专用评测指标。

与 evaluation/metrics.py（多类语义分割 IoU）完全独立，互不影响。

输入约定：
- pred_mask:  (N,) bool/int —— 模型对单条 query 的目标点云掩码
- gt_mask:    (N,) bool/int —— GT 中该 query 对应 part 的点云掩码
其中 N 是同一 instance 的点云大小。

输出指标（按 RES 文献惯例）：
- IoU                 单条 query 的 IoU
- aggregated_IoU      多条 query 平均 IoU（即 mIoU 在 RES 语境下的称呼）
- cIoU (cumulative)   累计交并比：sum(intersections) / sum(unions)
- Precision@0.5/0.7   IoU >= 阈值 的 query 占比
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RESPerQueryResult:
    """单条 RES query 的评测结果。"""
    uid: str
    part_id: str
    caption_type: str
    intersection: int
    pred_count: int
    gt_count: int
    union: int

    @property
    def iou(self) -> float:
        return self.intersection / self.union if self.union > 0 else float("nan")


@dataclass
class RESAggregateResult:
    """多 query 聚合结果。"""
    num_queries: int = 0
    valid_queries: int = 0       # gt_count > 0 的有效 query 数
    aggregated_iou: float = 0.0  # 各有效 query IoU 的平均
    cumulative_iou: float = 0.0  # sum(intersection) / sum(union)
    precision_at_05: float = 0.0
    precision_at_07: float = 0.0
    by_caption_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_query: List[RESPerQueryResult] = field(default_factory=list)


def compute_query_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    计算单条 query 的 (intersection, union, pred_count, gt_count)。
    长度不匹配时返回 (0, 0, 0, 0) 并打印警告。
    """
    pred_b = np.asarray(pred_mask, dtype=bool).ravel()
    gt_b = np.asarray(gt_mask, dtype=bool).ravel()
    if pred_b.shape != gt_b.shape:
        print(f"[RES Metrics] WARN pred shape {pred_b.shape} != gt shape {gt_b.shape}; returning zeros.")
        return 0, 0, 0, 0
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    return inter, union, int(pred_b.sum()), int(gt_b.sum())


def aggregate_results(per_query: List[RESPerQueryResult]) -> RESAggregateResult:
    """聚合 per-query 结果为 RES 整体指标。"""
    out = RESAggregateResult(per_query=list(per_query), num_queries=len(per_query))
    if not per_query:
        return out

    valid = [q for q in per_query if q.gt_count > 0]
    out.valid_queries = len(valid)
    if not valid:
        return out

    ious = [q.iou for q in valid]
    out.aggregated_iou = float(np.mean(ious))

    total_inter = sum(q.intersection for q in valid)
    total_union = sum(q.union for q in valid)
    out.cumulative_iou = total_inter / total_union if total_union > 0 else 0.0

    out.precision_at_05 = float(np.mean([1.0 if q.iou >= 0.5 else 0.0 for q in valid]))
    out.precision_at_07 = float(np.mean([1.0 if q.iou >= 0.7 else 0.0 for q in valid]))

    # 按 caption type 聚合
    by_ct: Dict[str, List[RESPerQueryResult]] = {}
    for q in valid:
        by_ct.setdefault(q.caption_type, []).append(q)
    for ct, qs in by_ct.items():
        ct_ious = [q.iou for q in qs]
        ct_inter = sum(q.intersection for q in qs)
        ct_union = sum(q.union for q in qs)
        out.by_caption_type[ct] = {
            "num_queries": len(qs),
            "aggregated_iou": float(np.mean(ct_ious)),
            "cumulative_iou": ct_inter / ct_union if ct_union > 0 else 0.0,
            "precision_at_05": float(np.mean([1.0 if iou >= 0.5 else 0.0 for iou in ct_ious])),
            "precision_at_07": float(np.mean([1.0 if iou >= 0.7 else 0.0 for iou in ct_ious])),
        }

    return out


def format_aggregate_summary(agg: RESAggregateResult) -> str:
    """把聚合结果格式化为多行字符串，便于日志/最终报告打印。"""
    lines = []
    lines.append("=" * 60)
    lines.append("RES Evaluation Summary")
    lines.append("=" * 60)
    lines.append(f"Total queries:  {agg.num_queries}")
    lines.append(f"Valid (gt>0):   {agg.valid_queries}")
    lines.append("")
    lines.append(f"aggregated IoU:  {agg.aggregated_iou:.4f}")
    lines.append(f"cumulative IoU:  {agg.cumulative_iou:.4f}")
    lines.append(f"Pr@0.5:          {agg.precision_at_05:.4f}")
    lines.append(f"Pr@0.7:          {agg.precision_at_07:.4f}")
    if agg.by_caption_type:
        lines.append("")
        lines.append("By caption_type:")
        for ct, info in sorted(agg.by_caption_type.items()):
            lines.append(
                f"  {ct:10s}  n={info['num_queries']:3d}  "
                f"aIoU={info['aggregated_iou']:.4f}  cIoU={info['cumulative_iou']:.4f}  "
                f"Pr@.5={info['precision_at_05']:.4f}  Pr@.7={info['precision_at_07']:.4f}"
            )
    lines.append("=" * 60)
    return "\n".join(lines)


def to_serializable(agg: RESAggregateResult) -> Dict:
    """转 JSON 可序列化字典。"""
    return {
        "num_queries": agg.num_queries,
        "valid_queries": agg.valid_queries,
        "aggregated_iou": agg.aggregated_iou,
        "cumulative_iou": agg.cumulative_iou,
        "precision_at_05": agg.precision_at_05,
        "precision_at_07": agg.precision_at_07,
        "by_caption_type": agg.by_caption_type,
        "per_query": [
            {
                "uid": q.uid,
                "part_id": q.part_id,
                "caption_type": q.caption_type,
                "intersection": q.intersection,
                "union": q.union,
                "pred_count": q.pred_count,
                "gt_count": q.gt_count,
                "iou": q.iou,
            }
            for q in agg.per_query
        ],
    }

import numpy as np
import os
from typing import Dict, Optional, Tuple

class IoUEvaluator:
    def __init__(self):
        pass

    def evaluate(self, pred_labels: np.ndarray, gt_path: str, class_mapping: Dict[str, int]) -> Tuple[Dict[str, float], float]:
        """
        计算各个类别的 IoU 以及整体的 mIoU（PartNetE 格式）。

        :param pred_labels: 模型预测的标签数组 (N,)
        :param gt_path: 真实标签 .npy 文件的路径（PartNetE: dict with 'semantic_seg'）
        :param class_mapping: 类别名称到 ID 的映射字典
        :return: (各个类别的 IoU 字典, mIoU 值)
        """
        if not os.path.exists(gt_path):
            print(f"[Evaluation] 找不到真实标签文件: {gt_path}")
            return {}, 0.0

        try:
            gt_data = np.load(gt_path, allow_pickle=True).item()
            gt_labels = gt_data.get('semantic_seg', None)
            if gt_labels is None:
                print("[Evaluation] 在 label.npy 中找不到 'semantic_seg' 键。")
                return {}, 0.0
        except Exception as e:
            print(f"[Evaluation] 加载真实标签失败: {e}")
            return {}, 0.0

        return self._compute_iou(pred_labels, gt_labels, class_mapping)

    def evaluate_with_gt_array(
        self,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        class_mapping: Dict[str, int],
    ) -> Tuple[Dict[str, float], float]:
        """
        直接传入 GT 标签数组进行评估（通用格式，适配 PartObjaverse-Tiny 等）。
        """
        return self._compute_iou(pred_labels, gt_labels, class_mapping)

    @staticmethod
    def load_gt_labels(gt_path: str) -> Optional[np.ndarray]:
        """
        灵活加载 GT 标签文件，自动检测格式：
        - PartNetE 格式：dict with 'semantic_seg' key
        - PartObjaverse-Tiny 格式：纯 int 数组 / 含 'labels' key 的 dict
        Returns:
            (N,) int64 数组，加载失败返回 None
        """
        if not os.path.exists(gt_path):
            return None
        try:
            raw = np.load(gt_path, allow_pickle=True)
            if isinstance(raw, np.ndarray) and raw.ndim == 0:
                raw = raw.item()
            if isinstance(raw, dict):
                for key in ("semantic_seg", "labels", "label"):
                    if key in raw:
                        return np.asarray(raw[key], dtype=np.int64)
                return None
            return np.asarray(raw, dtype=np.int64).ravel()
        except Exception as e:
            print(f"[Evaluation] 加载 GT 失败 ({gt_path}): {e}")
            return None

    def _compute_iou(
        self,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        class_mapping: Dict[str, int],
    ) -> Tuple[Dict[str, float], float]:
        """IoU / mIoU 核心计算逻辑（被 evaluate 和 evaluate_with_gt_array 共用）"""
        if len(pred_labels) != len(gt_labels):
            print(f"[Evaluation] 警告: 预测点数 ({len(pred_labels)}) 与真实点数 ({len(gt_labels)}) 不一致！")
            return {}, 0.0

        print("\n" + "="*60)
        print("📊 定量评估结果 (Intersection over Union) - [忽略 GT 中不存在的类别]")
        print("="*60)

        ious = {}
        eval_classes = {name: cid for name, cid in class_mapping.items() if name != "background"}

        for class_name, class_id in eval_classes.items():
            pred_mask = (pred_labels == class_id)
            gt_mask = (gt_labels == class_id)
            gt_points = gt_mask.sum()

            if gt_points == 0:
                ious[class_name] = float('nan')
                pred_points = pred_mask.sum()
                if pred_points > 0:
                    print(f"{class_name.ljust(15)}: [Ignored] GT 中不存在此标注 (但预测了 {pred_points} 个假阳性点)")
                else:
                    print(f"{class_name.ljust(15)}: [Ignored] GT 和 Pred 中均不存在")
                continue

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            ious[class_name] = iou
            print(f"{class_name.ljust(15)}: {iou:.4f}  (交集: {intersection:<6}, 并集: {union:<6} | GT 真实点数: {gt_points})")

        valid_ious = [iou for iou in ious.values() if not np.isnan(iou)]
        miou = float('nan')
        if valid_ious:
            miou = np.mean(valid_ious)
            print("-" * 60)
            print(f"{'mIoU'.ljust(15)}: {miou:.4f}  (参与计算的有效类别数: {len(valid_ious)})")
        else:
            print("[Evaluation] 无法计算 mIoU，没有包含 GT 的有效类别。")
        print("="*60 + "\n")

        return ious, miou

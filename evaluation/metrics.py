import numpy as np

class Evaluator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def compute_mIoU(self, pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
        """
        计算平均交并比 (mIoU)
        """
        ious = []
        for c in range(self.num_classes):
            pred_c = (pred_labels == c)
            gt_c = (gt_labels == c)
            
            intersection = np.logical_and(pred_c, gt_c).sum()
            union = np.logical_or(pred_c, gt_c).sum()
            
            if union == 0:
                if intersection == 0 and gt_c.sum() == 0:
                    ious.append(1.0) # 此类别无 GT 也无预测，视为正确
            else:
                ious.append(intersection / union)
                
        if len(ious) == 0:
            return 0.0
        return np.mean(ious)

    def compute_mAP50(self, pred_instances: list, gt_instances: list) -> float:
        """
        计算 3D 实例分割的 mAP@50 (简化版示意)
        """
        # TODO: 完整实现 3D mAP@0.5 计算逻辑
        return 0.0

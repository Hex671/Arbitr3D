import numpy as np
import torch
from typing import List
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAMAutoSegmenter:
    def __init__(self, model_path: str, model_type: str = "vit_h"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM model ({model_type}) from {model_path}...")
        self.sam = sam_model_registry[model_type](checkpoint=model_path)
        self.sam.to(device=self.device)
        self.model_type = model_type
        self.model_path = model_path

        # Initialize Automatic Mask Generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=30,  # 从30增加到48，撒点更密，更容易打中细小物体 (如wheel)
            pred_iou_thresh=0.86,  # 降低对掩码质量的要求，0.86 足够放行边缘不那么清晰的轮子
            stability_score_thresh=0.88,  # 降低稳定性阈值，接受稍微波动的掩码
            crop_n_layers=2,  # 保持图像切片再搜索，这对小物体很有用
            crop_n_points_downscale_factor=2, # 切片后的撒点密度衰减因子
            min_mask_region_area=100,  # 从250降低到100，防止小轮子被当成噪点抛弃
        )

    def generate_masks_automatically(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        使用 SAM 的 AutomaticMaskGenerator 生成全图无语义掩码，
        并使用基于层级包含关系的算法剔除跨部件的“大包围”冗余掩码。
        返回: 每个视角对应的 mask 列表 (H, W) 布尔矩阵
        """
        all_masks = []
        for idx, img in enumerate(images):
            print(f"SAM Auto Segmenting image {idx+1}/{len(images)}...")
            # SAM expects RGB images
            masks_dicts = self.mask_generator.generate(img)

            if not masks_dicts:
                all_masks.append([])
                continue

            # 1. 按照面积从小到大排序 (从小掩码开始处理，保留最精细的部件)
            masks_dicts = sorted(masks_dicts, key=lambda x: x['area'])

            valid_masks = []

            # 2. 遍历每一个掩码，检查它是否是一个“冗余的父节点”
            for i, current_mask_dict in enumerate(masks_dicts):
                current_mask = current_mask_dict['segmentation']
                current_area = current_mask_dict['area']

                # 统计当前大掩码内部，已经被之前保留的小掩码覆盖了多少面积
                covered_area = 0
                if valid_masks:
                    # 将所有已经保留的小掩码合并成一个 mask
                    # 使用 np.any 替代 np.logical_or.reduce 提高性能
                    combined_small_masks = np.any(valid_masks, axis=0)
                    # 计算当前大掩码与这些小掩码的交集
                    intersection = np.logical_and(current_mask, combined_small_masks)
                    covered_area = np.sum(intersection)

                # 计算覆盖率：内部小掩码面积 / 当前大掩码面积
                coverage_ratio = covered_area / current_area if current_area > 0 else 0

                # 核心判断：如果这个掩码有超过 70% 的区域已经被更小的精细掩码覆盖了，
                # 说明它是一个跨部件的“大包围”父节点（例如包住了整个椅子的掩码），我们丢弃它！
                # 否则，说明它是一个独立的部件（哪怕它面积很大，比如完整的椅背），我们保留它。
                if coverage_ratio < 0.70:
                    valid_masks.append(current_mask)

            # 3. 过滤掉面积过小（可能是噪点）的极小掩码
            # 降低丢弃阈值，防止细小轮子(wheel)被当做噪点扔掉，改为 > 100
            final_masks = []
            for mask in valid_masks:
                if np.sum(mask) > 100:
                    final_masks.append(mask)

            all_masks.append(final_masks)

        return all_masks

    def get_predictor(self):
        """
        获取 SAM Predictor，用于局部重分割。
        """
        if not hasattr(self, '_predictor') or self._predictor is None:
            self._predictor = SamPredictor(self.sam)
        return self._predictor

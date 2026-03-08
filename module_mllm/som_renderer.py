from typing import List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datatypes import Entity3D

class SoMRenderer:
    def __init__(self):
        # 预定义高对比度颜色列表
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]

    def render_som_collage(self, images: List[np.ndarray], index_maps: List[np.ndarray], 
                           entities: List[Entity3D], key_views: List[int]) -> Image.Image:
        """
        在关键视角的 RGB 图上，根据实体的掩码位置覆盖半透明遮罩，并绘制中心点数字标签。
        最后拼贴成一张大图。
        """
        annotated_images = []
        
        pt_to_eid = {}
        for e in entities:
            for pt in e.point_indices:
                pt_to_eid[pt] = e.entity_id

        for view_idx in key_views:
            img = images[view_idx].copy()
            idx_map = index_maps[view_idx].squeeze(-1)
            
            overlay = img.copy()
            
            for i, entity in enumerate(entities):
                eid = entity.entity_id
                color = self.colors[i % len(self.colors)]
                
                mask = np.zeros(idx_map.shape, dtype=bool)
                for y in range(idx_map.shape[0]):
                    for x in range(idx_map.shape[1]):
                        pt = idx_map[y, x]
                        if pt != -1 and pt_to_eid.get(pt) == eid:
                            mask[y, x] = True
                            
                if not np.any(mask):
                    continue
                    
                # 应用半透明遮罩
                overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
                
                # 计算质心
                y_coords, x_coords = np.where(mask)
                cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
                
                # 画标记背景圆
                cv2.circle(overlay, (cx, cy), 15, color, -1)
                cv2.circle(overlay, (cx, cy), 15, (255, 255, 255), 2)
                # 写数字
                cv2.putText(overlay, str(eid), (cx - 7, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            annotated_images.append(Image.fromarray(overlay.astype(np.uint8)))
            
        return self._create_collage(annotated_images)
        
    def _create_collage(self, pil_images: List[Image.Image]) -> Image.Image:
        if not pil_images:
            return Image.new('RGB', (800, 800))
        num_images = len(pil_images)
        cols = 2
        rows = math.ceil(num_images / cols)
        
        w, h = pil_images[0].size
        collage = Image.new('RGB', (w * cols, h * rows))
        
        for i, img in enumerate(pil_images):
            row = i // cols
            col = i % cols
            collage.paste(img, (col * w, row * h))
            
        return collage

"""
深度图可视化增强工具。

将原始浮点深度图转换为高对比度的可视化图像，
仅在物体有效区域内归一化，背景保持纯黑，
并应用 CLAHE 进一步增强局部对比度。
"""

import cv2
import numpy as np


def enhance_depth_map(depth_map: np.ndarray, colormap: bool = False) -> np.ndarray:
    """
    将原始深度图转换为高对比度的 uint8 可视化图像（BGR 格式）。

    步骤：
      1. 提取有效像素（depth > 0），仅在有效范围内归一化到 0-255
      2. 反转：越近越亮（符合 prompt 描述）
      3. 应用 CLAHE 增强局部对比度
      4. 背景区域保持纯黑
      5. 可选：应用伪彩色映射（TURBO colormap）

    Args:
        depth_map: (H, W) float32/float64 原始深度图
        colormap: 若为 True，输出 TURBO 伪彩色；否则输出灰度转 BGR

    Returns:
        (H, W, 3) uint8 BGR 图像
    """
    depth = depth_map.astype(np.float64)
    # 确保是 2D (H, W)
    if depth.ndim == 3:
        depth = depth[:, :, 0] if depth.shape[2] == 1 else np.mean(depth, axis=2)
    valid_mask = depth > 0

    if not np.any(valid_mask):
        # 全背景，返回黑图
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    # 1. 仅在有效区域内归一化
    d_min = np.min(depth[valid_mask])
    d_max = np.max(depth[valid_mask])

    normalized = np.zeros_like(depth)
    if d_max - d_min > 1e-6:
        normalized[valid_mask] = (depth[valid_mask] - d_min) / (d_max - d_min)
    else:
        normalized[valid_mask] = 0.5

    # 2. 反转：越近 (depth 越小) → 值越大 → 越亮
    normalized[valid_mask] = 1.0 - normalized[valid_mask]

    # 转为 uint8
    gray = np.zeros_like(depth, dtype=np.uint8)
    gray[valid_mask] = (normalized[valid_mask] * 255).astype(np.uint8)

    # 3. CLAHE 增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 背景强制纯黑
    gray[~valid_mask] = 0

    # 4. 输出
    if colormap:
        bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        bgr[~valid_mask] = [0, 0, 0]
    else:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return bgr

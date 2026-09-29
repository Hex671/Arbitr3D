"""
掩码后处理脚本：处理 SAM 分割后的掩码，解决掩码之间的交集和包含关系。

核心逻辑：
1. 存在交集的掩码：将交集部分单独提取为新掩码，原掩码去除交集
2. 存在包含关系的掩码：小掩码保留，大掩码去除内部小掩码区域
最终目标：将所有掩码处理成彼此之间不存在交集的状态
"""

import os
import sys
import numpy as np
import cv2
from typing import List, Tuple, Optional

# 确保能找到 Arbitr3D 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个二值掩码的 IoU (Intersection over Union)
    mask1, mask2: 布尔数组，形状为 (H, W)
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def compute_intersection_area(mask1: np.ndarray, mask2: np.ndarray) -> int:
    """计算两个掩码的交集面积"""
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection)


def is_contained(small_mask: np.ndarray, large_mask: np.ndarray, threshold: float = 0.95) -> bool:
    """
    判断 small_mask 是否被 large_mask 包含
    threshold: 小掩码中被大掩码覆盖的比例，超过这个比例则认为是被包含
    """
    small_area = np.sum(small_mask)
    if small_area == 0:
        return False

    intersection = np.logical_and(small_mask, large_mask)
    overlap_ratio = np.sum(intersection) / small_area
    return overlap_ratio >= threshold


def process_masks(masks: List[np.ndarray], iou_threshold: float = 0.01, containment_threshold: float = 0.95, min_area: int = 200) -> List[np.ndarray]:
    """
    处理掩码列表，消除彼此之间的交集。

    处理策略：
    1. 对于有交集的两个掩码 A 和 B：
       - 交集部分 I = A ∩ B 作为独立掩码
       - A' = A - I (A去除交集部分)
       - B' = B - I (B去除交集部分)
       - 如果 A' 或 B' 面积过小（小于原面积的10%），则不保留

    2. 对于包含关系（完全包含）：
       - 小掩码保留
       - 大掩码去除内部小掩码区域，保留外部部分

    参数:
        masks: 输入掩码列表，每个掩码为 (H, W) 的布尔数组
        iou_threshold: IoU 超过这个阈值认为有交集需要处理（默认0.01，处理所有有交集的掩码）
        containment_threshold: 包含关系判定阈值（默认0.95，即小掩码95%以上被大掩码覆盖则认为包含）

    返回:
        处理后的掩码列表，彼此之间不存在交集
    """
    if not masks:
        return []

    # 转换为 numpy 数组列表，确保类型一致
    masks = [m.astype(bool) for m in masks]

    # 按面积从小到大排序（先处理小掩码，再处理大掩码）
    areas = [np.sum(m) for m in masks]
    sorted_indices = np.argsort(areas)  # 升序排列，小的在前
    masks = [masks[i] for i in sorted_indices]

    processed_masks = []

    for i, current_mask in enumerate(masks):
        if np.sum(current_mask) == 0:
            continue

        # 复制当前掩码作为基础
        remaining_mask = current_mask.copy()

        # 检查当前掩码与已处理掩码的关系
        new_intersections = []

        for j, processed_mask in enumerate(processed_masks):
            intersection = np.logical_and(remaining_mask, processed_mask)
            intersection_area = np.sum(intersection)

            if intersection_area == 0:
                continue

            # 检查包含关系
            current_contains_processed = is_contained(processed_mask, remaining_mask, containment_threshold)
            processed_contains_current = is_contained(remaining_mask, processed_mask, containment_threshold)

            if current_contains_processed:
                # 当前掩码包含已处理掩码（当前是大掩码，已处理是小掩码）
                # 保持已处理掩码不变，从当前掩码去除交集
                remaining_mask = np.logical_and(remaining_mask, ~intersection)
            elif processed_contains_current:
                # 已处理掩码包含当前掩码（已处理是大掩码，当前是小掩码）
                # 保持当前掩码不变，从已处理掩码去除交集区域
                processed_masks[j] = np.logical_and(processed_mask, ~intersection)
            else:
                # 部分重叠（非包含关系）
                # 将交集部分单独提取为新掩码，两个原始掩码都去除交集
                if np.sum(intersection) > 100:
                    new_intersections.append(intersection)
                remaining_mask = np.logical_and(remaining_mask, ~intersection)
                processed_masks[j] = np.logical_and(processed_mask, ~intersection)

        # 添加所有新发现的交集掩码
        for inter_mask in new_intersections:
            if np.sum(inter_mask) > 100:
                processed_masks.append(inter_mask)

        # 如果处理后的掩码面积足够大，添加到结果中
        if np.sum(remaining_mask) >= min_area:
            processed_masks.append(remaining_mask)

    # 迭代合并，直到没有重叠
    max_iterations = 10
    for _ in range(max_iterations):
        if len(processed_masks) <= 1:
            break

        changed = False
        final_masks = []

        for i, mask in enumerate(processed_masks):
            if np.sum(mask) < min_area:
                continue

            merged = False
            for j in range(len(final_masks)):
                if compute_intersection_area(mask, final_masks[j]) > 50:
                    # 合并两个掩码
                    final_masks[j] = np.logical_or(mask, final_masks[j])
                    merged = True
                    changed = True
                    break

            if not merged:
                final_masks.append(mask)

        processed_masks = final_masks

        if not changed:
            break

    # 过滤太小的掩码
    processed_masks = [m for m in processed_masks if np.sum(m) >= min_area]

    return processed_masks


def apply_masks_to_image(image: np.ndarray, masks: List[np.ndarray], colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    将掩码应用到图像上并可视化

    参数:
        image: 原始图像 (H, W, 3) 或 (H, W)，如果是灰度图会自动转换为 BGR
        masks: 掩码列表，每个掩码为 (H, W) 的布尔数组
        colors: 每个掩码的颜色列表，如果为 None 则自动生成随机颜色

    返回:
        带有掩码可视化的图像
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    result = image.copy()

    if colors is None:
        # 生成随机颜色
        np.random.seed(42)  # 固定种子保证可复现
        colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
                  for _ in range(len(masks))]

    for i, mask in enumerate(masks):
        if np.sum(mask) == 0:
            continue

        color = colors[i % len(colors)]

        # 创建彩色掩码层
        colored_mask = np.zeros_like(result)
        colored_mask[mask] = color

        # 叠加到原图
        result = cv2.addWeighted(result, 1.0, colored_mask, 0.5, 0)

        # 绘制边界
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)

        # 添加掩码编号标签
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(result, (cx, cy), 5, color, -1)
            cv2.putText(result, str(i), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result


def create_mask_grid(masks: List[np.ndarray], canvas_size: Tuple[int, int], n_cols: int = 4) -> np.ndarray:
    """
    将多个掩码平铺到一张大图上显示

    参数:
        masks: 掩码列表
        canvas_size: 每个小掩码图的尺寸 (H, W)
        n_cols: 每行显示的掩码数量

    返回:
        拼合后的大图
    """
    h, w = canvas_size
    n_masks = len(masks)
    n_rows = (n_masks + n_cols - 1) // n_cols

    # 创建黑色画布
    grid = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)

    np.random.seed(42)

    for i, mask in enumerate(masks):
        row = i // n_cols
        col = i % n_cols

        # 在对应位置创建掩码图
        mask_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if np.sum(mask) > 0:
            # 生成随机颜色
            color = (np.random.randint(80, 255), np.random.randint(80, 255), np.random.randint(80, 255))

            # 缩放掩码到画布大小（如果需要）
            if mask.shape != (h, w):
                scaled_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                scaled_mask = mask.astype(np.uint8)

            mask_canvas[scaled_mask > 0] = color

            # 添加编号
            cv2.putText(mask_canvas, f"#{i} ({np.sum(scaled_mask)}px)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 放置到网格中
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = mask_canvas

    return grid


def visualize_mask_processing(original_image: np.ndarray,
                               original_masks: List[np.ndarray],
                               processed_masks: List[np.ndarray],
                               output_path: str):
    """
    可视化掩码处理前后的对比

    参数:
        original_image: 原始图像
        original_masks: 处理前的掩码列表
        processed_masks: 处理后的掩码列表
        output_path: 输出路径
    """
    # 转换灰度图到 BGR
    if len(original_image.shape) == 2:
        vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = original_image.copy()

    # 应用原始掩码
    original_vis = apply_masks_to_image(vis_image.copy(), original_masks)

    # 应用处理后掩码
    processed_vis = apply_masks_to_image(vis_image.copy(), processed_masks)

    # 创建掩码网格对比
    original_grid = create_mask_grid(original_masks, (vis_image.shape[0] // 2, vis_image.shape[1] // 2))
    processed_grid = create_mask_grid(processed_masks, (vis_image.shape[0] // 2, vis_image.shape[1] // 2))

    # 水平拼合
    top_row = np.hstack([original_vis, processed_vis])
    bottom_row = np.hstack([original_grid, processed_grid])

    final_vis = np.vstack([top_row, bottom_row])

    # 添加标题
    h, w = final_vis.shape[:2]
    header = np.zeros((50, w, 3), dtype=np.uint8)
    cv2.putText(header, "Left: Original Masks | Right: Processed Masks | Bottom: Mask Grids",
               (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    final_vis = np.vstack([header, final_vis])

    cv2.imwrite(output_path, final_vis)
    print(f"可视化结果已保存到: {output_path}")


def batch_process_masks(image_paths: List[str],
                        masks_list: List[List[np.ndarray]],
                        output_dir: str,
                        save_individual: bool = True,
                        save_visualization: bool = True) -> List[List[np.ndarray]]:
    """
    批量处理多张图片的掩码

    参数:
        image_paths: 图片路径列表
        masks_list: 每个图片对应的掩码列表
        output_dir: 输出目录
        save_individual: 是否保存每个处理后的掩码
        save_visualization: 是否保存可视化对比图

    返回:
        处理后的掩码列表
    """
    os.makedirs(output_dir, exist_ok=True)

    processed_masks_list = []

    for i, (img_path, masks) in enumerate(zip(image_paths, masks_list)):
        print(f"\n处理图片 {i + 1}/{len(image_paths)}: {os.path.basename(img_path)}")
        print(f"  原始掩码数量: {len(masks)}")

        # 处理掩码
        processed = process_masks(masks)
        print(f"  处理后掩码数量: {len(processed)}")

        processed_masks_list.append(processed)

        if save_visualization and masks:
            # 加载原图
            img = cv2.imread(img_path)
            if img is not None:
                vis_path = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}_mask_comparison.png")
                visualize_mask_processing(img, masks, processed, vis_path)

        if save_individual:
            # 保存每个掩码为单独的 numpy 文件
            mask_dir = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}_masks")
            os.makedirs(mask_dir, exist_ok=True)
            for j, mask in enumerate(processed):
                np.save(os.path.join(mask_dir, f"mask_{j:03d}.npy"), mask)

    return processed_masks_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="处理 SAM 分割后的掩码，去除交集和包含关系")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入图片路径或目录")
    parser.add_argument("--masks", "-m", type=str, required=True, help="掩码文件路径（.npy格式，包含掩码列表）")
    parser.add_argument("--output", "-o", type=str, default="./processed_masks", help="输出目录")
    parser.add_argument("--iou-threshold", type=float, default=0.01, help="IoU阈值，超过此值认为有交集")
    parser.add_argument("--containment-threshold", type=float, default=0.95, help="包含关系判定阈值")
    parser.add_argument("--visualize", "-v", action="store_true", help="是否保存可视化对比图")

    args = parser.parse_args()

    # 加载掩码
    masks = np.load(args.masks, allow_pickle=True)
    if isinstance(masks, list):
        masks = masks

    # 加载图片
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 处理掩码
    if masks:
        processed = process_masks(masks, args.iou_threshold, args.containment_threshold)

        print(f"\n原始掩码数量: {len(masks)}")
        print(f"处理后掩码数量: {len(processed)}")

        if args.visualize and image_paths:
            img = cv2.imread(image_paths[0])
            if img is not None:
                vis_path = os.path.join(args.output, "mask_comparison.png")
                visualize_mask_processing(img, masks, processed, vis_path)

        # 保存处理后的掩码
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "processed_masks.npy")
        np.save(output_path, processed)
        print(f"处理后的掩码已保存到: {output_path}")

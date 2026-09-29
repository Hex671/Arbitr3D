import os
import sys
import numpy as np
import cv2
import math
from typing import Tuple

# 确保能找到 Arbitr3D 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入集中配置
from scripts.config import (
    DATASET_BASE_PATH, BASE_DIR, get_category, SEM_KNOWLEDGE_IMAGES_DIR,
    RENDER_RESOLUTION, RENDER_RADIUS, SEM_VIEWS, SEM_VIEW_LABELS, NUM_SEM_VIEWS
)

from data_prep.dataloader import PointCloudDataset
from module_render.renderer import MultiViewRenderer

def get_four_classic_cameras(radius: float = RENDER_RADIUS) -> Tuple[np.ndarray, np.ndarray]:
    """获取 4 个最经典的差异化视角，用于给 MLLM 提取类别整体特征"""
    try:
        from pytorch3d.renderer import look_at_view_transform
        has_pytorch3d = True
    except ImportError:
        has_pytorch3d = False

    # 从配置读取四视角参数
    views = SEM_VIEWS

    if has_pytorch3d:
        from pytorch3d.renderer import look_at_view_transform
        R_list, T_list = [], []
        for view in views:
            R, T = look_at_view_transform(dist=radius, elev=view[0], azim=view[1])
            R_list.append(R[0].numpy())
            T_list.append(T[0].numpy())

        camera_positions, camera_rotations = [], []
        for r, t in zip(R_list, T_list):
            r_mat = r.T
            t_vec = -r @ t
            camera_positions.append(t_vec)
            camera_rotations.append(r_mat)
        return np.array(camera_positions), np.array(camera_rotations)
    else:
        # Fallback for Open3D
        camera_positions, R_matrices = [], []
        for view in views:
            elev, azim = math.radians(view[0]), math.radians(view[1])
            y = math.sin(elev)
            radius_at_y = math.cos(elev)
            x = math.sin(azim) * radius_at_y
            z = math.cos(azim) * radius_at_y
            pos = np.array([x * radius, y * radius, z * radius])
            camera_positions.append(pos)

            forward = -pos
            forward = forward / np.linalg.norm(forward)
            up = np.array([0, 1, 0])
            if np.abs(np.dot(forward, up)) > 0.999: up = np.array([1, 0, 0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            real_up = np.cross(right, forward)
            R = np.stack([right, real_up, -forward], axis=-1)
            R_matrices.append(R)
        return np.array(camera_positions), np.array(R_matrices)

def create_four_view_collage(images: list) -> np.ndarray:
    """将4张图拼接成 2x2 的一张大图，并在图上用黑底白字标出对应的物理视角"""
    labels = SEM_VIEW_LABELS

    labeled_images = []
    for img, label in zip(images, labels):
        # 渲染器吐出的是 RGB，由于 OpenCV 使用 BGR，需要转换
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 增加黑色底色条以便看清文字，防止白色模型或背景干扰
        cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(img_bgr, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        labeled_images.append(img_bgr)

    # 水平拼接成行
    top_row = np.hstack((labeled_images[0], labeled_images[1]))
    bottom_row = np.hstack((labeled_images[2], labeled_images[3]))

    # 垂直拼接成 2x2 网格
    collage = np.vstack((top_row, bottom_row))
    return collage

def main():
    # ==========================================
    # 1. 核心配置区 (从 config.py 集中读取)
    # ==========================================
    CATEGORY = get_category()

    # 图片集中存放的目录
    OUTPUT_DIR = os.path.join(SEM_KNOWLEDGE_IMAGES_DIR, CATEGORY)
    # ==========================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataloader = PointCloudDataset(DATASET_BASE_PATH)

    # 为了防止 4 张图拼接后过大无法塞给 MLLM，这里把单张图分辨率稍微降低为 512x512
    # 拼接后总图大小为 1024x1024，非常适合 GPT-4V 吞吐
    renderer = MultiViewRenderer(resolution=RENDER_RESOLUTION)

    category_dir = os.path.join(DATASET_BASE_PATH, CATEGORY)
    if not os.path.exists(category_dir):
        print(f"❌ 找不到数据集目录: {category_dir}")
        return

    # 获取所有实例 ID
    object_ids = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
    object_ids.sort()

    print(f"🔍 找到 {len(object_ids)} 个 {CATEGORY} 实例。准备生成四视图拼图并保存至: {OUTPUT_DIR}\n")
    print(f"📷 使用 {NUM_SEM_VIEWS} 个经典视角进行渲染")

    # 获取固定的 4 个经典相机视角矩阵
    cam_pos, cam_rot = get_four_classic_cameras(radius=RENDER_RADIUS)

    for idx, obj_id in enumerate(object_ids):
        # 如果中断过，可以加上 if idx < START_INDEX: continue
        print(f"[{idx+1}/{len(object_ids)}] 正在渲染实例: {obj_id} ...", end=" ")

        # 注意: dataloader 内部会自动加上 DATASET_BASE_PATH
        pc_path = f"{CATEGORY}/{obj_id}/pc.ply"

        try:
            pcd = dataloader.load_point_cloud(pc_path)
            norm_pcd = dataloader.normalize_pc(pcd)
            pc_xyz = np.asarray(norm_pcd.points)
            pc_colors = dataloader.extract_color(norm_pcd)

            # 渲染四视图
            render_output = renderer.render(pc_xyz, pc_colors, cam_pos, cam_rot)

            # 将4张图片拼成一张带标签的大图
            collage = create_four_view_collage(render_output.images)

            # 以实例 ID 命名并保存
            save_path = os.path.join(OUTPUT_DIR, f"{obj_id}_{NUM_SEM_VIEWS}views.png")
            cv2.imwrite(save_path, collage)
            print("✅ 完成")

        except Exception as e:
            print(f"❌ 失败: {e}")

    print("\n🎉 所有实例渲染并拼接完成！")

if __name__ == "__main__":
    main()

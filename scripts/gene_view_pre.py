import os
import sys
import numpy as np
import cv2
import math

# 确保能找到 Arbitr3D 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入集中配置
from scripts.config import (
    DATASET_BASE_PATH, BASE_DIR, get_category, VIEW_KNOWLEDGE_IMAGES_DIR,
    RENDER_RESOLUTION, RENDER_RADIUS, get_views_info, NUM_VIEWS
)

from data_prep.dataloader import PointCloudDataset
from module_render.renderer import MultiViewRenderer
from module_render.camera_poses import generate_sphere_cameras

def create_collage(images: list) -> np.ndarray:
    """
    将图片列表（最多4张）拼合成一个 2x2 的网格。
    不足 4 张的部分用黑色图片填充。
    所有图片假设具有相同的分辨率。
    """
    if not images:
        raise ValueError("提供的图片列表为空！")

    h, w, c = images[0].shape
    blank_img = np.zeros((h, w, c), dtype=np.uint8)

    # 填充至刚好 4 张图
    while len(images) < 4:
        images.append(blank_img)

    # 水平拼接成行
    top_row = np.hstack((images[0], images[1]))
    bottom_row = np.hstack((images[2], images[3]))

    # 垂直拼接成 2x2 网格
    collage = np.vstack((top_row, bottom_row))
    return collage

def main():
    # ==========================================
    # 1. 核心配置区 (从 config.py 集中读取)
    # ==========================================
    CATEGORY = get_category()

    # 推荐的存储路径：与 sem_knowledge_images 同级
    OUTPUT_BASE_DIR = os.path.join(VIEW_KNOWLEDGE_IMAGES_DIR, CATEGORY)
    # ==========================================

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    dataloader = PointCloudDataset(DATASET_BASE_PATH)

    # 单张图 512x512，拼出 2x2 后总计 1024x1024，适合 MLLM 吞吐
    renderer = MultiViewRenderer(resolution=RENDER_RESOLUTION)

    category_dir = os.path.join(DATASET_BASE_PATH, CATEGORY)
    if not os.path.exists(category_dir):
        print(f"❌ 找不到数据集目录: {category_dir}")
        return

    # 获取所有实例 ID
    object_ids = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
    object_ids.sort()

    print(f"🔍 找到 {len(object_ids)} 个 {CATEGORY} 实例。")
    print(f"📷 正在获取 {NUM_VIEWS} 个预设相机视角...")

    # 从配置获取视角信息，生成相机位姿
    views_info = get_views_info(NUM_VIEWS)
    cam_pos, cam_rot = generate_sphere_cameras(K=NUM_VIEWS, radius=RENDER_RADIUS)

    # 我们需要一个数据结构来临时存储：视角ID -> 实例图片列表 的映射关系
    # 结构: { view_idx: [(obj_id_1, img_bgr_1), (obj_id_2, img_bgr_2), ...] }
    view_images_dict = {v_idx: [] for v_idx in range(NUM_VIEWS)}

    print("\n▶️ [阶段一] 开始逐个实例渲染所有视角...")
    for idx, obj_id in enumerate(object_ids):
        print(f"  [{idx+1}/{len(object_ids)}] 渲染实例: {obj_id} ...", end=" ", flush=True)
        pc_path = f"{CATEGORY}/{obj_id}/pc.ply"

        try:
            pcd = dataloader.load_point_cloud(pc_path)
            norm_pcd = dataloader.normalize_pc(pcd)
            pc_xyz = np.asarray(norm_pcd.points)
            pc_colors = dataloader.extract_color(norm_pcd)

            # 渲染出当前实例在 16 个视角下的所有图片
            render_output = renderer.render(pc_xyz, pc_colors, cam_pos, cam_rot)

            for v_idx in range(NUM_VIEWS):
                img_rgb = render_output.images[v_idx]
                # OpenGL 渲染出来是 RGB，OpenCV 需要 BGR 用于保存和打标签
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # 在单张图上打上实例 ID 标签，方便 MLLM 辨认
                cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 35), (0, 0, 0), -1)
                cv2.putText(img_bgr, f"Instance: {obj_id}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 存入对应的视角列表里
                view_images_dict[v_idx].append(img_bgr)

            print("✅ 成功")
        except Exception as e:
            print(f"❌ 失败: {e}")

    print("\n▶️ [阶段二] 开始按照 4个/图 的规格将各视角图片进行拼合并保存...")

    for v_idx in range(NUM_VIEWS):
        # 建立独立视角的文件夹
        view_dir = os.path.join(OUTPUT_BASE_DIR, f"View_{v_idx:02d}")
        os.makedirs(view_dir, exist_ok=True)

        all_imgs_for_this_view = view_images_dict[v_idx]
        total_imgs = len(all_imgs_for_this_view)

        # 将图片 4个分为一组进行拼合
        batch_id = 0
        for i in range(0, total_imgs, 4):
            batch_imgs = all_imgs_for_this_view[i:i+4]
            collage = create_collage(batch_imgs)

            # 保存该拼图：视角路径 / batch_XX.png
            save_path = os.path.join(view_dir, f"batch_{batch_id:03d}.png")
            cv2.imwrite(save_path, collage)
            batch_id += 1

        print(f"  📂 视角 View_{v_idx:02d} 生成了 {batch_id} 张拼图，保存在: {view_dir}")

    print("\n🎉 视角维度渲染与拼接大功告成！接下来可以使用大模型去逐视角总结知识了。")

if __name__ == "__main__":
    main()

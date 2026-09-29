import open3d as o3d
import numpy as np

class Visualizer3D:
    def __init__(self):
        # 预定义颜色表 (使用易于区分的颜色)
        self.color_names = [
            "红色 (Red)", "绿色 (Green)", "蓝色 (Blue)", "黄色 (Yellow)",
            "紫色 (Purple)", "青色 (Cyan)", "橙色 (Orange)", "粉色 (Pink)",
            "棕色 (Brown)", "青柠色 (Lime)"
        ]
        self.colors = np.array([
            [1.0, 0.0, 0.0], # Red
            [0.0, 1.0, 0.0], # Green
            [0.0, 0.0, 1.0], # Blue
            [1.0, 1.0, 0.0], # Yellow
            [0.5, 0.0, 0.5], # Purple
            [0.0, 1.0, 1.0], # Cyan
            [1.0, 0.5, 0.0], # Orange
            [1.0, 0.75, 0.8], # Pink
            [0.6, 0.3, 0.0], # Brown
            [0.75, 1.0, 0.0] # Lime
        ])

        # 如果类别超过 10 个，补充随机颜色
        if len(self.colors) < 100:
            np.random.seed(42)
            extra_colors = np.random.rand(100 - len(self.colors), 3)
            self.colors = np.vstack((self.colors, extra_colors))

    def get_color_name(self, color_idx):
        if color_idx < len(self.color_names):
            return self.color_names[color_idx]
        return f"随机颜色_{color_idx}"

    def visualize_3d_result(self, pcd: o3d.geometry.PointCloud, point_labels: np.ndarray = None, save_path: str = "output_result.ply", class_to_id: dict = None):
        """
        渲染最终分割好的彩色 3D 点云，并保存到本地
        """
        import os
        if point_labels is not None:
            # 收集需要渲染为灰色的标签 ID（background 和 unlabeled）
            gray_label_ids = set()
            if class_to_id is not None:
                for name, cid in class_to_id.items():
                    if name in ("background", "unlabeled"):
                        gray_label_ids.add(cid)
            colors = np.zeros((len(point_labels), 3))
            for i, label in enumerate(point_labels):
                if label < 0 or label in gray_label_ids:
                    colors[i] = [0.7, 0.7, 0.7] # 背景/unlabeled 统一灰色
                else:
                    colors[i] = self.colors[label % 100]
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 如果提供了类别映射，打印颜色图例
            if class_to_id is not None:
                print("\n--- 3D 分割颜色图例 ---")
                for cls_name, cls_id in class_to_id.items():
                    if cls_name in ("background", "unlabeled"):
                        color_name = f"灰色 (Gray) [{cls_name}]"
                    else:
                        color_name = self.get_color_name(cls_id % 100)
                    print(f"类别: {cls_name:<15} | 颜色: {color_name}")
                print("-----------------------\n")

        # o3d.visualization.draw_geometries([pcd])

        o3d.io.write_point_cloud(save_path, pcd)
        print(f"✅ 可视化结果已保存至: {os.path.abspath(save_path)}")

    def save_parts_separately(self, pcd: o3d.geometry.PointCloud, point_labels: np.ndarray,
                              class_to_id: dict, save_dir: str):
        """
        将每个部件单独保存为一个高亮的点云文件。
        非该部件的点以灰色显示，该部件以对应颜色高亮。

        Args:
            pcd: Open3D 点云对象
            point_labels: (N,) 数组，每个点的语义标签
            class_to_id: 类别名到ID的映射
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 反转映射：id -> class_name
        id_to_class = {v: k for k, v in class_to_id.items()}

        # 获取点坐标和颜色
        points = np.asarray(pcd.points)
        n_points = len(points)

        for class_name, class_id in class_to_id.items():
            # 创建输出点云
            part_pcd = o3d.geometry.PointCloud()

            # 该部件使用高亮颜色，其他点用灰色
            part_colors = np.zeros((n_points, 3))
            part_colors[:] = [0.7, 0.7, 0.7]  # 灰色背景

            # unlabeled 与 background 共享灰色，不需要单独输出
            if class_name == "unlabeled":
                continue
            # 标记属于该部件的点
            part_mask = (point_labels == class_id)
            if class_name == "background":
                # background 的 label 通常是 -1 或最大 id；同时包含 unlabeled 的点
                unlabeled_id = class_to_id.get("unlabeled", -999)
                part_mask = (point_labels < 0) | (point_labels == class_id) | (point_labels == unlabeled_id)

            if np.any(part_mask):
                # 该部件使用高亮颜色
                highlight_color = self.colors[class_id % 100]
                part_colors[part_mask] = highlight_color

                # 设置点云
                part_pcd.points = o3d.utility.Vector3dVector(points)
                part_pcd.colors = o3d.utility.Vector3dVector(part_colors)

                # 保存文件，文件名使用类别名（替换特殊字符）
                safe_name = class_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                save_path = os.path.join(save_dir, f"part_{safe_name}.ply")
                o3d.io.write_point_cloud(save_path, part_pcd)
                point_count = np.sum(part_mask)
                print(f"  ✅ 已保存部件 '{class_name}' 的点云: {point_count} 点 -> {save_path}")
            else:
                print(f"  ⚠️ 部件 '{class_name}' 在点云中无对应点，跳过。")

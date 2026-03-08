import open3d as o3d
import numpy as np

class Visualizer3D:
    def __init__(self):
        # 预定义颜色表
        np.random.seed(42)
        self.colors = np.random.rand(100, 3)

    def visualize_3d_result(self, pcd: o3d.geometry.PointCloud, point_labels: np.ndarray = None):
        """
        渲染最终分割好的彩色 3D 点云
        """
        if point_labels is not None:
            colors = np.zeros((len(point_labels), 3))
            for i, label in enumerate(point_labels):
                if label >= 0:
                    colors[i] = self.colors[label % 100]
                else:
                    colors[i] = [0.7, 0.7, 0.7] # 背景灰色
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
        o3d.visualization.draw_geometries([pcd])

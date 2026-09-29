import numpy as np
import torch
from typing import Tuple, List
import open3d as o3d

class MultiViewRenderer:
    def __init__(self, resolution: Tuple[int, int] = (512, 512), point_radius: float = 0.015,
                 albedo_min_value: float = 0.55):
        self.H, self.W = resolution
        self.point_radius = point_radius
        # 渲染前对 albedo 做一次 HSV V 通道下限抬升，避免源网格中的纯黑材质
        # （如灯泡玻璃 / 灯罩内壁）经 Phong 公式 `shaded = colors * shade` 后塌成黑洞。
        # 设为 0 关闭该机制；默认 0.55 仅对极暗点生效，常规物体几乎无影响。
        self.albedo_min_value = float(albedo_min_value)
        # Try to import pytorch3d
        try:
            import pytorch3d
            from pytorch3d.structures import Pointclouds
            from pytorch3d.renderer import (
                look_at_view_transform,
                FoVPerspectiveCameras,
                PointsRasterizationSettings,
                PointsRenderer,
                PointsRasterizer,
                AlphaCompositor,
                NormWeightedCompositor
            )
            self.has_pytorch3d = True
            # 移除硬编码的 ":0"，让 PyTorch 根据环境变量自动映射到第一张可见的卡
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.has_pytorch3d = False
            print("Warning: PyTorch3D not found. Falling back to Open3D rendering (slower, no exact z-buffer index mapping).")

    def render(self, point_cloud_xyz: np.ndarray, point_cloud_colors: np.ndarray,
               camera_positions: np.ndarray, camera_rotations: np.ndarray):
        """
        执行 3D 多视角渲染
        返回: RenderOutput 对象
        """
        if self.has_pytorch3d:
            return self._render_pytorch3d(point_cloud_xyz, point_cloud_colors, camera_positions, camera_rotations)
        else:
            return self._render_open3d(point_cloud_xyz, point_cloud_colors, camera_positions, camera_rotations)

    @staticmethod
    def _lift_dark_albedo(rgb: np.ndarray, min_v: float) -> np.ndarray:
        """
        将过暗的逐点 albedo 在 HSV V 通道上抬升到 min_v，色相 / 饱和度保留。

        - 当一个点的 RGB 至少有一个通道 >= min_v 时，原样保留；
        - 当 max(R,G,B) ∈ (0, min_v) 时，按比例缩放到 max(R,G,B) = min_v（等价 HSV V 抬升）；
        - 当 RGB 完全 = 0（无任何色彩信息），赋值为 (min_v, min_v, min_v) 中性灰。

        这样做可以避免源网格中的纯黑材质（如灯泡玻璃 / 灯罩内壁）经
        Phong 公式 `shaded = colors * shade` 后塌成纯黑的“黑洞”，
        同时对正常颜色的点完全不产生影响。

        :param rgb: (N, 3) 浮点 RGB，区间 [0, 1]
        :param min_v: HSV V 通道下限（0 关闭该机制；典型值 0.4 ~ 0.6）
        """
        if min_v <= 0.0:
            return rgb
        rgb_out = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
        cmax = rgb_out.max(axis=1)
        too_dark = cmax < min_v
        if not np.any(too_dark):
            return rgb_out

        rgb_out = rgb_out.copy()
        # 仍带有色相的过暗点：按比例放大，使 max 通道恰好达到 min_v
        has_color = too_dark & (cmax > 1e-6)
        if np.any(has_color):
            scale = min_v / cmax[has_color]
            rgb_out[has_color] = rgb_out[has_color] * scale[:, None]
        # 完全纯黑（无色相信息）：替换为中性灰
        pure_black = too_dark & (cmax <= 1e-6)
        if np.any(pure_black):
            rgb_out[pure_black] = min_v
        return np.clip(rgb_out, 0.0, 1.0)

    @staticmethod
    def _apply_phong_shading(
        xyz: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
        cam_position: np.ndarray,
        ambient: float = 0.45,
        diffuse: float = 0.45,
        specular: float = 0.10,
        shininess: float = 16.0,
    ) -> np.ndarray:
        """
        对点云颜色施加逐视角 Phong 着色，使纯色模型也能展现立体层次。
        使用双面光照（abs(n·L)）防止法线朝向不一致导致的突兀黑块。
        """
        light_dir = cam_position - xyz
        norms = np.linalg.norm(light_dir, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        light_dir = light_dir / norms

        n_dot_l = np.abs(np.sum(normals * light_dir, axis=1))

        view_dir = light_dir
        n_dot_l_signed = np.sum(normals * light_dir, axis=1)
        reflect_dir = 2.0 * n_dot_l_signed[:, None] * normals - light_dir
        r_dot_v = np.abs(np.sum(reflect_dir * view_dir, axis=1))
        r_dot_v = np.clip(r_dot_v, 0.0, 1.0)
        spec = np.power(r_dot_v, shininess)

        shade = ambient + diffuse * n_dot_l + specular * spec
        shade = np.clip(shade, 0.0, 1.0)

        shaded = colors * shade[:, None]
        return np.clip(shaded, 0.0, 1.0)

    def _render_pytorch3d(self, xyz: np.ndarray, colors: np.ndarray, cam_pos: np.ndarray, cam_rot: np.ndarray):
        from pytorch3d.structures import Pointclouds
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            PointsRasterizationSettings,
            PointsRenderer,
            PointsRasterizer,
            AlphaCompositor,
            NormWeightedCompositor
        )

        from datatypes import RenderOutput

        # 一次性把过暗 albedo 抬到 V>=albedo_min_value，避免源网格纯黑材质
        # （灯泡玻璃 / 灯罩内壁等）渲成黑洞。
        colors = self._lift_dark_albedo(colors, self.albedo_min_value)

        normals = self._estimate_normals(xyz)

        K = cam_pos.shape[0]
        images = []
        index_maps = []
        depth_maps = []

        verts_t = torch.Tensor(xyz).to(self.device)

        raster_settings = PointsRasterizationSettings(
            image_size=(self.H, self.W),
            radius=self.point_radius,
            points_per_pixel=1,
            bin_size=0
        )

        for i in range(K):
            cam_world_pos = cam_pos[i]
            shaded_colors = self._apply_phong_shading(
                xyz, colors, normals, cam_world_pos)

            rgb_t = torch.Tensor(shaded_colors).to(self.device)
            point_cloud = Pointclouds(points=[verts_t], features=[rgb_t])

            R = torch.Tensor(cam_rot[i:i+1]).to(self.device)
            T = torch.Tensor(cam_pos[i:i+1]).to(self.device)
            R = R.transpose(1, 2)
            T = -torch.bmm(R.transpose(1, 2), T.unsqueeze(2)).squeeze(2)

            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=60.0)

            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

            fragments = rasterizer(point_cloud)

            idx_map = fragments.idx[0].cpu().numpy()
            depth_map = fragments.zbuf[0].cpu().numpy()

            renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=NormWeightedCompositor(background_color=(1, 1, 1))
            )
            img_tensor = renderer(point_cloud)

            rgb_img = img_tensor[0, ..., :3].cpu().numpy()

            gamma = 0.85
            rgb_img = np.power(np.clip(rgb_img, 0, 1), gamma)

            img = (rgb_img * 255).astype(np.uint8)

            images.append(img)
            index_maps.append(idx_map)
            depth_maps.append(depth_map)

            del renderer, rasterizer, cameras, fragments, img_tensor, R, T, rgb_t, point_cloud
            torch.cuda.empty_cache()

        return RenderOutput(images=images, depth_maps=depth_maps, index_maps=index_maps)

    @staticmethod
    def _estimate_normals(xyz: np.ndarray) -> np.ndarray:
        """估算点云法线（用于 Phong 着色）"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 0.0]))
        return np.asarray(pcd.normals)

    def _render_open3d(self, xyz: np.ndarray, colors: np.ndarray, cam_pos: np.ndarray, cam_rot: np.ndarray):
        # Fallback implementation using Open3D.
        # 与 PyTorch3D 路径保持一致：先抬升过暗 albedo，避免纯黑顶点导致渲出黑洞。
        colors = self._lift_dark_albedo(colors, self.albedo_min_value)
        # 关键设计：
        # 1) 不再把 PyTorch3D 风格的 (R_c2w, t_c2w) 直接灌进 Open3D 的 extrinsic，
        #    而是用 Open3D 的 lookat / front / up 高层 API，直接锚定到原点（点云已归一化到单位球）。
        # 2) 把窗口设为可见、显式刷新事件、强制 do_render=True，避免 Windows 上的全白截图。
        # 3) 显式设置白底 + 合理的 point_size，避免肉眼看不到点。
        from datatypes import RenderOutput

        K = cam_pos.shape[0]
        images = []
        index_maps = []
        depth_maps = []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.W, height=self.H, visible=True)
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        # point_radius 是世界坐标尺寸；Open3D 这里要的是屏幕像素大小，做一个稳健的近似
        opt.point_size = float(np.clip(self.point_radius * self.W, 2.0, 8.0))

        # Open3D 的 set_zoom 只吃一个标量，它控制"视野中物体的大小"，不直接等于距离。
        # 我们以 "distance=2.0 对应 zoom=0.7" 为基准，让 YAML 里的 camera_distance
        # 能像 PyTorch3D 路径那样：距离变大 → 物体在画面中变小。
        ZOOM_AT_REF_DIST = 0.7
        REF_DIST = 2.0

        world_up_default = np.array([0.0, 1.0, 0.0])

        for i in range(K):
            ctr = vis.get_view_control()

            cam_position = np.asarray(cam_pos[i], dtype=np.float64)
            dist = float(np.linalg.norm(cam_position))

            # Open3D 的语义：front 是 "从 lookat 目标指向相机" 的单位向量
            front = cam_position / max(dist, 1e-8)
            world_up = world_up_default.copy()
            if abs(float(np.dot(front, world_up))) > 0.999:
                world_up = np.array([1.0, 0.0, 0.0])

            # zoom ∝ 1/distance，保持小范围合理区间
            zoom = ZOOM_AT_REF_DIST * REF_DIST / max(dist, 1e-6)
            zoom = float(np.clip(zoom, 0.05, 5.0))

            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_front(front.tolist())
            ctr.set_up(world_up.tolist())
            ctr.set_zoom(zoom)

            # 多刷几轮，保证视角切换真正落到 framebuffer
            for _ in range(3):
                vis.poll_events()
                vis.update_renderer()

            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            images.append(img)

            # ---- 索引图 / 深度图（近似，仅供需要的下游代码使用） ----
            # 用 Open3D 当前 view 的 extrinsic + intrinsic 自己重投影一遍点云
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            extrinsic = np.asarray(cam_params.extrinsic, dtype=np.float64)
            K_intr = np.asarray(cam_params.intrinsic.intrinsic_matrix, dtype=np.float64)
            fx, fy = K_intr[0, 0], K_intr[1, 1]
            cx, cy = K_intr[0, 2], K_intr[1, 2]

            idx_map = np.full((self.H, self.W, 1), -1, dtype=np.int32)
            depth_map = np.full((self.H, self.W, 1), np.inf, dtype=np.float32)

            pts_homo = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
            pts_cam = (extrinsic @ pts_homo.T).T[:, :3]
            valid = pts_cam[:, 2] > 0
            if np.any(valid):
                u = (pts_cam[valid, 0] * fx / pts_cam[valid, 2] + cx).astype(int)
                v = (pts_cam[valid, 1] * fy / pts_cam[valid, 2] + cy).astype(int)
                valid_indices = np.where(valid)[0]
                depths = pts_cam[valid, 2]
                sort_idx = np.argsort(depths)[::-1]  # 远 -> 近
                for j in sort_idx:
                    x, y = int(u[j]), int(v[j])
                    if 0 <= x < self.W and 0 <= y < self.H:
                        r = 2
                        y_min, y_max = max(0, y - r), min(self.H, y + r + 1)
                        x_min, x_max = max(0, x - r), min(self.W, x + r + 1)
                        idx_map[y_min:y_max, x_min:x_max, 0] = valid_indices[j]
                        depth_map[y_min:y_max, x_min:x_max, 0] = depths[j]

            index_maps.append(idx_map)
            depth_maps.append(depth_map)

        vis.destroy_window()
        return RenderOutput(images=images, depth_maps=depth_maps, index_maps=index_maps)

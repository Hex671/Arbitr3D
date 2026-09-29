import numpy as np
import open3d as o3d
import os

_MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl", ".off", ".fbx"}


class PointCloudDataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_point_cloud(
        self, file_path: str, num_points: int = 0
    ) -> o3d.geometry.PointCloud:
        """
        加载点云文件。自动检测格式：
        - .ply / .pcd / .xyz → 直接读取点云
        - .glb / .gltf / .obj / .stl 等网格格式 → 从三角面均匀采样

        Args:
            file_path: 相对于 data_dir 的路径
            num_points: 仅网格格式有效；0 表示使用网格原始顶点
        """
        full_path = os.path.join(self.data_dir, file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Point cloud file not found: {full_path}")

        ext = os.path.splitext(full_path)[1].lower()
        if ext in _MESH_EXTENSIONS:
            return self._load_mesh_as_point_cloud(full_path, num_points)

        pcd = o3d.io.read_point_cloud(full_path)
        return pcd

    @staticmethod
    def _load_mesh_as_point_cloud(
        mesh_path: str, num_points: int = 0
    ) -> o3d.geometry.PointCloud:
        """
        从三角网格文件加载并转换为点云。
        对于 .glb/.gltf 等纹理网格，用 trimesh 采样并通过面颜色获取真实色彩。

        Args:
            mesh_path: 网格文件绝对路径
            num_points: 采样点数。0 = 直接使用网格顶点
        Returns:
            Open3D PointCloud
        """
        ext = os.path.splitext(mesh_path)[1].lower()
        mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
        if mesh.is_empty():
            raise ValueError(f"Failed to load mesh or mesh is empty: {mesh_path}")
        mesh.compute_vertex_normals()

        if not mesh.has_vertex_colors() and ext in {'.glb', '.gltf', '.obj'}:
            pcd = PointCloudDataset._sample_with_trimesh_colors(
                mesh_path, num_points)
            if pcd is not None:
                return pcd

        if num_points > 0 and len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            if mesh.has_vertex_colors():
                pcd.colors = mesh.vertex_colors
            if mesh.has_vertex_normals():
                pcd.normals = mesh.vertex_normals
        return pcd

    @staticmethod
    def _get_face_colors(geom) -> 'np.ndarray | None':
        """
        从 trimesh.Trimesh 子网格提取逐面 RGB 颜色 (N_faces, 3) float64 [0,1]。
        按优先级尝试多种途径，确保纹理/材质/顶点色都能覆盖。
        """
        import trimesh as _tm
        n_faces = len(geom.faces)
        if n_faces == 0:
            return None

        # 确保 visual 持有 mesh 引用（to_color 内部需要）
        if hasattr(geom.visual, 'mesh'):
            geom.visual.mesh = geom

        # 策略 1：TextureVisuals → 直接从纹理采样逐面顶点色再平均
        if isinstance(geom.visual, _tm.visual.TextureVisuals):
            try:
                vc = geom.visual.to_color().vertex_colors
                vc = np.asarray(vc)[:, :3].astype(np.float64) / 255.0
                tri = geom.faces
                fc = vc[tri].mean(axis=1)
                return fc
            except Exception:
                pass
            # 策略 1b：材质有 main_color / baseColorFactor
            try:
                mat = geom.visual.material
                base = None
                if hasattr(mat, 'baseColorFactor'):
                    base = np.asarray(mat.baseColorFactor, dtype=np.float64)
                elif hasattr(mat, 'main_color'):
                    base = np.asarray(mat.main_color, dtype=np.float64)
                if base is not None:
                    if base.max() > 1.0:
                        base = base / 255.0
                    rgb = base[:3]
                    return np.tile(rgb, (n_faces, 1))
            except Exception:
                pass

        # 策略 2：ColorVisuals（顶点色 / 面色已经内嵌）
        if isinstance(geom.visual, _tm.visual.ColorVisuals):
            try:
                cv = geom.visual
                cv.mesh = geom
                fc = np.asarray(cv.face_colors)[:, :3].astype(np.float64) / 255.0
                return fc
            except Exception:
                pass

        # 策略 3：通用 to_color() 回退
        try:
            cv = geom.visual.to_color()
            cv.mesh = geom
            fc = np.asarray(cv.face_colors)[:, :3].astype(np.float64) / 255.0
            return fc
        except Exception:
            pass

        return None

    @staticmethod
    def _sample_with_trimesh_colors(
        mesh_path: str, num_points: int
    ) -> 'o3d.geometry.PointCloud | None':
        """
        用 trimesh 加载纹理网格，采样点的同时通过面索引获取面颜色。
        对于多子网格 Scene，在合并前逐子网格提取面颜色，避免合并后纹理丢失。
        """
        try:
            import trimesh

            tm_raw = trimesh.load(mesh_path)

            all_pts, all_colors = [], []
            n_target = num_points if num_points > 0 else 10000

            if isinstance(tm_raw, trimesh.Scene):
                geom_list = [
                    g for g in tm_raw.geometry.values()
                    if isinstance(g, trimesh.Trimesh)
                       and g.faces is not None and len(g.faces) > 0
                ]
                if not geom_list:
                    print("    [Color Warning] Scene 中无有效 Trimesh 子网格")
                    return None

                total_area = sum(g.area for g in geom_list)
                if total_area <= 0:
                    return None

                for geom in geom_list:
                    n_sub = max(1, int(n_target * geom.area / total_area))
                    fc = PointCloudDataset._get_face_colors(geom)
                    pts, fidx = geom.sample(n_sub, return_index=True)
                    pts = np.asarray(pts, dtype=np.float64)
                    all_pts.append(pts)
                    if fc is not None:
                        all_colors.append(fc[fidx])
                    else:
                        all_colors.append(
                            np.full((len(pts), 3), 0.5, dtype=np.float64))
                        print(f"    [Color] 子网格({len(geom.faces)} faces)颜色提取失败，用灰色")

            elif isinstance(tm_raw, trimesh.Trimesh):
                tm = tm_raw
                if tm.faces is None or len(tm.faces) == 0:
                    return None
                n_target = num_points if num_points > 0 else len(tm.vertices)
                fc = PointCloudDataset._get_face_colors(tm)
                pts, fidx = tm.sample(n_target, return_index=True)
                pts = np.asarray(pts, dtype=np.float64)
                all_pts.append(pts)
                if fc is not None:
                    all_colors.append(fc[fidx])
                else:
                    all_colors.append(
                        np.full((len(pts), 3), 0.5, dtype=np.float64))
            else:
                print(f"    [Color Warning] trimesh 返回了非预期类型: {type(tm_raw)}")
                return None

            if not all_pts:
                return None

            sample_pts = np.vstack(all_pts)
            sample_colors = np.vstack(all_colors)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sample_pts)
            pcd.colors = o3d.utility.Vector3dVector(sample_colors)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))

            mean_c = sample_colors.mean(axis=0)
            print(f"    [Color] trimesh 采样 {len(sample_pts)} 点, "
                  f"平均颜色 RGB=({mean_c[0]:.2f}, {mean_c[1]:.2f}, {mean_c[2]:.2f})")
            return pcd

        except Exception as e:
            print(f"    [Color Warning] trimesh 颜色采样失败，回退到 Open3D: {e}")
            return None

    @staticmethod
    def load_mesh_with_labels(
        mesh_path: str,
        gt_label_path: str,
        num_points: int = 0,
    ):
        """
        加载网格 + GT 标签，保证点和标签严格对齐。
        对 per-face GT（如 PartObjaverse-Tiny），使用 trimesh 加载以确保面顺序与 GT 一致。

        返回:
            (pcd, gt_labels): pcd 为 Open3D PointCloud，gt_labels 为 (N,) int 数组
        """
        import trimesh as _trimesh
        from collections import Counter as _Counter

        mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
        if mesh.is_empty():
            raise ValueError(f"Failed to load mesh: {mesh_path}")
        mesh.compute_vertex_normals()

        raw = np.load(gt_label_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.ndim == 0:
            raw = raw.item()
        if isinstance(raw, dict):
            gt_labels = raw.get("semantic_seg", raw.get("labels", None))
            if gt_labels is None:
                raise ValueError(f"Cannot find label array in dict keys: {list(raw.keys())}")
            gt_labels = np.asarray(gt_labels, dtype=np.int64)
        else:
            gt_labels = np.asarray(raw, dtype=np.int64).ravel()

        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.triangles)
        n_gt = len(gt_labels)

        is_per_vertex = (n_gt == num_vertices)
        is_per_face = (n_gt == num_faces and num_faces > 0)

        if not is_per_vertex and is_per_face:
            # per-face GT: 用 trimesh 采样+面索引直接赋标签（官方加载方式）
            tm = _trimesh.load(mesh_path)
            if isinstance(tm, _trimesh.Scene):
                tm = tm.dump(concatenate=True)

            if num_points <= 0:
                num_points = len(tm.vertices)

            n_sample = max(num_points * 3, 500000)
            sample_pts, face_idx = tm.sample(n_sample, return_index=True)
            sample_pts = np.asarray(sample_pts, dtype=np.float64)
            sample_labels = gt_labels[face_idx]

            ref_pcd = o3d.geometry.PointCloud()
            ref_pcd.points = o3d.utility.Vector3dVector(sample_pts)

            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            kdtree = o3d.geometry.KDTreeFlann(ref_pcd)
            sampled_pts = np.asarray(pcd.points)
            sampled_labels = np.zeros(len(sampled_pts), dtype=np.int64)
            for i, pt in enumerate(sampled_pts):
                _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
                sampled_labels[i] = sample_labels[idx[0]]
            return pcd, sampled_labels

        # per-vertex GT 路径（PartNetE 等）
        if num_points <= 0 or num_points >= num_vertices:
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            if mesh.has_vertex_colors():
                pcd.colors = mesh.vertex_colors
            if mesh.has_vertex_normals():
                pcd.normals = mesh.vertex_normals

            if n_gt == num_vertices:
                return pcd, gt_labels
            else:
                print(f"  [Warning] GT labels ({n_gt}) != mesh vertices ({num_vertices}), "
                      f"using min({n_gt}, {num_vertices}) points")
                n = min(n_gt, num_vertices)
                pts = np.asarray(mesh.vertices)[:n]
                pcd_trimmed = o3d.geometry.PointCloud()
                pcd_trimmed.points = o3d.utility.Vector3dVector(pts)
                if mesh.has_vertex_colors():
                    pcd_trimmed.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[:n])
                return pcd_trimmed, gt_labels[:n]

        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        if is_per_vertex:
            vertex_pts = np.asarray(mesh.vertices)
            sampled_pts = np.asarray(pcd.points)
            vertex_pcd = o3d.geometry.PointCloud()
            vertex_pcd.points = o3d.utility.Vector3dVector(vertex_pts)
            kdtree = o3d.geometry.KDTreeFlann(vertex_pcd)
            sampled_labels = np.zeros(len(sampled_pts), dtype=np.int64)
            for i, pt in enumerate(sampled_pts):
                _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
                sampled_labels[i] = gt_labels[idx[0]]
            return pcd, sampled_labels
        else:
            return pcd, gt_labels[:len(np.asarray(pcd.points))]

    def normalize_pc(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        点云归一化：将对象平移至坐标原点，并缩放至单位球内。
        这确保了后续渲染时固定机位的一致性。
        """
        points = np.asarray(pcd.points)

        # 平移至原点
        centroid = np.mean(points, axis=0)
        points -= centroid

        # 缩放至单位球
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points /= max_dist

        normalized_pcd = o3d.geometry.PointCloud()
        normalized_pcd.points = o3d.utility.Vector3dVector(points)

        # 继承原点云的其他属性
        if pcd.has_colors():
            normalized_pcd.colors = pcd.colors
        if pcd.has_normals():
            normalized_pcd.normals = pcd.normals

        return normalized_pcd

    def extract_color(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        提取点云的 RGB 颜色特征。
        有颜色则直接使用；无颜色时生成基于法线方向的伪彩色，
        使不同朝向的表面呈现不同色调，配合 Phong 着色产生层次感。
        """
        if pcd.has_colors():
            return np.asarray(pcd.colors)
        else:
            return self._generate_normal_colors(pcd)

    @staticmethod
    def _generate_normal_colors(pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        将法线方向映射为柔和的伪彩色。
        不同朝向的面获得不同色调，使纯几何点云也有丰富的视觉边界。
        """
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))

        normals = np.asarray(pcd.normals)
        abs_n = np.abs(normals)

        base_gray = 0.62
        color_strength = 0.18
        colors = np.full((len(normals), 3), base_gray, dtype=np.float64)
        colors[:, 0] += color_strength * abs_n[:, 0]
        colors[:, 1] += color_strength * abs_n[:, 1]
        colors[:, 2] += color_strength * abs_n[:, 2]

        colors = np.clip(colors, 0.0, 1.0)
        return colors

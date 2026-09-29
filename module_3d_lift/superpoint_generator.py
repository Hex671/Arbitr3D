import numpy as np
import open3d as o3d
import sys
import os

# 将 partition 目录加入环境变量
partition_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'partition'))
if partition_dir not in sys.path:
    sys.path.append(partition_dir)

# 将 libcp.so 实际所在的编译目录也加进去！
libcp_dir = os.path.join(partition_dir, 'cut-pursuit', 'build', 'src')
if libcp_dir not in sys.path:
    sys.path.append(libcp_dir)

try:
    import graphs
    # 注意：这里需要确保 cut-pursuit 的 C++ 库已经正确编译
    # 并且 libcp.so (Linux) 或 libcp.pyd (Windows) 所在的目录在 PYTHONPATH 中
    import libcp
    HAS_CUT_PURSUIT = True
except ImportError as e:
    print(f"    [Superpoint Warning] Could not import cut-pursuit library ({e}). Falling back to pure Python implementation.")
    HAS_CUT_PURSUIT = False

def _orient_normals_outward(pcd):
    """
    基于质心的法向量外朝向定向。
    将每个点的法向量翻转为指向远离物体质心的方向，
    使薄板正反两面的法向量天然相反。

    相比 orient_normals_consistent_tangent_plane，此方法不通过近邻图传播，
    不会因薄结构两侧互为近邻而把对面法向量翻转到同一方向。
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    centroid = np.mean(points, axis=0)
    outward_dirs = points - centroid

    dots = np.sum(normals * outward_dirs, axis=1)
    flip_mask = dots < 0
    normals[flip_mask] = -normals[flip_mask]

    pcd.normals = o3d.utility.Vector3dVector(normals)
    print(f"    [Superpoint] 法向量外朝向定向完成，翻转了 {np.sum(flip_mask)}/{len(flip_mask)} 个点")


def generate_superpoints_cp(pcd: o3d.geometry.PointCloud, k_nn_adj: int = 10,
                            reg_strength: float = 0.1,
                            spatial_weight: float = 5.0) -> np.ndarray:
    """
    使用 Cut-Pursuit 算法生成超点。
    参照 SPT (ICCV'23) 的做法，将空间坐标加入特征向量并乘以归一化因子 μ，
    使 Cut-Pursuit 自然地限制超点的空间跨度，无需后处理拆分。

    :param pcd: Open3D 点云
    :param k_nn_adj: 构建邻接图时的 KNN 邻居数
    :param reg_strength: 正则化强度 (越小超点越细)
    :param spatial_weight: 空间坐标权重 μ (越大超点空间跨度越小)
    """
    points = np.asarray(pcd.points).astype('float32')
    num_points = points.shape[0]

    # 1. 估算法线（质心外朝向定向，确保薄板正反面法向量相反）
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    _orient_normals_outward(pcd)
    normals = np.asarray(pcd.normals).astype('float32')

    # 提取颜色
    if pcd.has_colors():
        # open3d 的 colors 应是 float64 ∈ [0,1]；少数损坏的 ply / 错误归一化会带 NaN/Inf 或越界值，
        # 直接 cast float32 会 overflow 进而让下游 cut-pursuit 的 BK max-flow 触发 C++ assert。
        colors_raw = np.asarray(pcd.colors, dtype=np.float64)
        colors_raw = np.nan_to_num(colors_raw, nan=0.5, posinf=1.0, neginf=0.0)
        colors = np.clip(colors_raw, 0.0, 1.0).astype('float32')
    else:
        colors = np.zeros_like(points, dtype='float32')

    # 顺手对法线也做一次健全性清洗：NaN 法线会让 cut-pursuit 的特征距离炸掉
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')

    # 2. 特征空间：法线 + 颜色 + 归一化空间坐标（SPT μ 策略）
    # 空间坐标归一化到 [0,1] 后乘以 spatial_weight，限制超点物理尺寸
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_extent = np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound())
    bbox_extent = np.maximum(bbox_extent, 1e-6)
    points_norm = (points - np.asarray(bbox.get_min_bound())) / bbox_extent

    features = np.hstack([
        normals * 2.0,
        colors * 0.5,
        points_norm * spatial_weight,
    ]).astype('float32')
    # 最后兜底：哪怕上面任何一步算出 NaN/Inf，这里都强制清成有限值，杜绝 C++ assert
    if not np.all(np.isfinite(features)):
        bad = (~np.isfinite(features)).sum()
        print(f"  [Superpoint][warn] features 含 {bad} 个 NaN/Inf，已置零")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')

    # 3. 使用 graphs.py 构建 KNN 空间连通图
    graph = graphs.compute_graph_nn(points, k_nn_adj)

    source = graph["source"]
    target = graph["target"]
    distances = graph["distances"]

    edge_weights = np.exp(-distances / (np.mean(distances) + 1e-5)).astype('float32')
    # cut-pursuit 内部 BK max-flow 要求边权严格非负且有限，否则会触发 C++ assert
    edge_weights = np.nan_to_num(edge_weights, nan=1e-6, posinf=1.0, neginf=1e-6)
    edge_weights = np.clip(edge_weights, 1e-6, 1.0).astype('float32')

    # 4. 运行 L0-cut pursuit 分割
    components, in_component = libcp.cutpursuit(
        features,
        source,
        target,
        edge_weights,
        reg_strength,
        0,
        1,
        1.0
    )

    return np.asarray(in_component, dtype=np.int32)

def generate_superpoints_fallback(pcd: o3d.geometry.PointCloud, num_points_per_superpoint: int = 50) -> np.ndarray:
    """
    备用方案：基于 NetworkX 的法线连通图切分 (如果 C++ 库没编译成功或缺失)
    """
    import networkx as nx

    points = np.asarray(pcd.points)
    num_points = points.shape[0]

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    _orient_normals_outward(pcd)
    normals = np.asarray(pcd.normals)

    cos_threshold = np.cos(np.radians(15.0))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    G = nx.Graph()
    G.add_nodes_from(range(num_points))

    k_neighbors = 15
    for i in range(num_points):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], 0.05)
        if k < 5:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k_neighbors)

        n_i = normals[i]
        for j in idx[1:]:
            n_j = normals[j]
            cos_angle = np.dot(n_i, n_j)
            if cos_angle >= cos_threshold:
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    superpoint_labels = np.full(num_points, -1, dtype=np.int32)
    current_label = 0

    for comp in components:
        comp_list = list(comp)
        if len(comp_list) > num_points_per_superpoint * 3:
            sub_points = points[comp_list]
            n_sub_clusters = max(1, len(comp_list) // num_points_per_superpoint)
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=n_sub_clusters, random_state=42, n_init=1)
            sub_labels = kmeans.fit_predict(sub_points)
            for sub_id in range(n_sub_clusters):
                idx_in_sub = np.where(sub_labels == sub_id)[0]
                global_idx = [comp_list[k] for k in idx_in_sub]
                superpoint_labels[global_idx] = current_label
                current_label += 1
        else:
            superpoint_labels[comp_list] = current_label
            current_label += 1

    unlabeled_indices = np.where(superpoint_labels == -1)[0]
    if len(unlabeled_indices) > 0:
        labeled_indices = np.where(superpoint_labels != -1)[0]
        if len(labeled_indices) > 0:
            import scipy.spatial
            tree = scipy.spatial.cKDTree(points[labeled_indices])
            _, nearest_idx = tree.query(points[unlabeled_indices])
            superpoint_labels[unlabeled_indices] = superpoint_labels[labeled_indices[nearest_idx]]
        else:
            superpoint_labels[:] = 0

    return superpoint_labels



def generate_superpoints(pcd: o3d.geometry.PointCloud,
                         num_points_per_superpoint: int = 50,
                         reg_strength: float = 0.03,
                         k_nn_adj: int = 10,
                         spatial_weight: float = 5.0,
                         max_pts_for_direct: int = 80000) -> np.ndarray:
    """
    统一接口：优先使用 Cut-Pursuit，失败则回退到 Python 连通图方案。
    参照 SPT (ICCV'23)，通过空间坐标特征 + 低正则化强度在生成阶段
    就控制超点大小，无需后处理拆分。

    当点数超过 max_pts_for_direct 时，先体素降采样再生成超点，
    最后通过最近邻映射回全分辨率。

    Args:
        num_points_per_superpoint: 回退方案的目标超点大小
        reg_strength: Cut-Pursuit 正则化强度（越小超点越细）
        k_nn_adj: Cut-Pursuit KNN 邻居数
        spatial_weight: 空间坐标权重 μ（越大超点空间跨度越小）
        max_pts_for_direct: 点数低于此值时直接计算，否则先降采样
    """
    import time
    points = np.asarray(pcd.points)
    num_points = len(points)
    t0 = time.time()

    if num_points < k_nn_adj + 2:
        print(f"    [Superpoint] 原始点数过少 ({num_points})，每点独立超点")
        return np.arange(num_points, dtype=np.int32)

    use_downsample = (num_points > max_pts_for_direct)

    if use_downsample:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        target_pts = max_pts_for_direct // 2

        voxel_size = bbox_diag / 100.0
        ds_pcd = pcd.voxel_down_sample(voxel_size)
        ds_num = len(ds_pcd.points)
        for _ in range(15):
            if ds_num == 0:
                voxel_size *= 0.5
            elif target_pts * 0.7 <= ds_num <= target_pts * 1.5:
                break
            else:
                voxel_size *= np.sqrt(max(ds_num, 1) / target_pts)
            ds_pcd = pcd.voxel_down_sample(voxel_size)
            ds_num = len(ds_pcd.points)

        min_ds_pts = max(k_nn_adj * 5, 100)
        if ds_num < min_ds_pts:
            print(f"    [Superpoint] 降采样点数过少 ({ds_num} < {min_ds_pts})，跳过降采样")
            use_downsample = False
            work_pcd = pcd
        else:
            print(f"    [Superpoint] 降采样: {num_points} -> {ds_num} 点 "
                  f"(voxel={voxel_size:.6f})")
            work_pcd = ds_pcd
    else:
        work_pcd = pcd

    work_num = len(work_pcd.points)
    if work_num < k_nn_adj + 2:
        print(f"    [Superpoint] 点数过少 ({work_num})，每点独立超点")
        sp_labels_work = np.arange(work_num, dtype=np.int32)
    elif HAS_CUT_PURSUIT:
        sp_labels_work = generate_superpoints_cp(
            work_pcd, k_nn_adj=k_nn_adj, reg_strength=reg_strength,
            spatial_weight=spatial_weight)
    else:
        sp_labels_work = generate_superpoints_fallback(
            work_pcd, num_points_per_superpoint=num_points_per_superpoint)

    if use_downsample:
        from scipy.spatial import cKDTree
        ds_points = np.asarray(work_pcd.points)
        tree = cKDTree(ds_points)
        _, nn_idx = tree.query(points, k=1)
        sp_labels = sp_labels_work[nn_idx]
    else:
        sp_labels = sp_labels_work

    num_sp_final = int(sp_labels.max()) + 1
    sp_sizes = np.bincount(sp_labels)
    elapsed = time.time() - t0
    print(f"    [Superpoint] {num_sp_final} 个超点, "
          f"大小 min={sp_sizes.min()} median={int(np.median(sp_sizes))} "
          f"max={sp_sizes.max()} ({elapsed:.1f}s)")

    return sp_labels

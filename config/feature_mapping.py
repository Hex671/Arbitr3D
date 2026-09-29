# 几何特征数值到文本描述的映射表，用于对齐人类直观印象，避免 MLLM 对纯数字不敏感

def map_principal_curvature(value: float) -> str:
    """主曲率 (表面弯曲度) [0.0 - 0.33]"""
    if value < 0.02:
        return "绝对平坦 (Perfectly flat)"
    elif value < 0.08:
        return "轻微弯曲 (Slightly curved/Almost flat)"
    elif value < 0.18:
        return "明显弯曲 (Moderately curved, e.g., cylinder surface/edge)"
    else:
        return "剧烈弯曲/球状 (Highly curved/Spherical, e.g., corner/ball/knob)"

def map_normalized_z(value: float) -> str:
    """全局归一化Z坐标 [0.0 - 1.0]"""
    if value < 0.15:
        return "最底部 (At the very bottom, touching the ground)"
    elif value < 0.40:
        return "下半部分 (Lower half)"
    elif value < 0.60:
        return "正中间 (Middle section)"
    elif value < 0.85:
        return "上半部分 (Upper half)"
    else:
        return "最顶部 (At the very top)"

def map_point_ratio(value: float) -> str:
    """点数占比 [0.0 - 1.0]：该掩码对应的 3D 点数占总点数的比例"""
    if value < 0.004:
        return "极小碎片 (Tiny fragment/noise/button)"
    elif value < 0.015:
        return "小部件或细管结构 (Small part or thin structural piece, e.g., wheel/thin bar)"
    elif value < 0.04:
        return "中等结构或支撑件 (Medium part or supporting structure, e.g., leg/armrest/base)"
    elif value < 0.12:
        return "较大部件 (Large component, e.g., backrest frame/seat cushion)"
    else:
        return "核心大面积表面 (Dominant surface, e.g., main backrest/large tabletop)"

def map_normal_vector_surface(angle_with_z: float) -> str:
    """表面的朝向形态 (对大模型更直观)"""
    if angle_with_z < 25.0:
        return "水平表面 (Horizontal surface, e.g., seat/tabletop)"
    elif angle_with_z < 65.0:
        return "倾斜表面 (Slanted/Inclined surface)"
    else:
        return "竖直表面 (Vertical surface, e.g., backrest/cabinet side)"

def map_geometric_features_to_text(features: dict) -> dict:
    """
    将数值型的几何特征字典，映射为带人类直观文本描述的字典
    """
    mapped = {}

    val = features.get("principal_curvature", 0.0)
    mapped["principal_curvature"] = f"{val:.4f} ({map_principal_curvature(val)})"

    val = features.get("normalized_z", 0.0)
    mapped["normalized_z"] = f"{val:.4f} ({map_normalized_z(val)})"

    val = features.get("point_ratio", 0.0)
    mapped["point_ratio"] = f"{val:.4f} ({map_point_ratio(val)})"

    norm = features.get("normal_vector", {})
    mapped["normal_vector"] = {
        "angle_with_z": f"{norm.get('angle_with_z', 0.0):.2f}°",
        "angle_with_xy": f"{norm.get('angle_with_xy', 0.0):.2f}°",
        "surface_orientation": map_normal_vector_surface(norm.get('angle_with_z', 0.0))
    }

    centroid = features.get("spatial_centroid", {})
    mapped["spatial_centroid"] = {
        "x_width": f"{centroid.get('x_width', 0.0):.3f}",
        "y_height": f"{centroid.get('y_height', 0.0):.3f} (与 normalized_z 一致，0为底 1为顶)",
        "z_depth": f"{centroid.get('z_depth', 0.0):.3f}"
    }

    dim = features.get("relative_dimensions", {})
    y_ratio = dim.get('y_height_ratio', 0.0)
    x_ratio = dim.get('x_width_ratio', 0.0)
    z_ratio = dim.get('z_depth_ratio', 0.0)

    # 精细的三维形状描述
    shape_desc = "普通块状 (Blocky)"

    # 获取最大、次大、最小维度的排序
    dims = {'y': y_ratio, 'x': x_ratio, 'z': z_ratio}
    sorted_dims = sorted(dims.items(), key=lambda item: item[1], reverse=True)
    max_dim, max_val = sorted_dims[0]
    mid_dim, mid_val = sorted_dims[1]
    min_dim, min_val = sorted_dims[2]

    if max_val > 1.5 * mid_val:
        # 一维显著大于另外两维 (杆状/长条状)
        if max_dim == 'y':
            shape_desc = "细长直立杆 (Tall and vertical pole/stick, e.g., leg)"
        else:
            shape_desc = "水平长条 (Horizontal bar/rail, e.g., armrest, bottom bar)"
    elif max_val <= 1.5 * mid_val and mid_val > 1.5 * min_val:
        # 两个维度显著大于第三个维度 (板状/面状)
        if min_dim == 'y':
            shape_desc = "水平宽平板 (Horizontal wide plate, e.g., seat, tabletop)"
        elif min_dim == 'x':
            shape_desc = "纵向竖直侧板 (Vertical side plate stretching Front-Back, e.g., chair side frame)"
        elif min_dim == 'z':
            shape_desc = "横向竖直面板 (Vertical front plate stretching Left-Right, e.g., main backrest)"
    else:
        # 三个维度差不多
        shape_desc = "普通块状/球状 (Blocky or Spherical)"

    mapped["relative_dimensions"] = {
        "x_width_ratio": f"{x_ratio:.3f}",
        "y_height_ratio": f"{y_ratio:.3f} (占整体高度的比例)",
        "z_depth_ratio": f"{z_ratio:.3f}",
        "shape_description": shape_desc
    }

    return mapped

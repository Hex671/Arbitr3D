import os
import sys

# ============================================================
# 全局配置区 - 所有脚本共享的配置
# 所有视角相关的参数都在这里集中定义！
# ============================================================

# 1. 数据集与路径配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "config", "knowledge")
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(BASE_DIR, "config", "config.yaml"))
import yaml
with open(CONFIG_PATH, "r", encoding="utf-8") as _config_file:
    _config = yaml.safe_load(_config_file) or {}
DATASET_BASE_PATH = os.environ.get("KNOWLEDGE_DATA_ROOT") or _config.get("data", {}).get("knowledge_base_path", "datasets/PartNetE/few_shot")

# 2. 视角配置 - 核心参数！
# 所有涉及视角数量的地方都应该引用这里的配置

# 【视角知识库】使用的视角数量 (用于 gene_view_pre.py 和 gene_view_knowledge.py)
NUM_VIEWS = 10

# 【视角知识库】视角详细参数列表
# 每个视角: [elev(仰角), azim(方位角)]
# 修改视角数量：1. 改变 NUM_VIEWS  2. 修改 RAW_VIEWS 列表
# RAW_VIEWS = [
#     [15, 45], [15, 135], [15, 225], [15, 315],  # 四个中等俯角（偏前/右/后/左）
#     [45, 0], [45, 120], [45, 240], [0, 0],       # 四个高俯角
#     [-40, 0], [-40, 90], [-40, 180], [-40, 270], # 四个仰视角度
#     [-50, 45], [-50, 135], [-50, 225], [-50, 315],# 四个更低的仰视
# ]
RAW_VIEWS = [
    [10, 60], [10, 300],
    [40, 0], [40, 120], [40, 240],
    [-20, 45], [-20, 180], [-20, 315],
    [-40, 135], [-40, 225]]

# 【语义知识库】使用的视角数量（固定为4个经典视角）
NUM_SEM_VIEWS = 4

# 【语义知识库】经典四视角参数
# 用于 gene_sem_pre.py 的全景解剖图渲染
SEM_VIEWS = [
    [15, 0],    # 1. 正前面，略带俯视 (看清工作面/桌面和前腿)
    [-10, 90],  # 2. 右侧面，略带仰视 (看清底部连接部件、抽屉下方)
    [0, 180],   # 3. 正背面，平视 (看清背面是否有背板、支撑结构)
    [30, 270]   # 4. 左侧面，高俯视 (看清整体空间布局和顶面全貌)
]

# 语义知识库四视角的标签描述
SEM_VIEW_LABELS = [
    "View 1: Front-Top (Elev: 15, Azim: 0)",
    "View 2: Right-Bottom (Elev: -10, Azim: 90)",
    "View 3: Back-Flat (Elev: 0, Azim: 180)",
    "View 4: Left-High (Elev: 30, Azim: 270)"
]

# 3. 渲染配置
RENDER_RESOLUTION = (512, 512)  # 单张图分辨率
RENDER_RADIUS = 2.2             # 相机渲染半径

# 4. 输出目录配置
SEM_KNOWLEDGE_IMAGES_DIR = os.path.join(BASE_DIR, "sem_knowledge_images")
VIEW_KNOWLEDGE_IMAGES_DIR = os.path.join(BASE_DIR, "view_knowledge_images")

# 5. 批处理配置
SEM_BATCH_SIZE = 4  # 语义知识库的图片批处理大小
VIEW_BATCH_SIZE = 2 # 视角知识库的图片批处理大小


def get_category():
    """获取目标类别，优先从环境变量读取"""
    return os.environ.get("TARGET_CATEGORY", "Chair")


def get_views_info(num_views=NUM_VIEWS):
    """
    获取视角信息列表
    返回: [{"view_id": "View_00", "elev": 15, "azim": 45}, ...]
    """
    views_info = []
    for i in range(num_views):
        elev, azim = RAW_VIEWS[i] if i < len(RAW_VIEWS) else (0, 0)
        views_info.append({"view_id": f"View_{i:02d}", "elev": elev, "azim": azim})
    return views_info


def get_sem_views_info():
    """
    获取语义知识库四视角信息
    """
    views_info = []
    for i, (elev, azim) in enumerate(SEM_VIEWS):
        views_info.append({
            "view_id": f"View_{i:02d}",
            "elev": elev,
            "azim": azim,
            "label": SEM_VIEW_LABELS[i]
        })
    return views_info


# ============================================================
# 文件名模板配置 - 所有知识库文件名统一在这里管理！
# 使用函数动态生成带视角数量的文件名
# ============================================================

def get_sem_knowledge_filename(category=None):
    """获取语义知识库文件名"""
    if category is None:
        category = get_category()
    return f"{category}_sem_knowledge.json"


def get_view_knowledge_filename(category=None, num_views=NUM_VIEWS):
    """获取视角知识库文件名（原始/未精炼）"""
    if category is None:
        category = get_category()
    return f"{category}_{num_views}_view_knowledge.json"


def get_view_knowledge_refined_filename(category=None, num_views=NUM_VIEWS):
    """获取视角知识库文件名（精炼后）"""
    if category is None:
        category = get_category()
    return f"{category}_{num_views}_view_knowledge_refined.json"


def get_ultimate_knowledge_filename(category=None):
    """获取终极知识库文件名（兼容旧版本，作为语义知识库的备选）"""
    if category is None:
        category = get_category()
    return f"{category}_ultimate_knowledge.json"


def get_sem_knowledge_path(category=None):
    """获取语义知识库完整路径"""
    return os.path.join(KNOWLEDGE_DIR, get_sem_knowledge_filename(category))


def get_sem_knowledge_refined_filename(category=None):
    """获取语义知识库文件名（精炼后）"""
    if category is None:
        category = get_category()
    return f"{category}_sem_knowledge_refined.json"


def get_sem_knowledge_refined_path(category=None):
    """获取语义知识库完整路径（精炼后）"""
    return os.path.join(KNOWLEDGE_DIR, get_sem_knowledge_refined_filename(category))


def get_view_knowledge_path(category=None, num_views=NUM_VIEWS):
    """获取视角知识库完整路径（原始/未精炼）"""
    return os.path.join(KNOWLEDGE_DIR, get_view_knowledge_filename(category, num_views))


def get_view_knowledge_refined_path(category=None, num_views=NUM_VIEWS):
    """获取视角知识库完整路径（精炼后）"""
    return os.path.join(KNOWLEDGE_DIR, get_view_knowledge_refined_filename(category, num_views))


def get_ultimate_knowledge_path(category=None):
    """获取终极知识库完整路径"""
    return os.path.join(KNOWLEDGE_DIR, get_ultimate_knowledge_filename(category))

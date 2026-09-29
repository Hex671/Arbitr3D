"""
图像预处理增强模块
针对点云渲染图进行边缘强化，使其更易于SAM分割
"""
import cv2
import numpy as np
from typing import Optional


class PointCloudEdgeEnhancer:
    """
    点云渲染图边缘增强器

    专门针对点云渲染图的特点进行优化：
    - 点径较粗：由大量可见的点组成，不是连续的表面
    - 颗粒感强：点与点之间有明显的间隙和噪点
    - 边缘模糊：边缘由点堆积形成，不是锐利的像素边界
    - 可能有内部空洞：点云本身的不均匀性导致
    """

    def __init__(
        self,
        edge_strength: float = 0.8,
        edge_color: tuple = (25, 25, 25),
        point_size_estimate: int = 3,
        close_kernel_size: int = 3,
        blur_kernel_size: int = 3,
        unsharp_amount: float = 0.5,
        canny_low_thresh: int = 30,
        canny_high_thresh: int = 100,
        erode_kernel_size: int = 2,
        enable_enhancement: bool = True
    ):
        """
        初始化边缘增强器

        Args:
            edge_strength: 边缘强度 (0.5-1.0)，越大边缘越深
            edge_color: 边缘颜色 (BGR格式)，默认深灰色 (25, 25, 25)
            point_size_estimate: 点径估计值，用于形态学操作核大小
            close_kernel_size: 闭操作核大小，用于填补点云空洞
            blur_kernel_size: 高斯模糊核大小，用于去噪
            unsharp_amount: 反锐化掩膜强度
            canny_low_thresh: Canny边缘检测低阈值
            canny_high_thresh: Canny边缘检测高阈值
            erode_kernel_size: 腐蚀操作核大小，用于细化边缘
            enable_enhancement: 是否启用增强
        """
        self.edge_strength = edge_strength
        self.edge_color = np.array(edge_color, dtype=np.uint8)
        self.enable_enhancement = enable_enhancement

        # 形态学操作核
        kernel_size = max(3, point_size_estimate * 2 - 1)
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        self.kernel_blur = (blur_kernel_size, blur_kernel_size)
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (erode_kernel_size, erode_kernel_size))

        # 反锐化参数
        self.unsharp_amount = unsharp_amount

        # Canny参数
        self.canny_low_thresh = canny_low_thresh
        self.canny_high_thresh = canny_high_thresh

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        对点云渲染图进行边缘增强

        Args:
            image: RGB图像 (H, W, 3)

        Returns:
            边缘增强后的图像
        """
        if not self.enable_enhancement:
            return image

        if image is None or image.size == 0:
            return image

        # Step 1: 形态学闭操作 - 填补点云内部空洞
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        filled = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, self.kernel_close)

        # Step 2: 高斯模糊 - 减少颗粒噪点
        blurred = cv2.GaussianBlur(filled, self.kernel_blur, 0)

        # Step 3: 反锐化掩膜 - 增强局部对比度，让点的边界更锐利
        blur_for_unsharp = cv2.GaussianBlur(blurred, (0, 0), 1.5)
        sharpened = cv2.addWeighted(
            blurred,
            1.0 + self.unsharp_amount,
            blur_for_unsharp,
            -self.unsharp_amount,
            0
        )
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Step 4: Canny边缘检测
        edges = cv2.Canny(sharpened, self.canny_low_thresh, self.canny_high_thresh)

        # Step 5: 形态学腐蚀 - 细化边缘，让粗点边缘变细线
        edges_thin = cv2.erode(edges, self.kernel_erode, iterations=1)

        # Step 6: 边缘加深 - 二值化，确保边缘只有0或255
        _, edges_deep = cv2.threshold(edges_thin, 40, 255, cv2.THRESH_BINARY)

        # Step 7: 创建深色边缘图并融合
        enhanced = self._blend_edges(image, edges_deep)

        return enhanced

    def _blend_edges(self, image: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        将边缘叠加到原图上

        Args:
            image: 原图 (H, W, 3)
            edges: 二值化边缘图 (H, W)

        Returns:
            融合后的图像
        """
        # 扩展边缘到3通道
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 创建掩膜
        mask = edges[:, :, None] > 0

        # Alpha融合
        enhanced = image.copy().astype(np.float32)
        for c in range(3):
            enhanced[:, :, c] = np.where(
                mask[:, :, 0],
                enhanced[:, :, c] * (1 - self.edge_strength) + self.edge_color[c] * self.edge_strength,
                enhanced[:, :, c]
            )

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def enhance_batch(self, images: list) -> list:
        """批量增强图像（纯 RGB Canny 边缘增强）"""
        return [self.enhance(img) for img in images]

    def get_edge_mask_only(self, image: np.ndarray) -> np.ndarray:
        """
        仅获取边缘掩膜（用于调试或可视化）

        Args:
            image: RGB图像

        Returns:
            二值化边缘图
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        filled = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, self.kernel_close)
        blurred = cv2.GaussianBlur(filled, self.kernel_blur, 0)

        blur_for_unsharp = cv2.GaussianBlur(blurred, (0, 0), 1.5)
        sharpened = cv2.addWeighted(blurred, 1.0 + self.unsharp_amount, blur_for_unsharp, -self.unsharp_amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        edges = cv2.Canny(sharpened, self.canny_low_thresh, self.canny_high_thresh)
        edges_thin = cv2.erode(edges, self.kernel_erode, iterations=1)
        _, edges_deep = cv2.threshold(edges_thin, 40, 255, cv2.THRESH_BINARY)

        return edges_deep


def create_edge_enhancer(config: Optional[dict] = None) -> PointCloudEdgeEnhancer:
    """
    从配置字典创建边缘增强器

    Args:
        config: 配置字典，包含以下可选键：
            - enable_edge_enhancement: bool, 是否启用
            - edge_strength: float, 边缘强度
            - edge_color_bgr: tuple, BGR颜色
            - point_size_estimate: int, 点径估计
            - close_kernel_size: int, 闭操作核大小
            - blur_kernel_size: int, 模糊核大小
            - unsharp_amount: float, 反锐化强度
            - canny_low_thresh: int, Canny低阈值
            - canny_high_thresh: int, Canny高阈值
            - erode_kernel_size: int, 腐蚀核大小
    """
    if config is None:
        config = {}

    # 从配置中获取参数，使用默认值
    enhancer_config = config.get('edge_enhancement', {})

    return PointCloudEdgeEnhancer(
        edge_strength=enhancer_config.get('edge_strength', 0.8),
        edge_color=tuple(enhancer_config.get('edge_color_bgr', [25, 25, 25])),
        point_size_estimate=enhancer_config.get('point_size_estimate', 3),
        close_kernel_size=enhancer_config.get('close_kernel_size', 3),
        blur_kernel_size=enhancer_config.get('blur_kernel_size', 3),
        unsharp_amount=enhancer_config.get('unsharp_amount', 0.5),
        canny_low_thresh=enhancer_config.get('canny_low_thresh', 30),
        canny_high_thresh=enhancer_config.get('canny_high_thresh', 100),
        erode_kernel_size=enhancer_config.get('erode_kernel_size', 2),
        enable_enhancement=enhancer_config.get('enable', True)
    )

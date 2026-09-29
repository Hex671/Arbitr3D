from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class RenderOutput:
    """
    渲染输出数据结构
    """
    images: List[np.ndarray]      # (K, H, W, 3) RGB 图像列表
    depth_maps: List[np.ndarray]  # (K, H, W) 深度图 (Z-buffer) 列表
    index_maps: List[np.ndarray]  # (K, H, W) 像素到 3D 点的索引映射列表

@dataclass
class SoftMask2D:
    """
    2D 软语义掩码
    """
    mask: np.ndarray              # (H, W) 布尔矩阵，SAM 切出的纯几何边界
    probabilities: Dict[str, float] = field(default_factory=dict) # CLIP 赋予的类别概率分布
    reasoning: Optional[str] = None  # Step 2.0 MLLM 的推理理由

@dataclass
class PointProbabilityField:
    """
    3D 点概率场
    """
    point_cloud: object           # 原始 3D 点云 (Open3D PointCloud)
    prob_matrix: np.ndarray       # (N, C) 矩阵，N 为点数，C 为类别数
    entropy: Optional[np.ndarray] = None # (N,) 数组，记录每个点的信息熵

    def calculate_entropy(self):
        """计算信息熵"""
        # 避免 log(0)
        safe_probs = np.clip(self.prob_matrix, 1e-10, 1.0)
        self.entropy = -np.sum(safe_probs * np.log(safe_probs), axis=1)

@dataclass
class MaskPrediction:
    """
    Step 2.0 单掩码语义预测结果，包含推理理由。
    """
    mask_id: int
    semantic: str
    reasoning: str  # MLLM 的推理过程
    reasoning_source: str  # 推理依据（来自哪个视图或知识库）
    confidence: float

@dataclass
class FeatureDescriptor:
    """
    描述一个具体特征的字典结构。
    """
    feature_type: str  # e.g., "shape", "color", "texture", "spatial_relation", "structure"
    description: str  # 具体描述
    source: str  # 来源: "view_0", "view_1", "knowledge_base", "fusion"
    confidence: float  # 该特征的置信度 (0.0 - 1.0)

@dataclass
class ViewInstanceKnowledge:
    """
    单视角提取出的实例级知识。
    """
    view_idx: int
    confirmed_semantics: Dict[int, str]  # mask_id -> semantic
    instance_view_features: Dict[str, Dict[str, FeatureDescriptor]]  # semantic -> feature_type -> FeatureDescriptor
    confidence: float

@dataclass
class PartFeature:
    """
    融合后的部件级特征。
    """
    part_name: str
    features: Dict[str, FeatureDescriptor]  # feature_type -> FeatureDescriptor
    evidence_summary: str  # 多视角/知识的证据总结
    is_normal_variation: bool  # 是否是正常变异（而非错误）
    multi_view_agreement: float  # 多视角一致性 (0.0 - 1.0)

@dataclass
class InstanceKnowledge:
    """
    完整的实例级知识库，用于指导后续的 3D 掩码修正。
    仅保留 instance_part_features 和 low_confidence_parts。
    """
    instance_id: str
    instance_part_features: Dict[str, PartFeature]  # semantic -> PartFeature
    low_confidence_parts: Dict[str, float]  # semantic -> confidence (低于阈值的部件)

    def to_dict(self) -> dict:
        """将实例知识转换为可序列化的字典，用于保存到文件。"""
        return {
            "instance_id": self.instance_id,
            "instance_part_features": {
                part_name: {
                    "part_name": pf.part_name,
                    "features": {
                        feat_type: {
                            "feature_type": fd.feature_type,
                            "description": fd.description,
                            "source": fd.source,
                            "confidence": fd.confidence
                        }
                        for feat_type, fd in pf.features.items()
                    },
                    "evidence_summary": pf.evidence_summary,
                    "is_normal_variation": pf.is_normal_variation,
                    "multi_view_agreement": pf.multi_view_agreement
                }
                for part_name, pf in self.instance_part_features.items()
            },
            "fusion_confidence": 1.0,
            "low_confidence_parts": self.low_confidence_parts
        }

    def to_prompt_string(self) -> str:
        """将实例知识转换为可用于 MLLM Prompt 的字符串。"""
        lines = [f"## 实例 {self.instance_id} 的实例级知识库\n"]

        if not self.instance_part_features:
            lines.append("当前尚未提取到有效的部件特征。\n")
            return "".join(lines)

        lines.append("### 各部件的关键特征\n")
        for part_name, part_feature in self.instance_part_features.items():
            lines.append(f"- **{part_name}**: {part_feature.evidence_summary}")
            if part_feature.features:
                lines.append("  - 详细特征:")
                for feat_type, feat_desc in part_feature.features.items():
                    lines.append(f"    - [{feat_type}] {feat_desc.description} (置信度: {feat_desc.confidence:.2f})")

        if self.low_confidence_parts:
            lines.append("\n### 低置信度部件警告\n")
            for part, conf in self.low_confidence_parts.items():
                lines.append(f"- {part}: 置信度 {conf:.2f}，建议谨慎处理。")

        return "".join(lines)

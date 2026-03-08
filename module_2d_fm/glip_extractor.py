from typing import List, Dict
import numpy as np

class GLIPExtractor:
    def __init__(self, weight_path: str):
        self.weight_path = weight_path
        self.model = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading GLIP from {self.weight_path}...")
            # TODO: 实例化真实的 GLIP 模型
            # from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
            self.model = "GLIP_Model_Instance"

    def get_bboxes_and_scores(self, images: List[np.ndarray], prompt_classes: str) -> List[List[Dict]]:
        """
        对 K 张图执行 GLIP 开放词汇检测
        prompt_classes: 如 "chair arm, chair back, chair leg, chair seat"
        
        返回: 
            K 个列表，每个列表中包含该图所有的 detection 结果字典
            [{'bbox': [x1, y1, x2, y2], 'score': 0.9, 'label': 'chair back'}, ...]
        """
        self._load_model()
        results_across_views = []
        
        for img in images:
            # 伪代码: 真实逻辑需要调用 GLIP 并提取 bbox, score, label
            # results = self.model.compute_prediction(img, prompt_classes)
            mock_detections = [
                {'bbox': [100, 100, 200, 200], 'score': 0.85, 'label': 'chair back'}
            ]
            results_across_views.append(mock_detections)
            
        return results_across_views

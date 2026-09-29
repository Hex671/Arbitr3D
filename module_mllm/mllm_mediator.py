import json
import re
import base64
from typing import Dict, List
import numpy as np
from PIL import Image
from datatypes import PointProbabilityField

class MLLMMediator:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            print("OpenAI package not installed. MLLMMediator will not work.")

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        import io
        import cv2
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def arbitrate_conflict(self, highlight_image: np.ndarray, conflict_info: Dict, prompt_classes: List[str], object_category: str = "Object", views_angle_info: str = "") -> str:
        # 必须与 fusion 里保持绝对一致！
        classes = prompt_classes + ["unlabeled", "background"]
        if self.client is None:
            print("Warning: OpenAI client not initialized. Returning fallback.")
            probs = conflict_info['avg_probs']
            best_idx = np.argmax(probs)
            return classes[best_idx]

        base64_image = self._encode_image_to_base64(highlight_image)

        prob_dict = {cls: round(p, 3) for cls, p in zip(classes, conflict_info['avg_probs']) if p > 0.05}

        prompt = (
            "你是一位顶尖的 3D 物理与几何分析专家。\n"
            f"图中展示了从几个不同关键视角观察同一个【{object_category}】（物体类别）的拼贴图。\n"
            f"【图中各画面的物理视角信息】：\n{views_angle_info}\n"
            "（注：仰角为正代表俯视，容易看到顶部；仰角为负代表仰视，容易看到底部支撑。请仔细对比同一高亮部位在不同仰角下的形态变化。）\n\n"
            "在每个画面中，用红色半透明遮罩和轮廓高亮标出的区域是同一个物理部件，它在初步检测中存在语义争议。\n"
            f"目前 AI 的初步概率分布为：{json.dumps(prob_dict, ensure_ascii=False)}\n"
            f"该物体的合法候选部件为：{prompt_classes}\n\n"
            "【任务与推理要求】：\n"
            "请综合拼贴图中的视觉信息和【物理视角信息】，结合图像中展示的空间拓扑关系（例如：谁在下方起支撑作用？谁在侧面做围挡？在当前仰角下这块红斑到底处于什么空间位置？），"
            "裁定该红色高亮区域在 3D 空间中真正属于哪个类别。\n\n"
            "要求：直接输出你选定的类别名称（英文），不要包含任何其他文字、推理过程、标点或 JSON 格式。\n"
            "示例输出：leg"
        )

        import time
        max_retries = 2
        retry_count = 0
        success = False
        response_text = ""

        while retry_count <= max_retries and not success:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=100,
                    temperature=0.1,
                    timeout=60.0  # 新增超时
                )
                response_text = (response.choices[0].message.content or "").strip()
                success = True

            except Exception as e:
                retry_count += 1
                err_msg = str(e)
                cluster_id = conflict_info.get('cluster_id', 'unknown')
                print(f"\n[Mediator] ❌ 裁决争议簇 {cluster_id} 时 MLLM API 调用失败 (尝试 {retry_count}/{max_retries+1}): {err_msg}")

                if retry_count <= max_retries:
                    print(f"等待 5 秒后准备重新发起请求...")
                    time.sleep(5)
                else:
                    # 如果这最后一步的几个点一直失败，你可以选择阻断，也可以选择像之前一样 fallback
                    # 为了批处理的严谨性，这里依然建议抛出异常阻断
                    raise RuntimeError(f"🚨 [致命错误] Mediator 裁决时 MLLM API 彻底失败。原因: {err_msg}")

        # 以下是解析逻辑 (保持不变，只是向外缩进对齐)
        decision = response_text.strip('`"\'{}[]\n\r\t ')

        matched = False
        for cls in classes:
            if cls.lower() in decision.lower():
                decision = cls
                matched = True
                break

        if not matched:
            decision = ''

        if decision in classes:
            return decision
        else:
            print(f"MLLM returned invalid class: {decision}")
            best_idx = np.argmax(conflict_info['avg_probs'])
            return classes[best_idx]

    def overwrite_probabilities(self, prob_field: PointProbabilityField, conflict_cluster: Dict, mllm_decision: str, prompt_classes: List[str]):
        # 严格保持类别长度与 prob_matrix 完全一致
        classes = prompt_classes + ["unlabeled", "background"]

        if mllm_decision not in classes:
            print(f"Warning: Mediator decided an invalid class '{mllm_decision}'. Ignoring overwrite.")
            return

        target_idx = classes.index(mllm_decision)
        point_indices = conflict_cluster['point_indices']

        num_classes = len(classes)
        other_prob = 0.01 / max(1, num_classes - 1)

        # 这里的 num_classes 现在一定是跟 prob_matrix 的列数对齐的，绝不会再发生 shape mismatch
        new_probs = np.ones(num_classes, dtype=np.float32) * other_prob
        new_probs[target_idx] = 0.99

        prob_field.prob_matrix[point_indices] = new_probs

        safe_probs = np.clip(new_probs, 1e-10, 1.0)
        new_entropy = -np.sum(safe_probs * np.log(safe_probs))
        if prob_field.entropy is not None:
            prob_field.entropy[point_indices] = new_entropy

from typing import Dict
import base64
import json
import re
from PIL import Image

class MLLMArbitrator:
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
            print("OpenAI package not installed. MLLMArbitrator will not work.")

    def construct_prompt(self, prior_dict: Dict[str, Dict[str, float]]) -> str:
        prompt = (
            "你是一位顶尖的 3D 物理与几何分析专家。图中不同视角的相同数字代表同一个拥有完美物理边界的部件。\n"
            f"这是 2D 基础模型提供的参考分布：{json.dumps(prior_dict, ensure_ascii=False)}\n"
            "任务：请利用你的物理常识（例如：谁在下方起支撑作用？谁具有容纳功能？），"
            "结合图像中展示的空间拓扑关系，纠正基础模型的歧义。\n"
            "要求：使用思维链一步步推理，最后输出严格的 JSON 格式，为每个 ID 分配一个唯一的正确类别。\n"
            "JSON 格式示例：```json\n{\"1\": \"chair back\", \"2\": \"chair leg\"}\n```"
        )
        return prompt

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def call_mllm_api(self, som_image: Image.Image, prompt: str) -> str:
        """调用 MLLM API"""
        if self.client is None:
            print("Warning: OpenAI client not initialized. Returning mock JSON.")
            return '```json\n{"1": "mock_part"}\n```'
            
        base64_image = self._encode_image_to_base64(som_image)
        
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
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Call failed: {e}")
            return '```json\n{}\n```'

    def parse_mllm_json(self, response_text: str) -> Dict[str, str]:
        """鲁棒地提取 JSON 映射表"""
        try:
            # 提取被 ```json ``` 包裹的内容
            json_str_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_str_match:
                json_str = json_str_match.group(1)
            else:
                json_str = response_text
                
            # 尝试找到第一个 { 和 最后一个 } 之间的内容
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = json_str[start:end]
                
            return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse JSON from MLLM response: {e}")
            return {}

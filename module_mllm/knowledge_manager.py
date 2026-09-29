import os
import json
import re
import time

class CategoryKnowledgeManager:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o", config_dir: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        self.config_dir = config_dir if config_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            print("OpenAI package not installed. KnowledgeManager will not work.")

    def _call_llm(self, prompt: str) -> str:
        if self.client is None:
            return ""

        max_retries = 2
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # 增加 timeout=60.0 防止卡死
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=60.0
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                retry_count += 1
                err_msg = str(e)
                print(f"\n[KnowledgeManager] ❌ LLM API 调用失败 (尝试 {retry_count}/{max_retries+1}): {err_msg}")

                if retry_count <= max_retries:
                    print(f"等待 5 秒后准备重新发起请求...")
                    time.sleep(5)
                else:
                    # 获取解剖学知识是 Pipeline 第一步，失败必须直接阻断
                    raise RuntimeError(f"🚨 [致命错误] KnowledgeManager LLM API 彻底失败。原因: {err_msg}")

    # ========================================================
    # 新增的离线知识加载接口 (包含消融实验开关)
    # ========================================================
    def load_offline_knowledge(self, object_category: str, view_angles_str: list) -> tuple[str, str]:
        """
        尝试加载离线生成的解剖学知识和视角知识。
        优先使用精炼后的知识库，回退到原始版本。
        返回: (anatomy_knowledge_json_str, view_guidance_json_str)
        """
        from scripts.config import get_view_knowledge_refined_filename, get_sem_knowledge_refined_filename, KNOWLEDGE_DIR

        # 优先使用精炼后的语义知识库，回退到原始版本
        sem_refined_path = os.path.join(KNOWLEDGE_DIR, get_sem_knowledge_refined_filename(object_category))
        sem_path = os.path.join(KNOWLEDGE_DIR, f"{object_category}_sem_knowledge.json")
        view_path = os.path.join(KNOWLEDGE_DIR, get_view_knowledge_refined_filename(object_category))

        # 1. 处理解剖学知识 (语义知识) - 优先使用精炼版本
        if os.path.exists(sem_refined_path):
            sem_path_to_use = sem_refined_path
            print(f"[KnowledgeManager] ✅ 加载精炼后的语义知识库: {sem_refined_path}")
        elif os.path.exists(sem_path):
            sem_path_to_use = sem_path
            print(f"[KnowledgeManager] ⚠️ 未找到精炼语义知识库，使用原始版本: {sem_path}")
        else:
            sem_path_to_use = None

        if sem_path_to_use:
            with open(sem_path_to_use, 'r', encoding='utf-8') as f:
                sem_dict = json.load(f)
            anatomy_str = json.dumps(sem_dict, ensure_ascii=False)
        else:
            anatomy_str = json.dumps({"anatomy_knowledge": {"predefined_parts": {}, "extra_parts": {}}}, ensure_ascii=False)
            print(f"[KnowledgeManager] ⚠️ 未找到 {object_category} 的离线语义知识库，返回空结构。")

        # 2. 处理视角知识
        pipeline_views_dict = {}
        if os.path.exists(view_path):
            with open(view_path, 'r', encoding='utf-8') as f:
                view_dict = json.load(f)

            view_knowledge = view_dict.get("view_knowledge", {})
            for i, angle_str in enumerate(view_angles_str):
                view_id = f"View_{i:02d}"
                v_info = view_knowledge.get(view_id, {})
                if v_info:
                    pipeline_views_dict[angle_str] = {
                        "highly_visible_parts": v_info.get("visible_parts", []),
                        "occluded_parts": v_info.get("occluded_parts", []),
                        "2d_shape_expectations": v_info.get("view_characteristics", "")
                    }
                else:
                    pipeline_views_dict[angle_str] = {}
        else:
            for angle_str in view_angles_str:
                 pipeline_views_dict[angle_str] = {}
            print(f"[KnowledgeManager] ⚠️ 未找到 {object_category} 的离线视角知识库，返回空结构。")

        view_guidance_str = json.dumps({"views": pipeline_views_dict}, ensure_ascii=False)

        # 无论如何都返回字符串，避免外部触发在线 MLLM 推理（除非我们在做对比实验）
        return anatomy_str, view_guidance_str


    # ========================================================
    # 保留原有的在线回退逻辑
    # ========================================================
    def build_category_knowledge(self, object_category: str, prompt_classes: list) -> str:
        print(f"--- [Knowledge Base] Step 1: Generating Anatomy Knowledge & Spatial Prior for {object_category} ---")
        prompt = (
            f"你是一位资深的 3D 几何与工业设计专家。\n"
            f"我现在需要对一个 3D 模型进行细粒度的部件分割。\n"
            f"物体类别是：【{object_category}】。\n"
            f"预先定义好的目标部件列表是：{prompt_classes}。\n\n"
            f"假设该物体被放置在一个标准的 3D 坐标系中，底部接触地面，Z 轴代表高度（0.0 代表最底部，1.0 代表最顶部）。\n"
            f"请你深入分析这个物体类别，提供以下信息：\n"
            f"1. 对于每一个【预定义部件】，详细描述它的典型 3D 几何形状和连接关系。\n"
            f"2. 【空间物理约束】：请预估每个预定义部件在 Z 轴（高度）上**最可能出现的相对区间范围**。用 [min_z, max_z] 表示。为了增加容错，请给出稍微宽泛、保守的区间。例如桌面的范围通常是 [0.7, 1.0]，桌腿是 [0.0, 0.9]。这对于我们过滤明显不合理的视觉误判极其重要。\n"
            f"3. 思考在这个物体类别中，除了预定义部件外，通常还会出现哪些【额外常见部件】（例如桌子可能还有把手、支撑横梁、脚垫等）。\n"
            f"   对于每一个额外常见部件，请给出它的特征，包括典型 3D 几何形状、连接关系等。\n"
            f"   如果你认为预定义部件列表已经非常完善，包含了该物体几乎所有的可能部件，你可以输出 0 个额外部件（即保持 extra_parts 为空对象）。\n\n"
            f"你必须严格输出如下格式的 JSON（不要输出其他任何文字）：\n"
            "{\n"
            '  "anatomy_knowledge": {\n'
            '    "predefined_parts": {\n'
            '      "部件名1": {\n'
            '        "spatial_location": "...",\n'
            '        "typical_3D_shape": "...",\n'
            '        "connection_context": "...",\n'
            '        "z_height_range": [0.0, 1.0]\n'
            '      }\n'
            '    },\n'
            '    "extra_parts": {\n'
            '      "额外部件名A": {\n'
            '          "feature": "...",\n'
            '      }\n'
            '    }\n'
            '  }\n'
            "}"
        )

        response = self._call_llm(prompt)

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        else:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx+1]

        return response

    def build_view_guidance(self, object_category: str, category_knowledge: str, view_angles_str: list) -> str:
        print(f"--- [Knowledge Base] Step 2: Generating View-Specific Guidance for {len(view_angles_str)} views ---")
        views_list_text = "\n".join([f"- {v}" for v in view_angles_str])

        prompt = (
            f"你现在是一位计算机视觉与 3D 渲染专家。\n"
            f"基于以下关于【{object_category}】的 3D 部件解剖知识：\n"
            f"{category_knowledge}\n\n"
            f"我们使用了多个虚拟相机环绕该物体进行拍摄。相机的参数列表如下：\n"
            f"{views_list_text}\n"
            f"（注：仰角 > 0 为俯视，仰角 < 0 为仰视，趋近 0 为平视。）\n\n"
            f"请你针对上面列表中的**每一个相机视角**，分析其在 2D 渲染图中会呈现出什么样的特征：\n"
            f"1. 哪些预定义部件最容易被看见？哪些会被遮挡？\n"
            f"2. 那些可见的部件，在 2D 画面中会呈现出什么样的几何形状（比如俯视时，桌腿可能只露出几个小圆点；仰视时，桌面可能看不见工作面而只能看到粗糙的底板）？\n\n"
            f"你必须严格输出如下格式的 JSON（不要输出其他任何文字）：\n"
            "{\n"
            '  "views": {\n'
            '    "仰角 XX度, 方位角 YY度": {\n'
            '      "highly_visible_parts": ["部件名", ...],\n'
            '      "occluded_parts": ["部件名", ...],\n'
            '      "2d_shape_expectations": "该视角下...(该视角的大致特征)，部件1...，部件2...，部件n..."\n'
            '    }\n'
            '  }\n'
            "}"
        )

        response = self._call_llm(prompt)

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        else:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx+1]

        return response

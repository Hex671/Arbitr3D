import os
import sys
import json
import base64
import time
import re
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入集中配置
from scripts.config import (
    BASE_DIR, KNOWLEDGE_DIR, CONFIG_PATH, get_category, get_views_info, NUM_VIEWS,
    VIEW_KNOWLEDGE_IMAGES_DIR, VIEW_BATCH_SIZE,
    get_sem_knowledge_path, get_ultimate_knowledge_path, get_view_knowledge_path
)

try:
    from openai import OpenAI
except ImportError:
    print("❌ 请先安装 openai 包: pip install openai")
    sys.exit(1)


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_json_from_response(response_text: str) -> str:
    """
    从 MLLM 回复中提取可解析的 JSON 字符串。
    处理：完整 ```json 块、未闭合的 fence、纯 { ... }。
    """
    s = (response_text or "").strip()
    if not s:
        return ""

    # 1) 成对 ```json ... ```（gene_sem_knowledge 同款）
    json_match = re.search(r'```json\s*(.*?)\s*```', s, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # 2) 未闭合 fence：去掉开头的 ```json 或 ``` 行，从第一个 { 开始
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :].strip()
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()

    start_idx = s.find("{")
    end_idx = s.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        return s[start_idx : end_idx + 1].strip()

    return s.strip()


class ViewKnowledgeGenerator:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = CONFIG_PATH

        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        from utils.runtime_config import resolve_api_key, resolve_base_url, resolve_model_name
        api_key = resolve_api_key(config["mllm"])
        base_url = resolve_base_url(config["mllm"])
        self.model_name = resolve_model_name(config["mllm"])
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_mllm(self, messages: list, max_retries: int = 5, fallback_text: str = "") -> str:
        retry_count = 0
        while retry_count < max_retries:
            try:
                print(f"    [API] 正在呼叫大模型 (尝试 {retry_count + 1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=120.0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                retry_count += 1
                print(f"    ❌ 调用失败: {e}")
                time.sleep(5)
        return fallback_text

    def stage1_text_prior(self, object_category: str, sem_knowledge: Dict, views_info: List[Dict]) -> str:
        """阶段一：结合解剖学知识，纯凭常识推理多视角的可见性先验"""
        print(f"\n▶️ [阶段一] 开始结合常识推理 {object_category} 的 {len(views_info)} 个视角的文本先验知识...")

        sem_knowledge_str = json.dumps(sem_knowledge, ensure_ascii=False, indent=2)
        views_str = json.dumps(views_info, ensure_ascii=False, indent=2)

        prompt = (
            f"你是一位资深的 3D 几何与多视角感知专家。\n"
            f"我现在需要为类别【{object_category}】建立多视角识别知识库。\n"
            f"下面是该类别的 3D 解剖学知识（包括各个部件的位置、形状和高度范围）：\n"
            f"```json\n{sem_knowledge_str}\n```\n\n"
            f"我设定了 {len(views_info)} 个固定的相机观测视角，它们的仰角(elev)和方位角(azim)定义如下：\n"
            f"```json\n{views_str}\n```\n"
            f"请你发挥空间想象力，逐个分析这 {len(views_info)} 个视角。\n"
            f"针对每个视角，请你给出：\n"
            f"1. `visible_parts`: 在这个视角下，【最容易被看到且特征最明显】的部件列表。\n"
            f"2. `occluded_parts`: 在这个视角下，【极易被遮挡或者完全看不见】的部件列表。\n"
            f"3. `view_characteristics`: 这个视角的整体视觉特征描述（例如：'这是一个从右前方略微仰视的角度，能清晰看到桌子底部的支撑结构，但桌面完全是一条线或不可见'）。\n\n"
            f"你必须严格输出如下格式的 JSON（包含完整的视角信息）：\n"
            "{\n"
            '  "view_knowledge": {\n'
            '    "View_00": {\n'
            '      "elev": 15.0,\n'
            '      "azim": 45.0,\n'
            '      "visible_parts": ["tabletop", "leg"],\n'
            '      "occluded_parts": ["backboard"],\n'
            '      "view_characteristics": "从右前上方俯视，能看清整个桌面和右侧的前腿。"\n'
            '    },\n'
            '    "View_01": { ... },\n'
            '    ...\n'
            '  }\n'
            "}"
        )

        messages = [{"role": "user", "content": prompt}]
        response_text = self._call_mllm(messages, fallback_text='{"view_knowledge": {}}')
        return extract_json_from_response(response_text)

    def stage2_visual_refinement(self, object_category: str, view_id: str, view_info: Dict, current_desc: Dict, image_paths: List[str]) -> str:
        """阶段二：看着实际渲染的图片，对某个特定视角进行纠偏和特征追加"""
        current_desc_str = json.dumps(current_desc, ensure_ascii=False, indent=2)
        empty_update = json.dumps({"view_knowledge": {view_id: {}}}, ensure_ascii=False)

        prompt_text = (
            f"你现在是一位计算机视觉专家。\n"
            f"我们正在研究【{object_category}】在特定相机视角下的表现。\n"
            f"当前视角的 ID 是 {view_id}，相机的物理参数是：仰角 {view_info['elev']} 度，方位角 {view_info['azim']} 度。\n"
            f"这是在不看图的情况下，凭借常识推断出来的该视角特征：\n"
            f"```json\n{current_desc_str}\n```\n\n"
            f"我为你提供了几张实际渲染的多实例拼图。请仔细观察这些图片在【该视角下】的真实表现。\n"
            f"【核心任务】：\n"
            f"若与常识有差异，或观察到该视角下值得记录的新现象，只输出**增量** JSON；否则输出空更新。\n\n"
            f"**规则 1**：只输出 JSON，不要输出任何解释性文字；不要使用 markdown 代码块（不要 ```）。\n"
            f"**规则 2**：增量时只在 `{view_id}` 下输出字符串字段 `view_characteristics`（一句新观察）。\n"
            f"**规则 3**：`view_characteristics` 必须极短：**不超过 80 个汉字**，一句话说完，避免输出被截断。\n"
            f"**规则 4**：不要输出 `visible_parts` / `occluded_parts` 等列表字段（阶段二只追加文字观察）。\n\n"
            f"若无新信息，请**只输出**这一行（整行可复制）：\n"
            f"{empty_update}\n\n"
            f"若有新信息，输出示例（注意引号闭合、结构完整）：\n"
            "{{\n"
            f'  "view_knowledge": {{\n'
            f'    "{view_id}": {{\n'
            '      "view_characteristics": "一句不超过80字的观察。"\n'
            "    }\n"
            "  }\n"
            "}\n"
        )

        content_list = [{"type": "text", "text": prompt_text}]
        for img_path in image_paths:
            base64_img = encode_image_to_base64(img_path)
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}", "detail": "high"},
            })

        messages = [{"role": "user", "content": content_list}]

        max_retries = 5
        retry_count = 0
        response_text = ""
        extracted_json = ""

        while retry_count < max_retries:
            try:
                print(f"      [API] 看图纠偏中 (尝试 {retry_count + 1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    # 与 gene_sem_knowledge：视觉+JSON 需要更大上限；原先 1024 易截断
                    max_tokens=4096,
                    timeout=120.0,
                )
                choice = response.choices[0]
                response_text = (choice.message.content or "").strip()
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    print("      ⚠️ 本回复因 max_tokens 到达上限被截断，将重试；若多次出现请再提高 max_tokens 或缩短图片/detail。")

                extracted_json = extract_json_from_response(response_text)
                json.loads(extracted_json)
                return extracted_json

            except json.JSONDecodeError as e:
                print(f"      ⚠️ JSON 解析失败: {e}")
                print(
                    f"      ----- 模型原始回复（前 2000 字符）-----\n{response_text[:2000]}\n"
                    f"      ----- 提取子串（前 2000 字符）-----\n{(extracted_json[:2000] if extracted_json else '(空)')}\n"
                )
                retry_count += 1
                time.sleep(3)
            except Exception as e:
                print(f"      ❌ 异常: {e}")
                retry_count += 1
                time.sleep(5)

        return json.dumps({"view_knowledge": {view_id: {}}}, ensure_ascii=False)


def main():
    CATEGORY = get_category()

    sem_knowledge_path = get_ultimate_knowledge_path()
    if not os.path.exists(sem_knowledge_path):
        sem_knowledge_path = get_sem_knowledge_path()
        if not os.path.exists(sem_knowledge_path):
            print(f"❌ 找不到语义知识库文件！请先生成。寻找路径: {sem_knowledge_path}")
            return

    with open(sem_knowledge_path, 'r', encoding='utf-8') as f:
        sem_knowledge = json.load(f)

    VIEW_IMAGE_DIR = os.path.join(VIEW_KNOWLEDGE_IMAGES_DIR, CATEGORY)
    BATCH_IMAGES = VIEW_BATCH_SIZE

    print(f"🚀 开始为类别 {CATEGORY} 构建增量式 MLLM 离线【视角知识库】...")
    generator = ViewKnowledgeGenerator()

    # 从配置获取视角信息
    views_info = get_views_info(NUM_VIEWS)

    view_knowledge_json_str = generator.stage1_text_prior(CATEGORY, sem_knowledge, views_info)

    try:
        current_knowledge_dict = json.loads(view_knowledge_json_str)
        print("  ✅ 阶段一：纯文本视角特征推理已完成，骨架搭建完毕。")
    except Exception as e:
        print(f"  ❌ 阶段一 JSON 格式异常。程序中断。错误信息: {e}")
        return

    if not os.path.exists(VIEW_IMAGE_DIR):
        print(f"⚠️ 找不到图片目录: {VIEW_IMAGE_DIR}。将跳过视觉验证阶段。")
    else:
        print(f"\n▶️ [阶段二] 准备对 {NUM_VIEWS} 个视角进行图片特征验证与安全追加...")

        for v_idx in range(NUM_VIEWS):
            view_id = f"View_{v_idx:02d}"
            target_view_dir = os.path.join(VIEW_IMAGE_DIR, view_id)

            if not os.path.exists(target_view_dir):
                print(f"  ⚠️ 缺少文件夹 {target_view_dir}，跳过此视角。")
                continue

            all_images = [os.path.join(target_view_dir, f) for f in os.listdir(target_view_dir) if f.endswith(".png")]
            all_images.sort()
            total_images = len(all_images)

            if total_images == 0:
                continue

            print(f"\n  [视角 {view_id}] 发现 {total_images} 张拼图，开始分批阅片...")

            current_desc = current_knowledge_dict.get("view_knowledge", {}).get(view_id, {})

            for i in range(0, total_images, BATCH_IMAGES):
                batch_images = all_images[i : i + BATCH_IMAGES]
                print(f"    处理第 {i + 1} 到 {min(i + BATCH_IMAGES, total_images)} 张拼图...")

                update_json_str = generator.stage2_visual_refinement(
                    object_category=CATEGORY,
                    view_id=view_id,
                    view_info=views_info[v_idx],
                    current_desc=current_desc,
                    image_paths=batch_images,
                )

                try:
                    update_dict = json.loads(update_json_str)
                    new_view_info = update_dict.get("view_knowledge", {}).get(view_id, {})

                    has_changes = False
                    for key, new_val in new_view_info.items():
                        if isinstance(new_val, str):
                            old_val = current_knowledge_dict["view_knowledge"][view_id].get(key, "")
                            clean_new_val = (
                                new_val.replace("在原有基础上，补充：", "")
                                .replace("在原有基础上补充：", "")
                                .replace("补充：", "")
                                .strip()
                            )

                            if clean_new_val and clean_new_val not in old_val:
                                separator = " 此外看图发现：" if old_val else ""
                                current_knowledge_dict["view_knowledge"][view_id][key] = old_val + separator + clean_new_val
                                has_changes = True

                    if has_changes:
                        print("      ✅ 发现了该视角独有的视觉新特征，已安全追加到知识库。")
                        current_desc = current_knowledge_dict["view_knowledge"][view_id]

                except Exception as e:
                    print(f"      ⚠️ 合并发生意外: {e}，跳过本批次修正。")

    output_file = get_view_knowledge_path()
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_knowledge_dict, f, indent=4, ensure_ascii=False)
        print(f"\n🎉 完美！视角知识库已成功生成并保存至: {output_file}")
    except Exception as e:
        print(f"\n⚠️ 最终保存文件时发生未知错误: {e}")


if __name__ == "__main__":
    main()

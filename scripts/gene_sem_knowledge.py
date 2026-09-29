import os
import sys
import json
import base64
import time
import re
from typing import List

# 确保能找到 Arbitr3D 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入集中配置
from scripts.config import (
    BASE_DIR, KNOWLEDGE_DIR, CONFIG_PATH, get_category,
    SEM_KNOWLEDGE_IMAGES_DIR, SEM_BATCH_SIZE,
    get_sem_knowledge_path
)

# 尝试导入你之前配置好的 OpenAI 客户端
try:
    from openai import OpenAI
except ImportError:
    print("❌ 请先安装 openai 包: pip install openai")
    sys.exit(1)

def encode_image_to_base64(image_path: str) -> str:
    """将本地图片转换为 Base64 编码，供 MLLM 视觉使用"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_from_response(response_text: str) -> str:
    """从 MLLM 回复中尽力提取 JSON 字符串"""
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    if start_idx != -1 and end_idx != -1:
        return response_text[start_idx:end_idx+1].strip()

    return response_text.strip()

class OfflineKnowledgeGenerator:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = CONFIG_PATH

        # 从你现有的 config.yaml 中读取 API key 和模型名称
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        from utils.runtime_config import resolve_api_key, resolve_base_url, resolve_model_name
        api_key = resolve_api_key(config["mllm"])
        base_url = resolve_base_url(config["mllm"])
        self.model_name = resolve_model_name(config["mllm"])

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_mllm_with_retry(self, messages: list, max_retries: int = 4, fallback_text: str = "") -> str:
        """带重试和超时机制的 MLLM 调用"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                print(f"    [API] 正在呼叫大模型 (尝试 {retry_count + 1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=120.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                retry_count += 1
                print(f"    ❌ 调用失败: {e}")
                if retry_count < max_retries:
                    print("    等待 5 秒后重试...")
                    time.sleep(5)
                else:
                    print("    ⚠️ MLLM API 连续调用失败，跳过本次请求。")
                    return fallback_text

    def stage1_text_prior(self, object_category: str, prompt_classes: List[str]) -> str:
        """阶段一：纯文本获取先验知识"""
        print(f"\n▶️ [阶段一] 开始获取 {object_category} 的纯文本先验解剖学知识...")

        prompt = (
            f"你是一位资深的 3D 几何与工业设计专家。\n"
            f"我现在需要对一个 3D 模型进行细粒度的部件分割。\n"
            f"物体类别是：【{object_category}】。\n"
            f"预先定义好的目标部件列表是：{prompt_classes}。\n\n"
            f"假设该物体被放置在一个标准的 3D 坐标系中，底部接触地面，Z 轴（对于本数据集是 Up-axis 高度轴）代表高度（0.0 代表最底部，1.0 代表最顶部）。\n"
            f"请你深入分析这个物体类别，提供以下信息：\n"
            f"1. 对于每一个【预定义部件】，详细描述它的典型 3D 几何形状和连接关系。\n"
            f"2. 【空间物理约束】：请预估每个预定义部件在高度轴上**最可能出现的相对区间范围**。用 [min_height, max_height] 表示。这对于我们过滤明显不合理的视觉误判极其重要。\n"
            f"   🚨 警告：这是确立物理规则的唯一机会！区间必须能体现该部件的物理约束意义。比如桌面（tabletop）通常只存在于物体的偏上部分，它的区间绝不可能是 [0.05, 1.0] 这种横跨整个物体的无意义区间。正常的桌面范围可能是在 [0.6, 1.0]；而轮子（wheel）一定在最底部，范围可能是 [0.0, 0.15]。请务必严谨判断！\n"
            f"3. 思考在这个物体类别中，除了预定义部件外，通常还会出现哪些【额外常见部件】。\n"
            f"你必须严格输出如下格式的 JSON：\n"
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
            '    "extra_parts": {}\n'
            '  }\n'
            "}"
        )

        messages = [{"role": "user", "content": prompt}]
        fallback = '{"anatomy_knowledge": {"predefined_parts": {}, "extra_parts": {}}}'
        response_text = self._call_mllm_with_retry(messages, fallback_text=fallback)
        return extract_json_from_response(response_text)

    def stage2_visual_refinement_batch(self, object_category: str, prompt_classes: List[str], current_knowledge_json: str, image_paths: List[str]) -> str:
        """阶段二子步骤：接收一批图片，以增量差异(Diff)的形式完善知识库"""
        prompt_text = (
            f"你现在是一位计算机视觉与 3D 渲染专家。\n"
            f"我们在研究【{object_category}】这个类别（合法部件包括：{prompt_classes}）。\n"
            f"这是目前的 3D 解剖学知识库内容：\n"
            f"```json\n{current_knowledge_json}\n```\n\n"
            f"我为你提供了 {len(image_paths)} 个新实例的多视角拼图。\n\n"
            f"【核心任务】：\n"
            f"结合新图片展示的特征，对上面的知识库内容进行**局部视觉特征更新**。\n"
            f"**极度重要 1：你绝对不需要重写整个 JSON！你只需要输出你需要【修改或新增】的部件！**\n"
            f"**极度重要 2：大语言模型对三维空间数值感知较弱，因此你【严禁修改任何部件的 z_height_range】！哪怕你觉得图里的东西超出了范围也不准改！你只被允许通过观察图片去丰富补充文字描述（如 spatial_location, typical_3D_shape）或发现新部件。\n\n"
            f"只有发现以下新情况才更新数据：\n"
            f"1. 发现某预定义部件在图片中呈现了知识库未涵盖的新形状或新结构，你只需要输出【纯粹的新增特征描述】。\n"
            f"2. 发现了原记录中没有的合理 `extra_parts`（如特殊的脚踏板）。\n\n"
            f"如果图片中的特征完全符合已有知识库，请直接输出空更新内容：\n"
            "{\n"
            '  "anatomy_knowledge": {\n'
            '    "predefined_parts": {},\n'
            '    "extra_parts": {}\n'
            '  }\n'
            "}\n\n"
            f"如果有需要补充的局部特征，只需包含新增的那一项。例如你发现桌腿有了新的交叉结构：\n"
            "{\n"
            '  "anatomy_knowledge": {\n'
            '    "predefined_parts": {\n'
            '      "leg": {\n'
            '        "typical_3D_shape": "图里出现了一种奇特的X型交叉支撑腿。"\n'
            '      }\n'
            '    },\n'
            '    "extra_parts": {\n'
            '      "footrest": "底部用于踩踏的金属杆"\n'
            '    }\n'
            '  }\n'
            "}\n"
            f"【极其重要】：对于预定义部件的更新，你输出的文字会被后台代码【自动追加】到原有特征之后。因此，绝对不要重复已有文字，也绝对不要带\"在原有基础上补充：\"之类的废话前缀，直接输出你观察到的新特征即可！\n"
            f"请输出**格式绝对合法且能直接解析的** JSON。不要包含多余的废话和 markdown 外部标记。"
        )

        content_list = [{"type": "text", "text": prompt_text}]
        for img_path in image_paths:
            base64_img = encode_image_to_base64(img_path)
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}",
                    "detail": "high"
                }
            })

        messages = [{"role": "user", "content": content_list}]

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                print(f"    [API] 正在呼叫大模型 (尝试 {retry_count+1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2048,
                    timeout=120.0
                )
                response_text = response.choices[0].message.content.strip()
                extracted_json = extract_json_from_response(response_text)

                # 当场尝试解析，如果不合法直接抛异常触发原地重试！
                update_dict = json.loads(extracted_json)

                return extracted_json

            except json.JSONDecodeError:
                print("    ⚠️ 警告：大模型输出 JSON 格式损坏或被截断，立即原地要求重发！")
                retry_count += 1
                time.sleep(3)
            except Exception as e:
                print(f"    ❌ 网络或 API 异常: {e}")
                retry_count += 1
                time.sleep(5)

        print("    ⚠️ 多次重试均无法获得合法 JSON，放弃提取本批次特征。")
        return '{"anatomy_knowledge": {"predefined_parts": {}, "extra_parts": {}}}'

def main():
    # ==== 配置区 (从 config.py 集中读取) ====
    CATEGORY = get_category()
    META_PATH = os.path.join(BASE_DIR, "PartNetE_meta.json")
    IMAGE_DIR = os.path.join(SEM_KNOWLEDGE_IMAGES_DIR, CATEGORY)
    OUTPUT_KNOWLEDGE_DIR = KNOWLEDGE_DIR
    BATCH_SIZE = SEM_BATCH_SIZE
    # ===========================================

    os.makedirs(OUTPUT_KNOWLEDGE_DIR, exist_ok=True)

    if not os.path.exists(META_PATH):
        print(f"❌ 找不到 Meta 文件: {META_PATH}")
        return

    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    prompt_classes = meta_data.get(CATEGORY, [])

    print(f"🚀 开始为类别 {CATEGORY} 构建增量式 MLLM 离线解剖学知识库...")

    generator = OfflineKnowledgeGenerator()

    # 【阶段一：盲猜先验】这里定死了高度
    current_knowledge_json = generator.stage1_text_prior(CATEGORY, prompt_classes)

    try:
        parsed_init = json.loads(current_knowledge_json)
        current_knowledge_json = json.dumps(parsed_init, ensure_ascii=False, indent=2)
        print("  ✅ 阶段一：纯文本解剖学先验已生成，骨架搭建完毕。")
    except Exception as e:
        print(f"  ❌ 阶段一 JSON 格式异常。程序中断。错误信息: {e}")
        return

    # 【阶段二：增量看图纠偏】这里只看图补充描述，绝不改高度
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 找不到图片目录: {IMAGE_DIR}。")
    else:
        # 查找图片，兼容不同命名格式
        all_images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith("_4views.png") or f.endswith("_4views.png")]
        # 兼容新的命名格式
        if not all_images:
             all_images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith("views.png")]
        all_images.sort()
        total_images = len(all_images)

        if total_images == 0:
            print(f"⚠️ 在 {IMAGE_DIR} 中找不到多视图图片。")
        else:
            print(f"\n▶️ [阶段二] 发现 {total_images} 张实例拼图，开始局部差异(Diff)更新学习...")

            for i in range(0, total_images, BATCH_SIZE):
                batch_images = all_images[i:i+BATCH_SIZE]
                print(f"\n  [Progress] 正在处理第 {i+1} 到 {min(i+BATCH_SIZE, total_images)} 个实例 (共 {total_images} 个)...")

                update_json_str = generator.stage2_visual_refinement_batch(
                    object_category=CATEGORY,
                    prompt_classes=prompt_classes,
                    current_knowledge_json=current_knowledge_json,
                    image_paths=batch_images
                )

                # 【Python 端的安全字典合并】
                try:
                    update_dict = json.loads(update_json_str)
                    current_dict = json.loads(current_knowledge_json)

                    new_predefined = update_dict.get("anatomy_knowledge", {}).get("predefined_parts", {})
                    new_extra = update_dict.get("anatomy_knowledge", {}).get("extra_parts", {})

                    has_changes = False

                    # 1. 更新预定义部件的局部特征（采用安全的字符串追加模式）
                    for part_name, part_info in new_predefined.items():
                        if part_name in current_dict["anatomy_knowledge"]["predefined_parts"]:
                            # 🚨 终极安全锁：如果在 update 字典里发现了它手欠改了 z_height_range，直接把它删掉！
                            if "z_height_range" in part_info:
                                del part_info["z_height_range"]

                            if part_info:
                                # 遍历 MLLM 给出的该部件的所有更新字段（如 typical_3D_shape）
                                for key, new_val in part_info.items():
                                    if isinstance(new_val, str):
                                        # 获取该字段原本的描述
                                        old_val = current_dict["anatomy_knowledge"]["predefined_parts"][part_name].get(key, "")

                                        # 清理大模型可能生成的啰嗦前缀
                                        clean_new_val = new_val.replace("在原有基础上，补充：", "").replace("在原有基础上补充：", "").replace("补充：", "").strip()

                                        # 如果这段新描述有实质内容且不在原文本中，则进行字符串安全拼接
                                        if clean_new_val and clean_new_val not in old_val:
                                            # 如果原文本非空，加个分隔符，防止两句话粘在一起
                                            separator = " 此外发现：" if old_val else ""
                                            current_dict["anatomy_knowledge"]["predefined_parts"][part_name][key] = old_val + separator + clean_new_val
                                            has_changes = True

                    # 2. 增加没见过的新额外部件
                    for extra_name, extra_desc in new_extra.items():
                        if extra_name not in current_dict["anatomy_knowledge"].get("extra_parts", {}):
                            if "extra_parts" not in current_dict["anatomy_knowledge"]:
                                current_dict["anatomy_knowledge"]["extra_parts"] = {}
                            current_dict["anatomy_knowledge"]["extra_parts"][extra_name] = extra_desc
                            has_changes = True

                    # 将更新后的字典转回 JSON 字符串，供下一轮使用
                    current_knowledge_json = json.dumps(current_dict, ensure_ascii=False, indent=4)

                    if has_changes:
                        print("    ✅ 知识库发现了新形态特征或新额外部件，已成功安全追加合并！(物理高度限制不受影响)")
                    else:
                        print("    ✅ 本批次实例符合预期，未对知识库进行修改。")

                except Exception as e:
                    print(f"  ⚠️ Python 端合并更新时发生意外: {e}，保留原版知识库继续。")

            print("\n  ✅ 阶段二：所有实例图已阅毕，解剖学知识库差异化合并完成！")

    # 保存最终结果
    output_file = get_sem_knowledge_path()

    try:
        parsed_json = json.loads(current_knowledge_json)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, indent=4, ensure_ascii=False)
        print(f"\n🎉 完美！终极离线知识库已成功保存为极其健壮的 JSON 文件: {output_file}")
    except Exception as e:
        print(f"\n⚠️ 最终保存文件时发生未知错误: {e}")

if __name__ == "__main__":
    main()

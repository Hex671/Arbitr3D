import os
import sys
import json
import time

# 确保能找到 scripts 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from openai import OpenAI
except ImportError:
    print("❌ 请先安装 openai 包: pip install openai")
    sys.exit(1)

# 导入集中配置
from scripts.config import BASE_DIR, KNOWLEDGE_DIR, CONFIG_PATH, get_category, get_sem_knowledge_path, get_sem_knowledge_refined_path


def refine_part_info(client: OpenAI, model_name: str, part_name: str, part_info: dict) -> dict:
    """
    调用大模型对一个部件的描述信息进行提炼

    保留字段：spatial_location, typical_3D_shape, connection_context
    绝对不修改字段：z_height_range
    """

    # 提取需要精炼的文本字段
    spatial_loc = part_info.get("spatial_location", "")
    shape_desc = part_info.get("typical_3D_shape", "")
    connection = part_info.get("connection_context", "")
    z_height = part_info.get("z_height_range", [0.0, 1.0])

    # 如果所有文本字段都太短，无需精炼
    total_text = spatial_loc + shape_desc + connection
    if len(total_text.strip()) < 50:
        return part_info

    # 构建精炼 prompt
    prompt = (
        "你是一位出色的技术文档编辑和 3D 计算机视觉专家。\n"
        f"下面是一段关于【{part_name}】这个 3D 部件的多视角视觉描述。\n"
        "这段描述是在多次增量观察后堆砌而成的，包含了大量重复的修饰语、冗余的抒情表达以及车轱辘话。\n\n"
        "【你的任务】：\n"
        "请对这段描述进行**逻辑重组与专业提炼**。将堆砌的散文改写为精准、客观的工程描述。\n\n"
        "**严格要求**：\n"
        "1. **信息零遗漏**：绝对不可删除任何关于【部件空间位置、几何形状特征、连接关系】的核心物理与视觉线索。\n"
        "2. **去重与客观**：消除重复啰嗦的废话和主观情感词。合并描述同一部件不同特征的零散句子。\n"
        "3. **字数不设限**：只要信息有用，该长就长，不要强行缩写。确保每一句话完整且语意连贯。\n"
        "4. **纯文本输出**：输出必须是一段完整的中文段落，不要使用列表，绝对不要使用 Markdown 格式。\n"
        "5. **绝对禁止**：绝对不能提及或暗示任何关于\"高度范围\"、\"Z 轴区间\"、\"z_height_range\" 等数值信息。你只负责精炼文字描述！\n\n"
        f"【原始堆砌描述】\n"
        f"- 空间位置描述：{spatial_loc}\n"
        f"- 几何形状描述：{shape_desc}\n"
        f"- 连接关系描述：{connection}\n\n"
        "请直接输出精炼后的文字描述，格式如下：\n"
        "【精炼后的空间位置】：xxx\n"
        "【精炼后的几何形状】：xxx\n"
        "【精炼后的连接关系】：xxx"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
                timeout=60.0
            )
            refined_text = response.choices[0].message.content.strip()

            # 解析精炼后的结果
            result = {
                "spatial_location": spatial_loc,
                "typical_3D_shape": shape_desc,
                "connection_context": connection,
                "z_height_range": z_height  # 绝对不修改
            }

            # 解析 LLM 输出
            lines = refined_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("【精炼后的空间位置】："):
                    result["spatial_location"] = line.replace("【精炼后的空间位置】：", "").strip()
                elif line.startswith("【精炼后的几何形状】："):
                    result["typical_3D_shape"] = line.replace("【精炼后的几何形状】：", "").strip()
                elif line.startswith("【精炼后的连接关系】："):
                    result["connection_context"] = line.replace("【精炼后的连接关系】：", "").strip()

            return result

        except Exception as e:
            print(f"      [API Error] 精炼失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(3)

    print("      ⚠️ 多次精炼失败，保留原文本。")
    return part_info


def main():
    CATEGORY = get_category()

    INPUT_JSON = get_sem_knowledge_path()
    OUTPUT_JSON = get_sem_knowledge_refined_path()

    if not os.path.exists(INPUT_JSON):
        print(f"❌ 找不到输入文件: {INPUT_JSON}")
        print(f"   请先运行 gene_sem_knowledge.py 生成知识库。")
        return

    import yaml
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    from utils.runtime_config import resolve_api_key, resolve_base_url, resolve_model_name
    api_key = resolve_api_key(config["mllm"])
    base_url = resolve_base_url(config["mllm"])

    MODEL_NAME = resolve_model_name(config["mllm"])
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    anatomy_knowledge = data.get("anatomy_knowledge", {})
    if not anatomy_knowledge:
        print("❌ JSON 文件中未找到 'anatomy_knowledge' 字段。")
        return

    predefined_parts = anatomy_knowledge.get("predefined_parts", {})
    extra_parts = anatomy_knowledge.get("extra_parts", {})

    print(f"🚀 开始使用 {MODEL_NAME} 对 {CATEGORY} 的语义解剖学知识进行精炼...")
    print(f"   待精炼的预定义部件数量: {len(predefined_parts)}")
    print(f"   额外部件数量: {len(extra_parts)} (额外部件不做文字精炼)")

    refined_count = 0
    for part_name, part_info in predefined_parts.items():
        total_text = (
            part_info.get("spatial_location", "") +
            part_info.get("typical_3D_shape", "") +
            part_info.get("connection_context", "")
        )

        if len(total_text.strip()) >= 50:
            print(f"\n  📝 正在精炼部件: {part_name}")
            print(f"    原始描述长度: {len(total_text)} 字符")

            refined_info = refine_part_info(client, MODEL_NAME, part_name, part_info)
            anatomy_knowledge["predefined_parts"][part_name] = refined_info

            new_total = (
                refined_info.get("spatial_location", "") +
                refined_info.get("typical_3D_shape", "") +
                refined_info.get("connection_context", "")
            )
            print(f"    精炼后长度: {len(new_total)} 字符")
            refined_count += 1
        else:
            print(f"\n  ⏭️ 部件 {part_name} 描述过短，跳过精炼。")

    # 更新 data
    data["anatomy_knowledge"] = anatomy_knowledge

    # 添加精炼元信息
    data["_refinement_meta"] = {
        "refined_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "refined_parts_count": refined_count,
        "total_parts_count": len(predefined_parts),
        "note": "额外部件 (extra_parts) 未做文字精炼，只做结构保留"
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 精炼完成！新文件已保存至: {OUTPUT_JSON}")
    print(f"   精炼了 {refined_count}/{len(predefined_parts)} 个部件的描述")


if __name__ == "__main__":
    main()

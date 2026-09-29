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
from scripts.config import BASE_DIR, KNOWLEDGE_DIR, CONFIG_PATH, get_category, get_view_knowledge_path, get_view_knowledge_refined_path

def refine_text_with_llm(client: OpenAI, model_name: str, raw_text: str) -> str:
    """调用大模型对冗长的描述进行提炼"""

    if not raw_text or len(raw_text.strip()) < 50:
        return raw_text

    prompt = (
        "你是一位出色的技术文档编辑和3D计算机视觉专家。\n"
        "下面是一段关于某个3D物体在特定相机视角下的视觉特征描述。\n"
        "这段描述是在多次增量观察后堆砌而成的，包含了大量重复的修饰语、冗余的抒情表达（如'极其出色地'、'完美展现了'）以及车轱辘话。\n\n"
        "【你的任务】：\n"
        "请对这段描述进行**逻辑重组与专业提炼**。将堆砌的散文改写为精准、客观的工程描述。\n\n"
        "**严格要求**：\n"
        "1. **信息零遗漏**：绝对不可删除任何关于【可见部件、遮挡关系、几何形变、空间深度、相对位置、结构特征】的核心物理与视觉线索。\n"
        "2. **去重与客观**：消除重复啰嗦的废话和主观情感词。合并描述同一部件不同特征的零散句子。\n"
        "3. **字数不设限**：只要信息有用，该长就长，不要强行缩写。确保每一句话完整且语意连贯。\n"
        "4. **纯文本输出**：输出必须是一段完整的中文段落，不要使用列表（如 1. 2. 3. 或 - ），绝对不要使用 Markdown 格式（不要加粗、不要代码块）。只输出提炼后的最终段落。\n\n"
        f"【原始堆砌描述】：\n{raw_text}\n\n"
        "请直接输出重组精炼后的结果："
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,  # 提高截断上限
                timeout=60.0
            )
            refined_text = response.choices[0].message.content.strip()

            refined_text = refined_text.replace('"', '').replace('\"', '').strip()
            return refined_text

        except Exception as e:
            print(f"      [API Error] 提炼失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(3)

    print("      ⚠️ 多次提炼失败，保留原文本。")
    return raw_text

def main():
    CATEGORY = get_category()

    INPUT_JSON = get_view_knowledge_path()
    OUTPUT_JSON = get_view_knowledge_refined_path()

    if not os.path.exists(INPUT_JSON):
        print(f"❌ 找不到输入文件: {INPUT_JSON}")
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

    view_knowledge = data.get("view_knowledge", {})
    if not view_knowledge:
        print("❌ JSON 文件中未找到 'view_knowledge' 字段。")
        return

    print(f"🚀 开始使用 {MODEL_NAME} 对 {CATEGORY} 视角的特征描述进行精炼...")

    for view_id, view_info in view_knowledge.items():
        raw_chars = view_info.get("view_characteristics", "")

        if raw_chars:
            print(f"\n  📝 正在精炼视角: {view_id}")
            print(f"    原始长度: {len(raw_chars)} 字符")

            refined_chars = refine_text_with_llm(client, MODEL_NAME, raw_chars)

            view_info["view_characteristics"] = refined_chars

            print(f"    精炼长度: {len(refined_chars)} 字符")
            print(f"    精炼结果: {refined_chars[:80]}...")
        else:
            print(f"\n  ⏭️ 视角 {view_id} 没有特征描述，跳过。")

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 精炼完成！新文件已安全保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

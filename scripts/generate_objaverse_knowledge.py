"""
PartObjaverse-Tiny 泛化知识库生成脚本（纯文本提示，不依赖渲染图）

用法:
    # 生成所有类别的知识库（默认 min-freq=2，只为出现≥2次的部件生成知识）
    python scripts/generate_objaverse_knowledge.py

    # 指定类别
    python scripts/generate_objaverse_knowledge.py -c "Human-Shape" "Animals"

    # 调整最低出现频次阈值（1=全部部件，3=只保留出现≥3次的部件）
    python scripts/generate_objaverse_knowledge.py --min-freq 1

    # 仅查看各类别部件频次分布（不调用 MLLM）
    python scripts/generate_objaverse_knowledge.py --dry-run

生成文件:
    config/knowledge/{Category}_unified_knowledge.json
    config/knowledge/{Category}_reasonable_adjacencies.json
    config/knowledge/{Category}_spatial_height_order.json
"""

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import re
import asyncio
import argparse
from typing import List, Dict, Any
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "config", "knowledge")
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(PROJECT_ROOT, "config", "config.yaml"))

# ============================================================
# 配置加载
# ============================================================
try:
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    from utils.runtime_config import resolve_api_key, resolve_base_url, resolve_model_name
    BASE_URL = resolve_base_url(config["mllm"])
    MODEL_NAME = resolve_model_name(config["mllm"])

    pot_cfg = config.get('partobjaverse_tiny', {})
    _meta_rel = pot_cfg.get('meta_json', 'PartObjaverse-Tiny_semantic.json')
    _meta_candidate = os.path.join(PROJECT_ROOT, _meta_rel)
    if os.path.exists(_meta_candidate):
        META_PATH = _meta_candidate
    else:
        _meta_candidate2 = os.path.join(pot_cfg.get('base_path', ''), _meta_rel)
        META_PATH = _meta_candidate2 if os.path.exists(_meta_candidate2) else _meta_candidate
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)

try:
    import openai
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

semaphore = asyncio.Semaphore(3)


# ============================================================
# MLLM 连通性检查
# ============================================================

def check_mllm_connectivity() -> bool:
    """启动前检测 MLLM 服务是否可达"""
    print(f"[预检] 测试 MLLM 连通性: {BASE_URL} (model={MODEL_NAME})")
    try:
        client = openai.OpenAI(api_key=resolve_api_key(config["mllm"]), base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply OK"}],
            max_tokens=8,
            timeout=15.0,
        )
        reply = resp.choices[0].message.content.strip()
        print(f"[预检] MLLM 连通成功，回复: {reply}\n")
        return True
    except openai.APIConnectionError as e:
        print(f"\n[错误] 无法连接到 MLLM 服务!")
        print(f"  地址: {BASE_URL}")
        print(f"  错误: {e}")
        print(f"\n  请检查:")
        print(f"    1. vLLM 服务是否已启动")
        print(f"    2. 端口是否正确（当前: {BASE_URL}）")
        print(f"    3. 防火墙/网络是否允许连接")
        return False
    except Exception as e:
        print(f"\n[错误] MLLM 预检失败: {type(e).__name__}: {e}")
        return False


# ============================================================
# 工具函数
# ============================================================

def extract_json(text: str) -> dict:
    """解析 JSON，支持抢救被截断的不完整输出"""
    candidates = []

    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        candidates.append(json_match.group(1))

    start_idx = text.find('{')
    if start_idx != -1:
        end_idx = text.rfind('}')
        if end_idx > start_idx:
            candidates.append(text[start_idx:end_idx + 1])
        candidates.append(text[start_idx:])

    if not candidates:
        candidates.append(text)

    for raw in candidates:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # 抢救截断 JSON：逐层补全闭合括号
    for raw in candidates:
        repaired = _repair_truncated_json(raw)
        if repaired:
            try:
                result = json.loads(repaired)
                if isinstance(result, dict) and result:
                    print(f"  [JSON修复] 从截断输出中抢救到 {len(result)} 个条目")
                    return result
            except json.JSONDecodeError:
                pass

    print(f"  JSON parse failed (含修复尝试)")
    return {}


def _repair_truncated_json(text: str) -> str:
    """尝试修复被截断的 JSON：截掉最后一个不完整的键值对，补全括号"""
    start = text.find('{')
    if start == -1:
        return ""
    s = text[start:]

    # 找到最后一个完整闭合的 value（以 } 结尾的部件条目）
    last_complete = -1
    brace_depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 1:
                last_complete = i

    if last_complete > 0:
        truncated = s[:last_complete + 1]
        # 去掉末尾可能的逗号/空白
        truncated = truncated.rstrip().rstrip(',')
        return truncated + "\n}"

    return ""


def _sync_call_mllm(messages: list) -> str:
    client = openai.OpenAI(api_key=resolve_api_key(config["mllm"]), base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=8192,
        timeout=180.0,
    )
    return response.choices[0].message.content.strip()


async def call_mllm_async(messages: list, max_retries: int = 3) -> str:
    async with semaphore:
        for i in range(max_retries):
            try:
                res = await asyncio.to_thread(_sync_call_mllm, messages)
                return res
            except openai.APIConnectionError as e:
                wait = 5 * (i + 1)
                print(f"  [Connection Error] 第 {i+1}/{max_retries} 次重试，{wait}s 后重连...")
                if i == max_retries - 1:
                    print(f"  [失败] MLLM 连接超时，请检查服务状态: {BASE_URL}")
                    return ""
                await asyncio.sleep(wait)
            except Exception as e:
                wait = 3 * (i + 1)
                if i == max_retries - 1:
                    print(f"  [失败] MLLM 调用错误: {type(e).__name__}: {e}")
                    return ""
                print(f"  [Warning] {type(e).__name__}, {wait}s 后重试...")
                await asyncio.sleep(wait)
        return ""


def load_meta() -> dict:
    if not os.path.exists(META_PATH):
        print(f"[错误] Meta JSON 不存在: {META_PATH}")
        print(f"  请检查 config.yaml 中 partobjaverse_tiny.meta_json 配置")
        sys.exit(1)
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_parts_with_freq(meta: dict, category: str) -> Dict[str, int]:
    """统计每个部件在该类别下出现的实例数"""
    instances = meta.get(category, {})
    counter = Counter()
    for obj_id, parts in instances.items():
        counter.update(parts)
    return dict(counter)


def filter_parts_by_freq(part_freq: Dict[str, int], min_freq: int) -> List[str]:
    """返回出现频次 >= min_freq 的部件列表（排序）"""
    return sorted([p for p, f in part_freq.items() if f >= min_freq])


def print_freq_distribution(category: str, part_freq: Dict[str, int], n_instances: int, min_freq: int):
    """打印部件频次分布和过滤效果"""
    sorted_parts = sorted(part_freq.items(), key=lambda x: -x[1])
    total = len(sorted_parts)
    kept = sum(1 for _, f in sorted_parts if f >= min_freq)

    print(f"\n  {'部件名':30s}  出现次数  占比")
    print(f"  {'-'*55}")
    for p, f in sorted_parts:
        ratio = f / n_instances * 100
        marker = " *" if f < min_freq else ""
        print(f"  {p:30s}  {f:3d}/{n_instances}   ({ratio:4.0f}%){marker}")

    print(f"  {'-'*55}")
    print(f"  总计: {total} 个独立部件")
    print(f"  min_freq={min_freq} 过滤后: {kept} 个部件（标 * 的 {total - kept} 个被排除）")


# ============================================================
# 阶段一：生成各部件 5 维特征（逐批串行，避免并发过高）
# ============================================================

async def gen_part_features_batch(
    category: str,
    parts_batch: List[str],
    all_parts: List[str],
) -> Dict[str, dict]:
    parts_str = ", ".join(parts_batch)
    all_parts_str = ", ".join(all_parts)

    prompt = (
        f"你是一位精通 3D 物体结构与语义分析的专家。\n"
        f"当前物体类别是「{category}」，该类别的 3D 实例通常是各种风格化的 {category} 模型。\n"
        f"该类别下所有可能出现的部件完整列表为：{all_parts_str}\n\n"
        f"请你针对以下部件，基于你对「{category}」类 3D 模型的理解，"
        f"输出每个部件的 5 个通用几何/空间特征。\n"
        f"需要分析的部件：{parts_str}\n\n"
        f"对于每个部件，输出以下 5 个字段：\n"
        f'  "spatial_location": "该部件在整个物体中的典型空间位置（如：位于最底部、连接桌面与地面等）",\n'
        f'  "typical_3D_shape": "该部件的典型 3D 形状（如：薄板状、竖直延伸的圆柱体等）",\n'
        f'  "connection_context": "该部件通常与哪些其他部件连接或接触",\n'
        f'  "relative_size": "该部件相对于整个物体的大小（如：占据极大体积、通常较小等）",\n'
        f'  "z_height_range": [min, max]，归一化到 0.0-1.0 的 Z 轴高度范围，需适度宽松以包容变化\n\n'
        f"请严格输出以下格式的 JSON（不要用 ```json 包裹），所有描述使用中文：\n"
        "{\n"
        f'  "部件名A": {{"spatial_location": "...", "typical_3D_shape": "...", '
        f'"connection_context": "...", "relative_size": "...", "z_height_range": [0.0, 1.0]}},\n'
        f'  "部件名B": {{...}}\n'
        "}\n"
        "注意：JSON 的键必须是原始的英文部件名称。"
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(3):
        res_text = await call_mllm_async(messages)
        data = extract_json(res_text)
        if data and len(data) >= len(parts_batch) * 0.5:
            return data
        if not res_text:
            print(f"  [批次失败] 空回复，跳过该批次")
            return {}
        print(f"  Retry {attempt + 1}/3 for batch ({len(parts_batch)} parts)...")
        await asyncio.sleep(3)

    return {}


async def gen_all_part_features(category: str, all_parts: List[str]) -> dict:
    BATCH_SIZE = 8
    batches = [all_parts[i:i + BATCH_SIZE] for i in range(0, len(all_parts), BATCH_SIZE)]

    print(f"  共 {len(all_parts)} 个部件，分 {len(batches)} 批串行处理")

    merged = {}
    for i, batch in enumerate(batches):
        print(f"  [批次 {i+1}/{len(batches)}] {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}")
        result = await gen_part_features_batch(category, batch, all_parts)
        merged.update(result)
        print(f"    -> 获得 {len(result)} 个部件特征")

    print(f"  成功生成 {len(merged)}/{len(all_parts)} 个部件的特征")
    return merged


# ============================================================
# 阶段二：生成合理邻接关系
# ============================================================

async def gen_reasonable_adjacencies(category: str, parts: List[str]) -> dict:
    BATCH_SIZE = 40
    if len(parts) <= BATCH_SIZE:
        return await _gen_adjacencies_single(category, parts)

    batches = [parts[i:i + BATCH_SIZE] for i in range(0, len(parts), BATCH_SIZE)]
    merged = {}
    for i, batch in enumerate(batches):
        print(f"  [邻接关系 批次 {i+1}/{len(batches)}]")
        result = await _gen_adjacencies_single(category, batch)
        for k, v in result.items():
            if k in merged:
                merged[k] = list(set(merged[k] + v))
            else:
                merged[k] = v
    return merged


async def _gen_adjacencies_single(category: str, parts: List[str]) -> dict:
    print(f"  生成邻接关系 ({len(parts)} 部件)...")
    prompt = (
        f"你是一位精通「{category}」类 3D 物体结构的专家。\n"
        f"以下是该类别可能出现的部件列表：{parts}\n"
        f"请判断这些部件在物理几何和功能上哪些是互相邻接（可以接触或连接）的。\n"
        f"输出 JSON 对象，每个键是部件名，值是与其合理邻接的部件列表。\n"
        f"注意：这些部件不一定同时出现在一个实例上，只需根据常识判断它们如果共存时的邻接关系。\n"
        "请直接输出 JSON，不要输出 markdown 代码块或解释。键值使用英文部件名。"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    for attempt in range(3):
        res_text = await call_mllm_async(messages)
        data = extract_json(res_text)
        if data:
            valid = {}
            for k, v in data.items():
                if k in parts and isinstance(v, list):
                    valid[k] = [p for p in v if p in parts and p != k]
            return valid
        await asyncio.sleep(3)
    return {}


# ============================================================
# 阶段三：生成空间高度顺序
# ============================================================

async def gen_spatial_height_order(category: str, parts: List[str]) -> dict:
    BATCH_SIZE = 40
    if len(parts) <= BATCH_SIZE:
        return await _gen_height_order_single(category, parts)

    batches = [parts[i:i + BATCH_SIZE] for i in range(0, len(parts), BATCH_SIZE)]
    merged = {}
    for i, batch in enumerate(batches):
        print(f"  [高度顺序 批次 {i+1}/{len(batches)}]")
        result = await _gen_height_order_single(category, batch)
        for k, v in result.items():
            if k in merged:
                merged[k] = list(set(merged[k] + v))
            else:
                merged[k] = v
    return merged


async def _gen_height_order_single(category: str, parts: List[str]) -> dict:
    print(f"  生成空间高度顺序 ({len(parts)} 部件)...")
    prompt = (
        f"你是一位精通「{category}」类 3D 物体结构的专家。\n"
        f"以下是该类别可能出现的部件列表：{parts}\n\n"
        f"请根据物理常识判断这些部件之间的竖直方向（高度）上下关系。\n"
        f"对于每个部件，列出它在正常姿态下应该处于其上方的其他部件。\n"
        f"\"A 的列表包含 B\" 表示 A 应该在 B 上方。\n"
        f"只列出有明确高度关系的，不确定或可能重叠的不要列。\n"
        "输出 JSON 对象，键是部件名，值是该部件应位于其上方的部件列表。\n"
        "请直接输出 JSON，不要输出 markdown 代码块。键值使用英文部件名。"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    for attempt in range(3):
        res_text = await call_mllm_async(messages)
        data = extract_json(res_text)
        if data:
            valid = {}
            for k, v in data.items():
                if k in parts and isinstance(v, list):
                    valid[k] = [p for p in v if p in parts and p != k]
            return valid
        await asyncio.sleep(3)
    return {}


# ============================================================
# 主流程
# ============================================================

async def process_category(category: str, all_parts: List[str]):
    print(f"\n{'='*60}")
    print(f"  类别: {category}")
    print(f"  部件数: {len(all_parts)}")
    print(f"  部件: {', '.join(all_parts)}")
    print(f"{'='*60}")

    # 阶段一
    print(f"\n[阶段一] 生成部件 5 维特征...")
    part_features = await gen_all_part_features(category, all_parts)

    unified_knowledge = {category: part_features}
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    out_path = os.path.join(KNOWLEDGE_DIR, f"{category}_unified_knowledge.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unified_knowledge, f, indent=4, ensure_ascii=False)
    print(f"  -> 保存至 {out_path}")

    # 阶段二
    print(f"\n[阶段二] 生成合理邻接关系...")
    adjacencies = await gen_reasonable_adjacencies(category, all_parts)
    adj_path = os.path.join(KNOWLEDGE_DIR, f"{category}_reasonable_adjacencies.json")
    with open(adj_path, "w", encoding="utf-8") as f:
        json.dump(adjacencies, f, indent=4, ensure_ascii=False)
    print(f"  -> 保存至 {adj_path} ({len(adjacencies)} 条)")

    # 阶段三
    print(f"\n[阶段三] 生成空间高度顺序...")
    height_order = await gen_spatial_height_order(category, all_parts)
    height_path = os.path.join(KNOWLEDGE_DIR, f"{category}_spatial_height_order.json")
    with open(height_path, "w", encoding="utf-8") as f:
        json.dump(height_order, f, indent=4, ensure_ascii=False)
    print(f"  -> 保存至 {height_path} ({len(height_order)} 条)")

    print(f"\n[完成] {category} 知识库生成完毕")


async def main_async(categories: List[str] = None, min_freq: int = 2, dry_run: bool = False):
    meta = load_meta()
    print(f"[Info] Meta JSON: {META_PATH}")

    if categories is None:
        categories = list(meta.keys())

    print(f"\nPartObjaverse-Tiny 泛化知识库生成")
    print(f"目标类别: {categories}")
    print(f"最低频次阈值: min_freq={min_freq}")
    print(f"MLLM: {MODEL_NAME} @ {BASE_URL}\n")

    # 整理各类别部件频次分布
    category_parts = {}
    print(f"{'='*60}")
    print(f"  各类别部件频次统计 (min_freq={min_freq})")
    print(f"{'='*60}")

    for cat in categories:
        if cat not in meta:
            print(f"\n  [Warning] 类别 '{cat}' 不在 meta JSON 中，跳过")
            continue
        n_instances = len(meta[cat])
        part_freq = collect_parts_with_freq(meta, cat)
        filtered = filter_parts_by_freq(part_freq, min_freq)
        total = len(part_freq)

        print(f"\n  {cat}: {n_instances} 个实例, {total} 个独立部件 -> 过滤后 {len(filtered)} 个 (min_freq={min_freq})")

        # 按频次分层显示
        sorted_parts = sorted(part_freq.items(), key=lambda x: -x[1])
        high = [(p, f) for p, f in sorted_parts if f / n_instances >= 0.3]
        mid = [(p, f) for p, f in sorted_parts if 0.3 > f / n_instances and f >= min_freq]
        low = [(p, f) for p, f in sorted_parts if f < min_freq]

        if high:
            print(f"    核心部件 (≥30%): {', '.join(f'{p}({f})' for p, f in high)}")
        if mid:
            print(f"    中频部件 (<30%, ≥{min_freq}次): {', '.join(f'{p}({f})' for p, f in mid)}")
        if low:
            print(f"    排除部件 (<{min_freq}次): {len(low)} 个")

        category_parts[cat] = filtered

    if dry_run:
        print(f"\n{'='*60}")
        print("  [Dry Run] 仅展示统计，不调用 MLLM")
        print(f"{'='*60}")
        return

    # 连通性检查
    if not check_mllm_connectivity():
        print("\n请先确保 MLLM 服务可用后再运行。")
        sys.exit(1)

    # 逐类别处理
    for cat, parts in category_parts.items():
        if not parts:
            print(f"\n[跳过] {cat}: 过滤后无部件")
            continue
        await process_category(cat, parts)

    print(f"\n{'='*60}")
    print("全部完成！")
    print(f"知识库目录: {KNOWLEDGE_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PartObjaverse-Tiny knowledge base (text-only, no rendering)")
    parser.add_argument("--category", "-c", nargs="*", default=None,
                        help="指定类别（默认全部）")
    parser.add_argument("--min-freq", type=int, default=2,
                        help="部件最低出现频次阈值（默认 2，即只为出现≥2次的部件生成知识）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅展示各类别部件频次分布，不调用 MLLM")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU 编号")
    args = parser.parse_args()

    asyncio.run(main_async(args.category, args.min_freq, args.dry_run))


if __name__ == "__main__":
    main()

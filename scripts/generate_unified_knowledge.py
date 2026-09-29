import os
import sys
import subprocess as _sp

# ============================================================
# GPU 自动选择（必须在任何 CUDA / torch import 之前执行）
# ============================================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 确保能找到 Arbitr3D 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _get_physical_gpu_map() -> dict:
    try:
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        mapping = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx = int(parts[0])
            free_gb = float(parts[1]) / 1024.0
            total_gb = float(parts[2]) / 1024.0
            mapping[idx] = (free_gb, total_gb)
        return mapping
    except Exception:
        return {}

def _auto_best_physical(physical_map: dict) -> int:
    import yaml
    threshold = 4.0
    _cfg_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "config", "config.yaml")
    try:
        with open(_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        threshold = float(cfg.get("gpu", {}).get("min_free_gb", 4.0))
    except Exception:
        pass
    if not physical_map:
        raise RuntimeError("No GPU available.")
    candidates = {idx: free for idx, (free, _) in physical_map.items() if free >= threshold}
    if not candidates:
        raise RuntimeError(f"No GPU with >= {threshold:.1f} GB free memory.")
    return max(candidates, key=lambda i: candidates[i])

raw_physical = os.environ.get("CUDA_VISIBLE_DEVICES", "auto").strip()
if raw_physical.lower() in ("", "auto"):
    _physical_map = _get_physical_gpu_map()
    _selected_gpu = _auto_best_physical(_physical_map)
    print(f"[generate_unified_knowledge] 自动选择 GPU {_selected_gpu}，"
          f"显存剩余 ~{_physical_map[_selected_gpu][0]:.1f} GB")
else:
    try:
        _selected_gpu = int(raw_physical)
    except ValueError:
        _physical_map = _get_physical_gpu_map()
        _selected_gpu = _auto_best_physical(_physical_map)
    print(f"[generate_unified_knowledge] 使用 GPU {_selected_gpu}")

import torch
torch.cuda.set_device(_selected_gpu)
# ============================================================

import json
import base64
import time
import re
import math
import numpy as np
import cv2
import asyncio
import argparse
from typing import List, Dict, Any

from scripts.config import (
    DATASET_BASE_PATH, BASE_DIR, get_category, KNOWLEDGE_DIR, CONFIG_PATH,
    RENDER_RESOLUTION, RENDER_RADIUS, NUM_SEM_VIEWS, SEM_VIEWS, SEM_VIEW_LABELS,
    NUM_VIEWS, get_views_info, RAW_VIEWS
)
from data_prep.dataloader import PointCloudDataset
from module_render.renderer import MultiViewRenderer
from module_render.camera_poses import generate_cameras_from_views, generate_sphere_cameras

try:
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    from utils.runtime_config import resolve_api_key, resolve_base_url, resolve_model_name
    BASE_URL = resolve_base_url(config["mllm"])
    MODEL_NAME = resolve_model_name(config["mllm"])
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)

try:
    import openai
except ImportError:
    print("Please install openai package: pip install openai")
    sys.exit(1)

# 限制并发数量，防止触发 API Rate Limit (429)
semaphore = asyncio.Semaphore(10)

def get_partnete_parts(category: str) -> List[str]:
    meta_path = os.path.join(BASE_DIR, "PartNetE_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get(category, [])

def create_grid_collage(images: List[np.ndarray], cols: int) -> np.ndarray:
    if not images:
        return np.zeros((RENDER_RESOLUTION[1], RENDER_RESOLUTION[0], 3), dtype=np.uint8)

    rows = math.ceil(len(images) / cols)
    h, w, c = images[0].shape

    # 补齐图像数量以适应网格
    padded_images = images.copy()
    while len(padded_images) < rows * cols:
        padded_images.append(np.zeros((h, w, c), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_imgs = padded_images[r*cols : (r+1)*cols]
        grid_rows.append(np.hstack(row_imgs))

    return np.vstack(grid_rows)

def encode_image_to_base64(img_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode('.png', img_bgr)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode('utf-8')

def extract_json(text: str) -> dict:
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx+1])

        return json.loads(text)
    except Exception as e:
        print(f"Failed to parse JSON: {e}\nRaw text: {text}")
        return {}

def _sanitize_name(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text))
    return text.strip('._') or 'unnamed'

def _prepare_image_dump_dirs(category: str) -> Dict[str, str]:
    root = os.path.join(
        BASE_DIR,
        "unified_knowledge_generation_images",
        _sanitize_name(category),
        time.strftime("%Y%m%d_%H%M%S")
    )
    semantic_views_dir = os.path.join(root, "semantic_views")
    semantic_collages_dir = os.path.join(root, "semantic_collages")
    raw_views_dir = os.path.join(root, "raw_views")
    view_collages_dir = os.path.join(root, "view_collages")
    for path in [root, semantic_views_dir, semantic_collages_dir, raw_views_dir, view_collages_dir]:
        os.makedirs(path, exist_ok=True)
    return {
        "root": root,
        "semantic_views": semantic_views_dir,
        "semantic_collages": semantic_collages_dir,
        "raw_views": raw_views_dir,
        "view_collages": view_collages_dir,
    }

def _save_image(img_path: str, image: np.ndarray) -> str:
    if not cv2.imwrite(img_path, image):
        raise ValueError(f"Failed to save image to {img_path}")
    return img_path

def _sync_call_mllm(messages: list) -> str:
    client = openai.OpenAI(api_key=resolve_api_key(config["mllm"]), base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
        timeout=120.0
    )
    return response.choices[0].message.content.strip()

async def call_mllm_async(messages: list, max_retries: int = 3) -> str:
    async with semaphore:
        for i in range(max_retries):
            try:
                # Use to_thread to run the synchronous OpenAI call without blocking the event loop
                res = await asyncio.to_thread(_sync_call_mllm, messages)
                return res
            except Exception as e:
                if i == max_retries - 1:
                    print(f"MLLM Call Failed after {max_retries} retries: {e}")
                    return ""
                await asyncio.sleep(2 * (i + 1))
        return ""

async def gen_general_features(part: str, category: str, sem_collages: List[np.ndarray]) -> tuple:
    print(f"[Start] 开始生成部件 {part} 的 5 个全局特征...")
    prompt = (
        f"你是一位 3D 几何与语义分析专家。我将向你展示 {len(sem_collages)} 个属于类别 '{category}' 的不同 3D 实例。\n"
        f"每张图片是一个实例在 4 个经典视角下的语义分割拼接图。\n"
        f"你的任务是分析在这些不同实例中，部件 '{part}' 的通用几何特征和规律。\n\n"
        f"请严格按照以下 JSON 格式输出，所有描述必须使用**中文**：\n"
        "{\n"
        f'  "spatial_location": "描述部件 {part} 在整个物体中的典型空间位置（如：位于最底部、连接桌面与地面等）",\n'
        f'  "typical_3D_shape": "描述部件 {part} 的典型 3D 形状（如：薄板状、竖直延伸的圆柱体、稍微弯曲的面等）",\n'
        f'  "connection_context": "描述部件 {part} 通常与哪些其他部件连接或接触",\n'
        f'  "relative_size": "描述部件 {part} 相对于整个物体的相对大小（如：占据极大体积、通常较小等）",\n'
        f'  "z_height_range": "[最低高度, 最高高度] (例如底座通常是 [0.0, 0.4])。请给出一个合理的归一化 Z 轴高度范围 (0.0 到 1.0)，需要稍微宽松一点，以包容不同实例之间的正常变化。"\n'
        "}\n"
        "只输出上述 JSON，不要包裹在 ```json 代码块中，不要输出任何其他文本。"
    )

    content = [{"type": "text", "text": prompt}]
    for idx, img in enumerate(sem_collages):
        content.append({"type": "text", "text": f"实例 {idx+1} (4个语义视角):"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(img)}"}
        })

    messages = [{"role": "user", "content": content}]

    # 增加针对 JSON 解析失败的重试机制
    max_json_retries = 3
    data = {}
    for attempt in range(max_json_retries):
        res_text = await call_mllm_async(messages)
        data = extract_json(res_text)
        if data:  # 解析成功，字典不为空
            break
        print(f"⚠️ [警告] 部件 {part} 的全局特征 JSON 解析失败，尝试重新请求 ({attempt+1}/{max_json_retries})...")
        await asyncio.sleep(2)

    print(f"[Done] 部件 {part} 的全局特征生成完毕。")
    return part, data

async def gen_view_appearance(part: str, view_id: str, category: str, view_collage: np.ndarray) -> tuple:
    print(f"[Start] 开始生成部件 {part} 在 {view_id} 视角下的外观特征...")
    prompt = (
        f"你是一位 3D 视觉分析专家。图片展示了最多 8 个属于 '{category}' 类别的不同实例在同一个视角（{view_id}）下的渲染图拼接。\n"
        f"请描述在 {view_id} 这个特定视角下，部件 '{part}' 呈现出的典型视觉外观和可见性。\n"
        f"它清晰可见吗？是否被遮挡？它在 2D 画面中投射出什么形状？请保持描述简明扼要（1-3 句话）。\n"
        f"请直接输出这段描述文字，必须使用**中文**。"
    )

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(view_collage)}"}
        }
    ]
    messages = [{"role": "user", "content": content}]

    res_text = await call_mllm_async(messages)
    print(f"[Done] 部件 {part} 在 {view_id} 视角下的外观特征生成完毕。")
    return part, view_id, res_text

async def gen_reasonable_adjacencies(category: str, parts: List[str]) -> dict:
    """离线生成部件间合理邻接关系，供 pipeline 推理时直接加载。"""
    print(f"[Start] 生成 {category} 的部件合理邻接关系...")
    prompt = (
        f"你是一位精通 {category}（或同类物体）3D 结构和制造逻辑的专家。\n"
        f"该物体预定义的部件列表如下：{parts}。\n"
        f"请判断这些部件在物理几何和功能上，哪些是『互相邻接』（即可以接触或连接）的。\n"
        f"输出一个 JSON 对象，其中每个键是一个部件名称，值是一个列表，包含与其合理邻接的所有部件名称。列表里的所有部件必须出现在上述预定义列表中。\n"
        "示例：\n"
        "{\n"
        '  "seat": ["back", "leg", "arm"],\n'
        '  "back": ["seat", "arm"]\n'
        "}\n"
        "请不要输出 markdown 代码块或解释说明，直接输出原生的 JSON 文本。注意：JSON 中的键值必须保持原始的英文部件名称（如 'seat', 'leg'），但你可以使用中文进行思考。"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    max_retries = 3
    for attempt in range(max_retries):
        res_text = await call_mllm_async(messages)
        data = extract_json(res_text)
        if data:
            print(f"[Done] 合理邻接关系生成完毕，包含 {len(data)} 个部件。")
            return data
        print(f"⚠️ JSON 解析失败，重试 ({attempt+1}/{max_retries})...")
        await asyncio.sleep(2)

    print("❌ 合理邻接关系生成失败，返回空字典。")
    return {}


async def gen_spatial_height_order(category: str, parts: List[str]) -> dict:
    """离线生成部件间空间高度顺序规则，供拓扑规则4（空间倒置检测）使用。"""
    print(f"[Start] 生成 {category} 的部件空间高度顺序规则...")
    prompt = (
        f"你是一位精通 {category}（或同类物体）3D 结构的专家。\n"
        f"该物体预定义的部件列表如下：{parts}。\n\n"
        f"请根据物理常识和该类物体的典型结构，判断这些部件之间的**竖直方向（高度）上下关系**。\n"
        f"具体来说，对于每个部件，列出它在正常摆放姿态下**应该处于其上方**的其他部件。\n\n"
        f"规则说明：\n"
        f"- 只列出有明确高度上下关系的部件对，如果两个部件的高度关系不确定或可能重叠，不要列入。\n"
        f"- \"A 的 expected_above 包含 B\" 表示 A 在正常情况下应该在 B 的上方。\n"
        f"- 例如对于椅子：back（靠背）应在 seat（座面）和 leg（腿）上方，seat 应在 leg 上方。\n\n"
        f"输出一个 JSON 对象，格式如下：\n"
        "{\n"
        '  "部件A": ["部件X", "部件Y"],\n'
        '  "部件B": ["部件Z"]\n'
        "}\n"
        f"其中键是部件名，值是该部件应该位于其**上方**的部件列表。\n"
        f"如果某个部件没有明确的「应在其上方」的关系，可以不列入。\n"
        "请不要输出 markdown 代码块或解释说明，直接输出原生的 JSON 文本。键值必须使用英文部件名称。"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    max_retries = 3
    for attempt in range(max_retries):
        res_text = await call_mllm_async(messages)
        data = extract_json(res_text)
        if data:
            # 校验：确保所有键和值都在 parts 列表中
            valid_data = {}
            for k, v in data.items():
                if k in parts and isinstance(v, list):
                    valid_data[k] = [p for p in v if p in parts and p != k]
            print(f"[Done] 空间高度顺序规则生成完毕，包含 {len(valid_data)} 个部件的规则。")
            return valid_data
        print(f"⚠️ JSON 解析失败，重试 ({attempt+1}/{max_retries})...")
        await asyncio.sleep(2)

    print("❌ 空间高度顺序规则生成失败，返回空字典。")
    return {}


async def main_async(category: str):
    parts = get_partnete_parts(category)
    if not parts:
        print(f"❌ 在 PartNetE_meta.json 中找不到类别 {category} 的部件信息！")
        return

    print(f"🔍 当前类别: {category}")
    print(f"🔍 目标部件: {parts}")

    # 1. 加载点云
    category_dir = os.path.join(DATASET_BASE_PATH, category)
    if not os.path.exists(category_dir):
        print(f"❌ 找不到数据集目录: {category_dir}")
        return

    object_ids = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
    object_ids.sort()

    # 取前 8 个实例
    if len(object_ids) > 8:
        object_ids = object_ids[:8]
    print(f"📸 使用 {len(object_ids)} 个实例进行知识提取: {object_ids}")

    image_dump_dirs = _prepare_image_dump_dirs(category)
    print(f"🖼️ 图片保存目录: {image_dump_dirs['root']}")
    image_manifest: Dict[str, Any] = {
        "category": category,
        "dataset_base_path": DATASET_BASE_PATH,
        "selected_object_ids": object_ids,
        "render_resolution": list(RENDER_RESOLUTION),
        "semantic_views": [
            {
                "view_index": sem_idx,
                "view_id": f"View_{sem_idx:02d}",
                "label": SEM_VIEW_LABELS[sem_idx],
                "elev": float(SEM_VIEWS[sem_idx][0]),
                "azim": float(SEM_VIEWS[sem_idx][1]),
            }
            for sem_idx in range(len(SEM_VIEWS))
        ],
        "raw_views": [],
        "instances": {},
        "mllm_inputs": {
            "general_features_semantic_collages": [],
            "view_appearance_collages": {}
        }
    }

    dataloader = PointCloudDataset(DATASET_BASE_PATH)
    renderer = MultiViewRenderer(resolution=RENDER_RESOLUTION)

    # 获取语义四视角的相机姿态
    sem_cam_pos, sem_cam_rot = generate_cameras_from_views(
        SEM_VIEWS,
        radius=RENDER_RADIUS,
    )

    # 获取 10 个外观视角的相机姿态
    raw_cam_pos, raw_cam_rot = generate_sphere_cameras(K=NUM_VIEWS, radius=RENDER_RADIUS)
    views_info = get_views_info(NUM_VIEWS)
    image_manifest["raw_views"] = [
        {
            "view_index": v_idx,
            "view_id": views_info[v_idx]["view_id"],
            "elev": float(views_info[v_idx]["elev"]),
            "azim": float(views_info[v_idx]["azim"]),
        }
        for v_idx in range(NUM_VIEWS)
    ]

    sem_collages = [] # 每个元素是一个实例的 4 视角拼接图
    raw_view_images = {v_idx: [] for v_idx in range(NUM_VIEWS)} # 每个视角下包含 8 个实例的图片列表

    print("\n▶️ [阶段一] 渲染所有实例的视角...")
    for idx, obj_id in enumerate(object_ids):
        print(f"  渲染实例 {idx+1}/{len(object_ids)}: {obj_id}")
        pc_path = f"{category}/{obj_id}/pc.ply"
        pcd = dataloader.load_point_cloud(pc_path)
        norm_pcd = dataloader.normalize_pc(pcd)
        pc_xyz = np.asarray(norm_pcd.points)
        pc_colors = dataloader.extract_color(norm_pcd)
        sem_instance_dir = os.path.join(image_dump_dirs["semantic_views"], _sanitize_name(obj_id))
        raw_instance_dir = os.path.join(image_dump_dirs["raw_views"], _sanitize_name(obj_id))
        os.makedirs(sem_instance_dir, exist_ok=True)
        os.makedirs(raw_instance_dir, exist_ok=True)
        image_manifest["instances"][obj_id] = {
            "semantic_views": [],
            "semantic_collage": "",
            "raw_views": []
        }

        # 渲染 4 个语义视角
        sem_out = renderer.render(pc_xyz, pc_colors, sem_cam_pos, sem_cam_rot)
        sem_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in sem_out.images]
        for sem_idx, img_bgr in enumerate(sem_imgs):
            sem_path = os.path.join(
                sem_instance_dir,
                f"view_{sem_idx:02d}_e{float(SEM_VIEWS[sem_idx][0]):g}_a{float(SEM_VIEWS[sem_idx][1]):g}.png"
            )
            _save_image(sem_path, img_bgr)
            image_manifest["instances"][obj_id]["semantic_views"].append({
                "view_index": sem_idx,
                "view_id": f"View_{sem_idx:02d}",
                "label": SEM_VIEW_LABELS[sem_idx],
                "elev": float(SEM_VIEWS[sem_idx][0]),
                "azim": float(SEM_VIEWS[sem_idx][1]),
                "path": os.path.relpath(sem_path, image_dump_dirs["root"])
            })
        sem_collage = create_grid_collage(sem_imgs, cols=2) # 2x2 拼接
        sem_collages.append(sem_collage)
        sem_collage_path = os.path.join(
            image_dump_dirs["semantic_collages"],
            f"{idx+1:02d}_{_sanitize_name(obj_id)}_semantic_collage.png"
        )
        _save_image(sem_collage_path, sem_collage)
        image_manifest["instances"][obj_id]["semantic_collage"] = os.path.relpath(sem_collage_path, image_dump_dirs["root"])
        image_manifest["mllm_inputs"]["general_features_semantic_collages"].append({
            "object_id": obj_id,
            "path": os.path.relpath(sem_collage_path, image_dump_dirs["root"])
        })

        # 渲染 10 个特定视角
        raw_out = renderer.render(pc_xyz, pc_colors, raw_cam_pos, raw_cam_rot)
        for v_idx in range(NUM_VIEWS):
            view_id = views_info[v_idx]['view_id']
            img_bgr = cv2.cvtColor(raw_out.images[v_idx], cv2.COLOR_RGB2BGR)
            cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 35), (0, 0, 0), -1)
            cv2.putText(img_bgr, f"ID: {obj_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            raw_view_images[v_idx].append(img_bgr)
            raw_img_path = os.path.join(
                raw_instance_dir,
                f"{view_id}_e{float(views_info[v_idx]['elev']):g}_a{float(views_info[v_idx]['azim']):g}.png"
            )
            _save_image(raw_img_path, img_bgr)
            image_manifest["instances"][obj_id]["raw_views"].append({
                "view_index": v_idx,
                "view_id": view_id,
                "elev": float(views_info[v_idx]['elev']),
                "azim": float(views_info[v_idx]['azim']),
                "path": os.path.relpath(raw_img_path, image_dump_dirs["root"])
            })

    # 将 10 个视角中的每个视角对应的 8 个实例，拼接成 1 张大图 (4x2 拼接)
    view_collages = {}
    for v_idx in range(NUM_VIEWS):
        view_id = views_info[v_idx]['view_id']
        view_collages[view_id] = create_grid_collage(raw_view_images[v_idx], cols=4)
        view_collage_path = os.path.join(image_dump_dirs["view_collages"], f"{view_id}.png")
        _save_image(view_collage_path, view_collages[view_id])
        image_manifest["mllm_inputs"]["view_appearance_collages"][view_id] = os.path.relpath(view_collage_path, image_dump_dirs["root"])

    manifest_path = os.path.join(image_dump_dirs["root"], "image_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(image_manifest, f, indent=4, ensure_ascii=False)
    print(f"📝 图片清单已保存至: {manifest_path}")

    print("\n▶️ [阶段二] 启动并行 MLLM 知识提取任务...")
    tasks = []

    # 1. 为每个部件创建获取 5 个全局特征的任务
    for part in parts:
        tasks.append(gen_general_features(part, category, sem_collages))

    # 2. 为每个部件的每个视角创建获取外观的任务
    view_tasks = []
    for part in parts:
        for v_idx in range(NUM_VIEWS):
            view_id = views_info[v_idx]['view_id']
            view_tasks.append(gen_view_appearance(part, view_id, category, view_collages[view_id]))

    # 执行全局特征提取
    general_results = await asyncio.gather(*tasks)

    # 执行视角外观提取
    view_results = await asyncio.gather(*view_tasks)

    print("\n▶️ [阶段三] 整合并保存到统一知识库...")
    unified_knowledge = {}
    for part, data in general_results:
        unified_knowledge[part] = {
            "spatial_location": data.get("spatial_location", ""),
            "typical_3D_shape": data.get("typical_3D_shape", ""),
            "connection_context": data.get("connection_context", ""),
            "relative_size": data.get("relative_size", ""),
            "z_height_range": data.get("z_height_range", ""),
            "view_appearance": {}
        }

    for part, view_id, text in view_results:
        if part in unified_knowledge:
            unified_knowledge[part]["view_appearance"][view_id] = text

    # 最终按照要求的格式组装 JSON
    final_output = {category: unified_knowledge}

    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    out_path = os.path.join(KNOWLEDGE_DIR, f"{category}_unified_knowledge.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"✅ 成功生成统一知识库到: {out_path}")

    # ============================================================
    # [阶段四] 生成部件合理邻接关系 (Reasonable Adjacencies)
    # ============================================================
    print("\n▶️ [阶段四] 生成部件合理邻接关系...")
    reasonable_adjacencies = await gen_reasonable_adjacencies(category, parts)

    adj_out_path = os.path.join(KNOWLEDGE_DIR, f"{category}_reasonable_adjacencies.json")
    with open(adj_out_path, "w", encoding="utf-8") as f:
        json.dump(reasonable_adjacencies, f, indent=4, ensure_ascii=False)
    print(f"✅ 成功生成合理邻接关系到: {adj_out_path}")

    # ============================================================
    # [阶段五] 生成部件空间高度顺序规则 (Spatial Height Order)
    # ============================================================
    print("\n▶️ [阶段五] 生成部件空间高度顺序规则...")
    spatial_height_order = await gen_spatial_height_order(category, parts)

    height_order_path = os.path.join(KNOWLEDGE_DIR, f"{category}_spatial_height_order.json")
    with open(height_order_path, "w", encoding="utf-8") as f:
        json.dump(spatial_height_order, f, indent=4, ensure_ascii=False)
    print(f"✅ 成功生成空间高度顺序规则到: {height_order_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate unified knowledge base using MLLM")
    parser.add_argument(
        "--category", "-c",
        type=str,
        required=True,
        help="The category of the object to process (e.g., Chair, Table, etc.)"
    )
    args = parser.parse_args()

    asyncio.run(main_async(args.category))

if __name__ == "__main__":
    main()

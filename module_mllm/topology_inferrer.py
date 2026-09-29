"""
RES 在线拓扑推断器 —— 看 4 视角 collage 推断 hit_semantics 之间的：
1. height_order：常规摆放下的高低关系（用于注入 TopologyGraphBuilder.spatial_height_order）
2. adjacency：直接接触/连接的部件对（用于 reasonable_adjacencies）

输出格式严格匹配现有 topology_builder.py 期望：
    spatial_height_order = {"head": ["body", "leg"], "body": ["leg"], ...}
    reasonable_adjacencies = {"head": ["body"], "body": ["head", "leg"], ...}

注：本模块独立运行，不修改任何现有文件。
"""

import base64
import json
import re
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.res_prompts import TOPOLOGY_INFER_PROMPT_TMPL
from module_mllm.k_u_distractor_discoverer import make_2x2_collage


class TopologyInferrer:
    """4 视角 → 邻接 + 高度顺序。"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o",
    ):
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
            print("[TopologyInferrer] OpenAI package not installed; will return empty topology.")
            self.client = None

    # ------------------------------------------------------------------
    def infer(
        self,
        view_images_rgb: List[np.ndarray],
        hit_semantics: List[str],
        elevation: float = 20.0,
        max_retries: int = 3,
        timeout: float = 90.0,
    ) -> Dict:
        """
        Args:
            view_images_rgb: 4 张视角 RGB 图（顺序：az=0/90/180/270）
            hit_semantics: 第一轮 RES 实际命中的语义集合（含 target，不含 unknown）

        Returns:
            {
                "spatial_height_order": {sem_a: [sem_lower, ...], ...},
                "reasonable_adjacencies": {sem_a: [neighbor, ...], ...},
                "raw": <原始 MLLM JSON>
            }
        """
        empty_result = {
            "spatial_height_order": {},
            "reasonable_adjacencies": {},
            "raw": {},
        }
        if self.client is None or not view_images_rgb or len(hit_semantics) < 2:
            return empty_result

        collage = make_2x2_collage(view_images_rgb)
        img_bgr = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", img_bgr)
        b64_collage = base64.b64encode(buf).decode("utf-8")

        # 用 hit_semantics 第一个作为 target（仅展示用，prompt 中不做特殊化）
        sem_lines = "\n".join(f"  - {s}" for s in hit_semantics)
        prompt = TOPOLOGY_INFER_PROMPT_TMPL.format(
            elevation=int(round(elevation)),
            hit_semantics_lines=sem_lines,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64_collage}"
                            }},
                        ],
                    }],
                    max_tokens=2048,
                    temperature=0.1,
                    timeout=timeout,
                )
                response_text = (response.choices[0].message.content or "").strip()
                parsed = self._parse_response(response_text)
                if parsed is not None:
                    return self._sanitize(parsed, hit_semantics)
            except Exception as e:
                print(f"[TopologyInferrer] attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt + 1 < max_retries:
                    time.sleep(2)

        print("[TopologyInferrer] All retries failed; returning empty topology.")
        return empty_result

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(text: str) -> Optional[Dict]:
        if not text:
            return None
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _sanitize(parsed: Dict, hit_semantics: List[str]) -> Dict:
        """
        清洗：
        - height_order 的 key/value 必须都在 hit_semantics 内
        - adjacency 的 pair 必须都在 hit_semantics 内
        - 把 adjacency 列表转为 {sem: [neighbor, ...]} 形式（与 reasonable_adjacencies 兼容）
        """
        valid_set = set(hit_semantics)

        height_order_raw = parsed.get("height_order", {})
        spatial_height_order: Dict[str, List[str]] = {}
        if isinstance(height_order_raw, dict):
            for k, v in height_order_raw.items():
                if k not in valid_set:
                    continue
                if not isinstance(v, list):
                    continue
                lowers = [x for x in v if isinstance(x, str) and x in valid_set and x != k]
                if lowers:
                    spatial_height_order[k] = list(dict.fromkeys(lowers))  # 去重保序

        adjacency_raw = parsed.get("adjacency", [])
        adjacency_map: Dict[str, List[str]] = {sem: [] for sem in hit_semantics}
        if isinstance(adjacency_raw, list):
            for pair in adjacency_raw:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                a, b = str(pair[0]).strip(), str(pair[1]).strip()
                if a not in valid_set or b not in valid_set or a == b:
                    continue
                if b not in adjacency_map[a]:
                    adjacency_map[a].append(b)
                if a not in adjacency_map[b]:
                    adjacency_map[b].append(a)
        # 移除空列表项以保持简洁
        adjacency_map = {k: v for k, v in adjacency_map.items() if v}

        return {
            "spatial_height_order": spatial_height_order,
            "reasonable_adjacencies": adjacency_map,
            "raw": parsed,
        }

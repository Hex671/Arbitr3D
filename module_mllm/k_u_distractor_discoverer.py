"""
K_u-Dynamic Track B —— 从 4 视角渲染图发现该 mesh 上**除目标外**的可能干扰部件，
为每个干扰部件给出 short_label + 简要特征。

输入：
- 4 张固定视角的 RGB 图（np.ndarray，H×W×3，uint8）
- 目标部件 K_u dict（用于让 MLLM 知道什么是 target，避免把 target 列回 distractor）

输出（与 K_u 风格对齐）：
    [
        {"short_label": "frog body", "brief_features": "圆鼓鼓的主体，连接四肢"},
        ...
    ]

注：本模块独立运行，不修改任何现有文件。
"""

import base64
import io
import json
import re
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.res_prompts import DISTRACTOR_DISCOVERY_PROMPT_TMPL


def make_2x2_collage(images: List[np.ndarray]) -> np.ndarray:
    """把 4 张同尺寸 RGB 图（H×W×3 uint8）拼成 2×2 collage。"""
    if not images:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    if len(images) < 4:
        # 不足 4 张时用黑色补全
        h, w = images[0].shape[:2]
        while len(images) < 4:
            images.append(np.zeros((h, w, 3), dtype=np.uint8))
    h, w = images[0].shape[:2]
    # 统一尺寸
    norm_imgs = []
    for img in images[:4]:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        norm_imgs.append(img)
    top = np.hstack([norm_imgs[0], norm_imgs[1]])
    bot = np.hstack([norm_imgs[2], norm_imgs[3]])
    return np.vstack([top, bot])


def _encode_rgb_png_b64(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


class DistractorDiscoverer:
    """4 视角 → distractor 列表。"""

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
            print("[DistractorDiscoverer] OpenAI package not installed; will return empty list.")
            self.client = None

    # ------------------------------------------------------------------
    def discover(
        self,
        view_images_rgb: List[np.ndarray],
        target_ku: Dict,
        elevation: float = 20.0,
        max_distractors: int = 8,
        max_retries: int = 3,
        timeout: float = 90.0,
    ) -> List[Dict[str, str]]:
        """
        Args:
            view_images_rgb: 4 张视角 RGB 图（顺序：az=0/90/180/270, elevation 统一）
            target_ku: TargetKURReconstructor 的输出
            elevation: 仰角，仅用于 prompt 描述
            max_distractors: prompt 中告诉 MLLM 的最大数量

        Returns:
            distractors: List[{short_label, brief_features}]，可能为空。
        """
        if self.client is None or not view_images_rgb:
            return []

        collage = make_2x2_collage(view_images_rgb)
        b64_collage = _encode_rgb_png_b64(collage)

        prompt = DISTRACTOR_DISCOVERY_PROMPT_TMPL.format(
            elevation=int(round(elevation)),
            target_label=target_ku.get("short_label", "target"),
            target_spatial=target_ku.get("spatial_location", "未提及"),
            target_shape=target_ku.get("typical_3D_shape", "未提及"),
            max_distractors=max_distractors,
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
                    return self._sanitize(
                        parsed,
                        target_label=target_ku.get("short_label", "target"),
                        max_distractors=max_distractors,
                    )
            except Exception as e:
                print(f"[DistractorDiscoverer] attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt + 1 < max_retries:
                    time.sleep(2)

        print("[DistractorDiscoverer] All retries failed; returning empty distractor list.")
        return []

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
    def _sanitize(
        parsed: Dict,
        target_label: str,
        max_distractors: int,
    ) -> List[Dict[str, str]]:
        """清洗 distractor 列表：去重、避免与 target 撞名、截断到 max_distractors。"""
        raw = parsed.get("distractor_parts", []) if isinstance(parsed, dict) else []
        if not isinstance(raw, list):
            return []

        target_lc = target_label.strip().lower()
        seen = set([target_lc])
        out: List[Dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = str(item.get("short_label", "")).strip()
            if not label:
                continue
            # 清洗
            label = re.sub(r"^a\s+|^an\s+|^the\s+", "", label, flags=re.IGNORECASE)
            label = re.sub(r"\s+", " ", label)[:40]
            label_lc = label.lower()
            if label_lc in seen or not label_lc:
                continue
            # 撞名兜底（包含关系 → 加后缀）
            collision = (label_lc == target_lc) or (
                target_lc and (label_lc in target_lc or target_lc in label_lc)
                and label_lc != target_lc
                and abs(len(label_lc) - len(target_lc)) <= 2
            )
            if collision:
                label = label + " (other)"
                label_lc = label.lower()
                if label_lc in seen:
                    continue
            seen.add(label_lc)
            features = str(item.get("brief_features", "")).strip()
            features = re.sub(r"\s+", " ", features)[:120]
            out.append({"short_label": label, "brief_features": features})
            if len(out) >= max_distractors:
                break
        return out

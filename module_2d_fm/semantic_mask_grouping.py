"""
Semantic grouping for 2D mask visualization and MLLM feedback.

Same semantic class shares one display id and one color. Adjacent masks of the
same class are merged for drawing; disconnected blobs keep the same id/color
and repeat the same digit at each blob's centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np


def _masks_touch(m1: np.ndarray, m2: np.ndarray, gap: int = 1) -> bool:
    a = m1.astype(bool)
    b = m2.astype(bool)
    if not a.any() or not b.any():
        return False
    k = max(1, 2 * gap + 1)
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(a.astype(np.uint8), kernel)
    return bool(np.logical_and(dil.astype(bool), b).any())


def _find_parent(parent: List[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent: List[int], i: int, j: int) -> None:
    ri, rj = _find_parent(parent, i), _find_parent(parent, j)
    if ri != rj:
        parent[rj] = ri


def _display_colors(n: int) -> List[Tuple[int, int, int]]:
    """BGR colors, stable and distinct."""
    if n <= 0:
        return []
    base = [
        (255, 128, 0),
        (0, 255, 128),
        (128, 0, 255),
        (255, 255, 100),
        (100, 200, 255),
        (255, 100, 200),
        (180, 255, 180),
        (255, 180, 100),
        (100, 255, 255),
        (200, 100, 255),
    ]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


@dataclass
class SemanticGroupingResult:
    """Metadata for feedback + refinement mapping."""

    id_to_semantic: Dict[int, str]
    semantic_to_display_id: Dict[str, int]
    display_id_to_mask_indices: Dict[int, List[int]]
    original_index_to_display_id: List[int]
    legend_lines: List[str]
    # --- Fragment-level tracking (Layer 0) ---
    # Per display_id: list of fragment structs.
    # Each fragment is a dict with keys:
    #   fragment_id  (int, unique within the whole view)
    #   mask_indices (List[int], original mask indices belonging to this fragment)
    #   merged_mask  (np.ndarray, combined boolean mask of the fragment)
    #   centroid      (tuple (x, y))
    display_id_to_fragments: Dict[int, List[dict]]
    all_fragment_ids: List[int]  # flat list of all fragment_ids in this view


def build_semantic_grouping(
    masks: List[np.ndarray],
    semantics: List[str],
    ignore_semantics: Optional[set] = None,
    merge_adjacent_same_semantic: bool = True,
) -> SemanticGroupingResult:
    """
    Assign a small integer display id per distinct semantic string (among masks).
    Maps each original mask index to that id. Multiple masks with the same
    semantic map to the same display id.

    Args:
        merge_adjacent_same_semantic: 若为 True（默认），同一语义下**空间相邻**的原始掩码会
            合并为一个 fragment（连通域）；若为 False，**每个原始掩码单独**成为一个 fragment，
            便于 MLLM 按碎片删除而不会因为相邻合并导致「一删删一片」。
    """
    if ignore_semantics is None:
        ignore_semantics = set()
    ignore_lower = {x.lower() for x in ignore_semantics}

    n = len(masks)
    if len(semantics) != n:
        raise ValueError("masks and semantics must have the same length")

    present: List[str] = []
    for i in range(n):
        sem = semantics[i] if i < len(semantics) else "unknown"
        if not isinstance(sem, str):
            sem = str(sem)
        sem = sem.strip() or "unknown"
        if sem.lower() in ignore_lower:
            continue
        ys, xs = np.where(masks[i] > 0)
        if len(ys) == 0:
            continue
        present.append(sem)

    unique_sorted = sorted(set(present), key=lambda s: (s.lower(), s))
    semantic_to_display_id: Dict[str, int] = {s: k for k, s in enumerate(unique_sorted)}
    id_to_semantic: Dict[int, str] = {k: s for s, k in semantic_to_display_id.items()}

    original_index_to_display_id: List[int] = []
    display_id_to_mask_indices: Dict[int, List[int]] = {k: [] for k in id_to_semantic}

    for i in range(n):
        sem = semantics[i] if i < len(semantics) else "unknown"
        if not isinstance(sem, str):
            sem = str(sem)
        sem = sem.strip() or "unknown"
        if sem.lower() in ignore_lower:
            original_index_to_display_id.append(-1)
            continue
        ys, _ = np.where(masks[i] > 0)
        if len(ys) == 0:
            original_index_to_display_id.append(-1)
            continue
        did = semantic_to_display_id.get(sem, -1)
        if did < 0:
            original_index_to_display_id.append(-1)
        else:
            original_index_to_display_id.append(did)
            display_id_to_mask_indices[did].append(i)

    # --- Fragment-level tracking (Layer 0) ---
    # Determine image dimensions from first non-empty mask
    frag_h, frag_w = 0, 0
    for i in range(n):
        if np.any(masks[i] > 0):
            frag_h, frag_w = masks[i].shape[:2]
            break

    display_id_to_fragments: Dict[int, List[dict]] = {}
    all_fragment_ids: List[int] = []
    # Build fragment structures per display_id
    fragment_counter = 0
    for did, idxs in display_id_to_mask_indices.items():
        valid_idxs = [i for i in idxs if 0 <= i < n and np.any(masks[i] > 0)]
        if not valid_idxs:
            display_id_to_fragments[did] = []
            continue

        if merge_adjacent_same_semantic:
            components = _connected_components_same_semantic(masks, valid_idxs, touch_gap=1)
        else:
            components = [[i] for i in valid_idxs]
        fragments_list = []
        for comp in components:
            merged = np.zeros((frag_h, frag_w), dtype=bool)
            for mi in comp:
                merged |= masks[mi].astype(bool)
            ys, xs = np.where(merged)
            if len(ys) == 0:
                cx, cy = 0, 0
            else:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
            frag = {
                "fragment_id": fragment_counter,
                "mask_indices": comp,
                "merged_mask": merged,
                "centroid": (cx, cy),
            }
            fragments_list.append(frag)
            all_fragment_ids.append(fragment_counter)
            fragment_counter += 1
        display_id_to_fragments[did] = fragments_list

    legend_lines = [f"编号{d}：{id_to_semantic[d]}" for d in sorted(id_to_semantic.keys())]
    return SemanticGroupingResult(
        id_to_semantic=id_to_semantic,
        semantic_to_display_id=semantic_to_display_id,
        display_id_to_mask_indices=display_id_to_mask_indices,
        original_index_to_display_id=original_index_to_display_id,
        legend_lines=legend_lines,
        display_id_to_fragments=display_id_to_fragments,
        all_fragment_ids=all_fragment_ids,
    )


def fragments_for_semantic_ordered(
    grouping: SemanticGroupingResult,
    sem_name: str,
) -> List[dict]:
    """
    返回某语义在该视角下的 fragment 列表，顺序稳定（用于 MLLM 标签 a0/a1 与 delete_fragments 下标对齐）。
    排序：(centroid_y, centroid_x, fragment_id)。
    """
    local_gid = semantic_string_to_display_id(grouping, sem_name)
    if local_gid is None:
        return []
    frags = grouping.display_id_to_fragments.get(local_gid, [])
    return sorted(
        frags,
        key=lambda f: (f.get("centroid", (0, 0))[1], f.get("centroid", (0, 0))[0], f.get("fragment_id", 0)),
    )


def semantic_string_to_display_id(
    grouping: SemanticGroupingResult,
    sem_name: str,
) -> Optional[int]:
    """
    将部件名字符串映射到该视角 grouping 的 display_id。

    display_id 由「当前视角出现的语义名」排序分配，与 prompt_classes 的下标一般不一致；
    Step 2.5 / 可视化里若把 MLLM 的 display_id（实为 prompt 下标）直接当 grouping id 用会错位。
    """
    if not sem_name or not grouping.semantic_to_display_id:
        return None
    if sem_name in grouping.semantic_to_display_id:
        return grouping.semantic_to_display_id[sem_name]
    low = sem_name.lower().strip()
    for k, v in grouping.semantic_to_display_id.items():
        if isinstance(k, str) and k.lower().strip() == low:
            return v
    return None


def _connected_components_same_semantic(
    masks: List[np.ndarray],
    mask_indices: List[int],
    touch_gap: int = 1,
) -> List[List[int]]:
    """Union masks by adjacency within one semantic group (mask_indices share one label)."""
    m = len(mask_indices)
    if m == 0:
        return []
    parent = list(range(m))

    local_masks = [masks[mask_indices[i]].astype(bool) for i in range(m)]

    for i in range(m):
        for j in range(i + 1, m):
            if _masks_touch(local_masks[i], local_masks[j], gap=touch_gap):
                _union(parent, i, j)

    comp: Dict[int, List[int]] = {}
    for i in range(m):
        r = _find_parent(parent, i)
        comp.setdefault(r, []).append(mask_indices[i])

    return list(comp.values())


def render_semantic_grouped_overlay(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    semantics: List[str],
    grouping: SemanticGroupingResult,
    alpha: float = 0.35,
    contour_thickness: int = 2,
    touch_gap: int = 1,
    only_display_ids: Optional[Set[int]] = None,
    label_style: str = "id",
) -> np.ndarray:
    """
    BGR image: base image + merged (adjacent same-semantic) overlays, one color
    per display id; same digit at each disconnected blob's centroid.

    Args:
        only_display_ids: 若给定，只绘制这些 grouping display_id（用于 sem_XX 目录下仅看当前部件）。
        label_style: "id" 仅编号； "semantic" 仅语义短名； "both" 为 "id:语义"（过长则截断）。
    """
    vis_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = vis_bgr.shape[:2]

    colors = _display_colors(len(grouping.id_to_semantic))

    # Collect components: (display_id, merged_bool_mask)
    blobs: List[Tuple[int, np.ndarray]] = []

    for did, sem in sorted(grouping.id_to_semantic.items()):
        if only_display_ids is not None and did not in only_display_ids:
            continue
        idxs = grouping.display_id_to_mask_indices.get(did, [])
        idxs = [i for i in idxs if 0 <= i < len(masks) and np.any(masks[i] > 0)]
        if not idxs:
            continue
        components = _connected_components_same_semantic(masks, idxs, touch_gap=touch_gap)
        for comp in components:
            merged = np.zeros((h, w), dtype=bool)
            for mi in comp:
                merged |= masks[mi].astype(bool)
            blobs.append((did, merged))

    # Draw in stable order so later contours sit on top predictably
    blobs.sort(key=lambda t: (t[0], -int(t[1].sum())))

    for did, merged in blobs:
        color = colors[did % len(colors)]
        overlay = vis_bgr.copy()
        overlay[merged] = color
        cv2.addWeighted(overlay, alpha, vis_bgr, 1.0 - alpha, 0, vis_bgr)

        contours, _ = cv2.findContours(merged.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_bgr, contours, -1, color, contour_thickness)

        dt = cv2.distanceTransform(merged.astype(np.uint8), cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dt)
        if max_val > 0:
            cX, cY = int(max_loc[0]), int(max_loc[1])
        else:
            ys, xs = np.where(merged)
            cX, cY = int(np.mean(xs)), int(np.mean(ys))

        sem_name = grouping.id_to_semantic.get(did, "")
        if label_style == "semantic":
            raw = str(sem_name) if sem_name else str(did)
            text = raw[:18] + ("…" if len(raw) > 18 else "")
        elif label_style == "both":
            raw_sem = str(sem_name) if sem_name else "?"
            short_sem = raw_sem[:14] + ("…" if len(raw_sem) > 14 else "")
            text = f"{did}:{short_sem}"
        else:
            text = str(did)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35 if label_style != "id" else 0.4
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            vis_bgr,
            (cX - tw // 2 - 1, cY - th - 1),
            (cX + tw // 2 + 1, cY + baseline + 1),
            (0, 0, 0),
            -1,
        )
        cv2.putText(vis_bgr, text, (cX - tw // 2, cY), font, font_scale, (255, 255, 255), thickness)

    return vis_bgr


def format_legend_for_prompt(lines: List[str]) -> str:
    if not lines:
        return "（当前无有效前景掩码的语义分组）"
    return "\n".join(lines)


def render_part_centric_overlay(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    semantics: List[str],
    grouping: SemanticGroupingResult,
    alpha: float = 0.35,
    contour_thickness: int = 2,
    touch_gap: int = 1,
) -> Dict[int, np.ndarray]:
    """
    为每个预定义部件（每个 display_id）渲染一张独立的视角图。
    同一部件内的碎片保持同一颜色，但每个碎片有独立的 fragment_id 标注。
    返回: Dict[display_id, vis_image_rgb]，只有出现碎片的 display_id 才有图。
    """
    import cv2
    vis_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = vis_bgr.shape[:2]
    colors = _display_colors(len(grouping.id_to_semantic))

    result_images: Dict[int, np.ndarray] = {}

    for did, fragments in grouping.display_id_to_fragments.items():
        if not fragments:
            continue

        color = colors[did % len(colors)]
        overlay = vis_bgr.copy()

        for frag in fragments:
            frag_merged = frag["merged_mask"]
            overlay[frag_merged] = color

            contours, _ = cv2.findContours(
                frag_merged.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, contour_thickness)

        cv2.addWeighted(overlay, alpha, vis_bgr.copy(), 1.0 - alpha, 0, overlay)

        cx_all, cy_all = [], []
        for frag in fragments:
            frag_merged = frag["merged_mask"]
            ys, xs = np.where(frag_merged)
            if len(ys) > 0:
                cx_all.extend(xs)
                cy_all.extend(ys)

        if cx_all:
            overall_cx, overall_cy = int(np.mean(cx_all)), int(np.mean(cy_all))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text = f"ID={did}"
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(
                overlay,
                (overall_cx - tw // 2 - 2, overall_cy - th - 2),
                (overall_cx + tw // 2 + 2, overall_cy + baseline + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                overlay, text, (overall_cx - tw // 2, overall_cy),
                font, font_scale, (255, 255, 255), thickness
            )

        for frag in fragments:
            frag_fid = frag["fragment_id"]
            frag_merged = frag["merged_mask"]
            cfx, cfy = frag["centroid"]

            text = f"f{frag_fid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            bx0 = cfx - tw // 2 - 1
            by0 = cfy - th - 1
            bx1 = cfx + tw // 2 + 1
            by1 = cfy + baseline + 1
            cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
            cv2.putText(overlay, text, (bx0 + 1, cfy), font, font_scale, (255, 255, 255), thickness)

        result_images[did] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return result_images

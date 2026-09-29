"""
test_mask_backproj.py

Generate paper-figure assets for the topology-knowledge pipeline. Given a
category/instance, run the relevant pipeline stages and write artifacts into
`<vis_dir>/mask_backproj/`:

Stage A -- single-mask back-projection (always produced):

    1. <id>_..._2d_view.png       Full rendered view with ONE chosen SAM mask
                                  outlined and tinted in a vivid color.
    2. <id>_..._3d_color.ply      Full point cloud rendered as gray balls,
                                  except the mask's back-projected 3D points
                                  which are recolored to the same vivid color.
    3. <id>_..._3d_color.png      Mitsuba ball-grain render of (2) at the
                                  paper's canonical (elev=20, azim=45) view.

Stage B -- multi-view-confirmed anchor set X^star (controlled by
``--no-anchor-set``; this is the "after the omitted pipeline steps" view):

    4. <id>_anchor_set_3d.ply     Full point cloud where every point that was
                                  confirmed by >=3 SAM-mask votes (conf>=0.70)
                                  with the same first-pass-MLLM label is
                                  recolored to its part color (saturated palette
                                  matching final_result.ply / 08_final_seg.png),
                                  and unconfirmed points stay light-gray.
    5. <id>_anchor_set_3d.png     Mitsuba render of (4) at (elev=20, azim=45),
                                  same render style as 08_final_seg.png.
    6. <id>_anchor_set_summary.json  per-class confirmed-point counts + params.

Stage B requires a real first-pass MLLM call (so it costs API tokens and a few
minutes). Stage A only needs the renderer + SAM, so it's fast.

Mirrors `test.py` for GPU selection and pipeline init.
"""

import os

# Mirror test.py: keep CUDA device order consistent with nvidia-smi.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import open3d as o3d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import Arbitr3DPipeline
from module_render.camera_poses import generate_sphere_cameras
from scripts.process_masks import process_masks
from module_3d_lift.geometric_features import GeometricFeatureExtractor
from module_3d_lift.topology_builder import TopologyGraphBuilder


# ---------------------------------------------------------------------------
# Vivid highlight color for the chosen mask (RGB 0-255). Warm red-orange reads
# well on a desaturated gray background and stays distinguishable when printed.
# ---------------------------------------------------------------------------
HIGHLIGHT_RGB = (242, 82, 58)


# ---------------------------------------------------------------------------
# Anchor-set palette: saturated primaries that match `final_result.ply` and
# `pred_chair_*.ply` -- the same colors `08_final_seg.png` is rendered from.
# Keys mirror PartNetE part labels for Chair; if other categories need extra
# parts, the unmatched semantic falls back to the gray "unconfirmed" color
# (so nothing crashes; the figure just won't highlight that part).
# ---------------------------------------------------------------------------
PART_RGB_SATURATED = {
    "back":  (  0, 255,   0),   # green
    "arm":   (255,   0,   0),   # red
    "seat":  (255, 255,   0),   # gold/yellow
    "leg":   (  0,   0, 255),   # blue
    "wheel": (128,   0, 128),   # purple
}

# Light gray used for points that did NOT receive >= n_min consistent votes.
# Matches the gray that `_make_highlight_ply` and `Visualizer3D._bg` use, so
# the anchor-set image lives in the same color space as 10_mask_backproj.png
# and 08_final_seg.png.
GRAY_RGB_UNCONFIRMED = (179, 179, 179)

# Topology builder defaults used by the real pipeline (pipeline.py 789-791).
# Pipeline passes only confidence_threshold=0.70; min_vote_count keeps the
# TopologyGraphBuilder default of 3.
ANCHOR_CONFIDENCE_THRESH = 0.70
ANCHOR_MIN_VOTE_COUNT = 3


# ---------------------------------------------------------------------------
# GPU selection: copy of the resolution logic in test.py so this script is
# runnable identically. We keep the helpers small and dependency-free.
# ---------------------------------------------------------------------------
def _get_physical_gpu_map() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
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


def _auto_best_physical(physical_map: dict, threshold_gb: float = 4.0) -> int:
    if not physical_map:
        raise RuntimeError("No GPU available. Cannot proceed.")
    candidates = {
        idx: free for idx, (free, _) in physical_map.items() if free >= threshold_gb
    }
    if not candidates:
        raise RuntimeError(
            f"No GPU with >= {threshold_gb:.1f} GB free memory. Aborting."
        )
    return max(candidates, key=lambda i: candidates[i])


def _resolve_gpu() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "auto").strip()
    if raw.lower() in ("", "auto"):
        return _auto_best_physical(_get_physical_gpu_map())
    try:
        return int(raw)
    except ValueError:
        return _auto_best_physical(_get_physical_gpu_map())


# ---------------------------------------------------------------------------
# Mask selection: find one "medium-sized" mask globally across all views.
# ---------------------------------------------------------------------------
def _is_background_mask(mask: np.ndarray, frac_thresh: float = 0.5) -> bool:
    """Mirror pipeline.py's is_background_mask: a mask that covers >50% of any
    image border edge is treated as the background plate, not a part."""
    h, w = mask.shape
    return bool(
        np.sum(mask[0, :]) > w * frac_thresh
        or np.sum(mask[-1, :]) > w * frac_thresh
        or np.sum(mask[:, 0]) > h * frac_thresh
        or np.sum(mask[:, -1]) > h * frac_thresh
    )


def _shape_stats(mask: np.ndarray) -> Optional[dict]:
    """Return geometric shape stats of a binary mask, or None if degenerate.

    Keys: area, bbox=(x0,y0,x1,y1), bw, bh, aspect=W/H, extent=area/bbox_area,
          solidity=area/convex_hull_area, centroid=(cx,cy).
    All values use image (row, col) indexing converted to (x=col, y=row).
    """
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    area = int(rows.size)
    bbox_area = bw * bh
    if bbox_area == 0 or bh == 0:
        return None
    aspect = bw / bh
    extent = area / bbox_area

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    return {
        "area": area,
        "bbox": (x0, y0, x1, y1),
        "bw": bw,
        "bh": bh,
        "aspect": aspect,
        "extent": extent,
        "solidity": float(solidity),
        "centroid": (float(cols.mean()), float(rows.mean())),
    }


def _select_medium_mask(
    masks_across_views: List[List[np.ndarray]],
    index_maps: List[np.ndarray],
    frac_low: float = 0.025,
    frac_high: float = 0.10,
    min_unique_3d: int = 200,
    aspect_min: float = 0.55,
    aspect_max: float = 1.80,
    extent_min: float = 0.45,
    solidity_min: float = 0.80,
) -> Optional[Tuple[int, int, np.ndarray, np.ndarray, dict]]:
    """Return (view_idx, mask_idx, mask_on_object, unique_pc_indices, stats) for
    the most paper-friendly medium-sized mask across all views.

    Filters (a mask must satisfy all):
        1. Area / object_pixels in [frac_low, frac_high].
        2. Not a background plate (no border edge >= 50% covered).
        3. >= min_unique_3d distinct back-projected 3D points.
        4. Bbox aspect ratio (W/H) in [aspect_min, aspect_max] -- avoids slivers.
        5. Extent (mask_area / bbox_area) >= extent_min -- mask fills its bbox.
        6. Solidity (mask_area / convex_hull_area) >= solidity_min -- mask is
           convex-ish, no skinny tendrils.

    Score (lower = better, ties broken by view index):
        score = compactness_term + 0.4 * centeredness_term
        compactness_term = (1 - solidity) + 0.5 * (1 - extent)
                         + 0.5 * |log(aspect)|
        centeredness_term = ((cy - h/2)/h)**2 + ((cx - w/2)/w)**2
    """
    best = None
    best_score = float("inf")

    for v_idx, (masks, idx_map) in enumerate(zip(masks_across_views, index_maps)):
        if not masks:
            continue
        imap = idx_map[:, :, 0] if idx_map.ndim == 3 else idx_map
        object_mask = imap >= 0
        object_pixels = int(np.sum(object_mask))
        if object_pixels == 0:
            continue
        h, w = imap.shape

        for m_idx, m in enumerate(masks):
            if _is_background_mask(m):
                continue
            mask_on_obj = np.logical_and(m, object_mask)
            stats = _shape_stats(mask_on_obj)
            if stats is None:
                continue
            area = stats["area"]
            ratio = area / object_pixels
            if not (frac_low <= ratio <= frac_high):
                continue
            if not (aspect_min <= stats["aspect"] <= aspect_max):
                continue
            if stats["extent"] < extent_min:
                continue
            if stats["solidity"] < solidity_min:
                continue

            pc_ids = imap[mask_on_obj]
            unique_pc = np.unique(pc_ids[pc_ids >= 0])
            if len(unique_pc) < min_unique_3d:
                continue

            cx_m, cy_m = stats["centroid"]
            centered = ((cy_m - h / 2) / h) ** 2 + ((cx_m - w / 2) / w) ** 2
            compact = (
                (1.0 - stats["solidity"])
                + 0.5 * (1.0 - stats["extent"])
                + 0.5 * abs(np.log(max(stats["aspect"], 1e-6)))
            )
            score = compact + 0.4 * centered

            stats_with_meta = dict(stats)
            stats_with_meta.update({
                "area_ratio": ratio,
                "unique_3d": int(len(unique_pc)),
                "compact_score": float(compact),
                "centered_score": float(centered),
                "total_score": float(score),
            })

            if score < best_score:
                best_score = score
                best = (v_idx, m_idx, mask_on_obj, unique_pc, stats_with_meta)

    return best


# ---------------------------------------------------------------------------
# 2D output: full rendered view with ONLY the chosen mask highlighted
# ---------------------------------------------------------------------------
def _render_2d_full_view(
    rgb_img: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    color_rgb: Tuple[int, int, int] = HIGHLIGHT_RGB,
    overlay_alpha: float = 0.40,
    contour_thickness: int = 3,
) -> None:
    """Save the full rendered view with the chosen mask semi-transparently
    tinted and outlined in `color_rgb`. The rest of the image (including any
    other SAM masks) stays as the raw render -- only one mask is highlighted.
    """
    if mask.shape[:2] != rgb_img.shape[:2]:
        raise ValueError("mask and rgb_img must share spatial dims")

    img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

    bool_mask = mask.astype(bool)
    overlay = img_bgr.copy()
    overlay[bool_mask] = color_bgr
    cv2.addWeighted(overlay, overlay_alpha, img_bgr, 1 - overlay_alpha, 0, img_bgr)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img_bgr, contours, -1, color_bgr, contour_thickness)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr)


# ---------------------------------------------------------------------------
# 3D output: original-color PC with vivid highlight on the mask's 3D points
# ---------------------------------------------------------------------------
def _stratified_subsample(
    pc_xyz: np.ndarray,
    pc_colors: np.ndarray,
    highlight_indices: np.ndarray,
    target_total: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample the point cloud down to ~`target_total` points, but keep ALL
    `highlight_indices` rows (so the back-projected mask region is fully
    rendered, not statistically thinned by Mitsuba's random subsampler).

    Returns (xyz_sub, colors_sub, new_highlight_indices_into_subsample).
    If `len(pc_xyz) <= target_total`, returns the inputs unchanged.
    """
    n = len(pc_xyz)
    if n <= target_total:
        return pc_xyz, pc_colors, np.asarray(highlight_indices, dtype=np.int64)

    valid_hl = np.asarray(highlight_indices, dtype=np.int64)
    valid_hl = np.unique(valid_hl[(valid_hl >= 0) & (valid_hl < n)])

    is_highlight = np.zeros(n, dtype=bool)
    is_highlight[valid_hl] = True
    other_idx = np.where(~is_highlight)[0]

    n_hl = int(valid_hl.size)
    n_other_budget = max(0, target_total - n_hl)

    rng = np.random.default_rng(seed)
    if n_other_budget >= other_idx.size:
        keep_other = other_idx
    elif n_other_budget == 0:
        keep_other = np.empty(0, dtype=np.int64)
    else:
        keep_other = rng.choice(other_idx, size=n_other_budget, replace=False)

    keep = np.concatenate([valid_hl, keep_other])
    # Mix order so the renderer doesn't see a positional bias (highlights
    # all at the front of the array).
    rng.shuffle(keep)

    new_is_hl = is_highlight[keep]
    new_hl = np.where(new_is_hl)[0].astype(np.int64)
    return pc_xyz[keep], pc_colors[keep], new_hl


def _write_highlighted_ply(
    pc_xyz: np.ndarray,
    pc_colors: np.ndarray,
    highlight_indices: np.ndarray,
    out_path: Path,
    highlight_rgb: Tuple[int, int, int] = HIGHLIGHT_RGB,
) -> None:
    """Save a PLY where `highlight_indices` rows of `pc_colors` are overridden
    by `highlight_rgb` and every other point keeps its original color.
    """
    colors = np.asarray(pc_colors, dtype=np.float32).copy()
    if colors.max() > 1.5:
        colors = colors / 255.0

    hl = np.array(highlight_rgb, dtype=np.float32) / 255.0
    valid = np.asarray(highlight_indices, dtype=np.int64)
    valid = valid[(valid >= 0) & (valid < len(colors))]
    if valid.size > 0:
        colors[valid] = hl[None, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pc_xyz, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)


# ---------------------------------------------------------------------------
# Mitsuba render (parameters match scripts/render_overview_thumbnails.py so
# the output sits next to overview_thumbs/01_input_pc.png stylistically.)
# ---------------------------------------------------------------------------
# Canonical view used by the paper's overview figure.
CANONICAL_ELEV = 20.0
CANONICAL_AZIM = 45.0

# Camera / look defaults pulled from render_overview_thumbnails.MITSUBA_DEFAULT_KWARGS
# so the chair fits the canvas and the granularity matches the reference.
MITSUBA_OVERVIEW_DEFAULTS = dict(
    spp=256,
    max_points=10000,
    ball_radius_ratio=0.009,
    camera_distance=7.0,
    fov=18.0,
    light_strength=5.5,
    variant="scalar_rgb",
)


def _mitsuba_supports_flag(script: Path, flag: str) -> bool:
    """Probe `render_balls_mitsuba.py --help` to see if `flag` is recognized.
    Older copies of the script (e.g. on the Linux server) lack `--variant`, so
    we have to skip those flags rather than crashing the whole invocation.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return False
    help_text = (result.stdout or "") + (result.stderr or "")
    return flag in help_text


def _maybe_render_mitsuba(
    ply_path: Path,
    png_path: Path,
    elev: float = CANONICAL_ELEV,
    azim: float = CANONICAL_AZIM,
    resolution: Tuple[int, int] = (1024, 1024),
    transparent_bg: bool = True,
    overrides: Optional[dict] = None,
) -> bool:
    """Invoke `scripts/render_balls_mitsuba.py` to render `ply_path` with the
    granular ball-grain style at (elev, azim). Camera params default to the
    overview-thumbnail look (camera_distance=8.0, fov=18, ...) so the whole
    object fits in frame; pass `overrides` to tweak any of those.
    """
    project_root = Path(__file__).parent
    script = project_root / "scripts" / "render_balls_mitsuba.py"
    if not script.is_file():
        print(f"  [mitsuba] script not found at {script}, skipping")
        return False

    params = {**MITSUBA_OVERVIEW_DEFAULTS, **(overrides or {})}
    cmd = [
        sys.executable, str(script),
        "--input-path", str(ply_path),
        "--output-dir", str(png_path.parent),
        "--palette", "as-is",
        "--views", f"{elev:.1f},{azim:.1f}",
        "--resolution", f"{resolution[0]},{resolution[1]}",
        "--spp", str(params["spp"]),
        "--max-points", str(params["max_points"]),
        "--ball-radius-ratio", str(params["ball_radius_ratio"]),
        "--camera-distance", str(params["camera_distance"]),
        "--fov", str(params["fov"]),
        "--light-strength", str(params["light_strength"]),
    ]
    # --variant was added in a later revision of render_balls_mitsuba.py; skip
    # gracefully on older copies (e.g. the Linux server currently has an older
    # version without this flag -- the script then auto-picks cuda > llvm >
    # scalar at runtime, which works fine on Linux + CUDA).
    variant = params.get("variant")
    if variant and _mitsuba_supports_flag(script, "--variant"):
        cmd.extend(["--variant", str(variant)])
    if transparent_bg:
        cmd.append("--transparent-bg")
    print(f"  [mitsuba] running: {' '.join(cmd[1:])}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except Exception as e:
        print(f"  [mitsuba] subprocess failed: {e}")
        return False

    if result.returncode != 0:
        print("  [mitsuba] non-zero exit:")
        print((result.stderr or result.stdout)[:1500])
        return False

    # render_balls_mitsuba.py writes <stem>_e<elev>_a<azim>.png; rename to the
    # predictable target so callers can reference it directly.
    expected = png_path.parent / (
        f"{ply_path.stem}_e{int(round(elev))}_a{int(round(azim))}.png"
    )
    if expected.is_file():
        if expected != png_path:
            try:
                expected.replace(png_path)
            except Exception as e:
                print(f"  [mitsuba] rename {expected.name} -> {png_path.name} failed: {e}")
                return False
        return png_path.is_file()

    # Fallback: pick the most recent PNG with matching stem prefix.
    candidates = sorted(
        png_path.parent.glob(f"{ply_path.stem}*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates and candidates[0] != png_path:
        try:
            candidates[0].replace(png_path)
        except Exception as e:
            print(f"  [mitsuba] rename fallback failed: {e}")
            return False
    return png_path.is_file()


# ---------------------------------------------------------------------------
# Anchor-set helpers (Stage B: first-pass MLLM + per-point voting)
# ---------------------------------------------------------------------------
def _load_prompt_classes(meta_path: Path, category: str) -> List[str]:
    """Read PartNetE_meta.json and return the list of part labels for `category`.
    Mirrors `test.py:load_prompt_classes_from_meta` but returns a list instead
    of a comma-joined string (the MLLM API takes a list).
    """
    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if category not in meta:
        raise ValueError(f"Category '{category}' not in {meta_path}")
    return [str(p) for p in meta[category]]


def _load_unified_knowledge(category: str, prompt_classes: List[str]) -> dict:
    """Load and per-instance filter `<category>_unified_knowledge.json`, mirroring
    the logic in pipeline.py 602-627 so the MLLM sees the same prompt text.
    """
    project_root = Path(__file__).parent
    path = project_root / "config" / "knowledge" / f"{category}_unified_knowledge.json"
    if not path.is_file():
        print(f"  [knowledge] not found, MLLM will run without K_u: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        knowledge = json.load(f)
    if category in knowledge:
        full = knowledge[category]
        prompt_set = set(prompt_classes)
        filtered = {p: info for p, info in full.items() if p in prompt_set}
        knowledge = {category: filtered}
        if len(filtered) < len(full):
            print(f"  [knowledge] filtered to {len(filtered)}/{len(full)} parts "
                  f"matching prompt_classes")
    return knowledge


def _build_view_angles_str(cam_positions: List[np.ndarray]) -> List[str]:
    """Construct view_angles_str using the exact format pipeline.py 451-456 uses,
    so the MLLM prompt is byte-identical to a real pipeline run."""
    out: List[str] = []
    for pos in cam_positions:
        pos = np.asarray(pos, dtype=np.float64)
        r = float(np.linalg.norm(pos))
        if r > 0:
            elev = float(np.degrees(np.arcsin(pos[1] / r)))
            azim = float(np.degrees(np.arctan2(pos[0], pos[2])))
        else:
            elev, azim = 0.0, 0.0
        out.append(f"仰角 {elev:.1f}度, 方位角 {azim:.1f}度")
    return out


def _build_initial_instance_knowledge(
    pc_xyz: np.ndarray,
    index_maps: List[np.ndarray],
    masks_across_views: List[List[np.ndarray]],
) -> dict:
    """Reproduce pipeline.py 648-686: per-(view, mask) raw geometric features
    keyed by `view_<v>` -> `mask_<m>` -> dict(curvature/z/ratio/...).
    """
    extractor = GeometricFeatureExtractor(pc_xyz)
    knowledge: dict = {}
    for vid, (masks, idx_map) in enumerate(zip(masks_across_views, index_maps)):
        vk = f"view_{vid}"
        knowledge[vk] = {}
        imap = idx_map[:, :, 0] if idx_map.ndim == 3 else idx_map
        for m_idx, m in enumerate(masks):
            ys, xs = np.where(m > 0)
            if ys.size > 0:
                pc_indices = imap[ys, xs]
                valid = pc_indices[pc_indices >= 0]
            else:
                valid = np.empty(0, dtype=np.int64)
            if valid.size > 0:
                feat = extractor.compute_features(pc_xyz[valid])
            else:
                feat = extractor._empty_features()
            knowledge[vk][f"mask_{m_idx}"] = feat
    return knowledge


def _subsample_keeping_anchor(
    pc_xyz: np.ndarray,
    pc_colors: np.ndarray,
    point_semantics: dict,
    target_total: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Subsample down to ~target_total points but keep EVERY confirmed anchor
    point (same idea as `_stratified_subsample` but rewrites the semantic dict
    so its keys index into the subsample rather than the original PC).

    Returns (xyz_sub, colors_sub, point_semantics_new).
    """
    n = len(pc_xyz)
    if n <= target_total:
        return pc_xyz, pc_colors, dict(point_semantics)

    confirmed_old = np.array(sorted(int(k) for k in point_semantics.keys()),
                              dtype=np.int64)
    confirmed_old = confirmed_old[(confirmed_old >= 0) & (confirmed_old < n)]

    is_conf = np.zeros(n, dtype=bool)
    is_conf[confirmed_old] = True
    other_idx = np.where(~is_conf)[0]

    n_other_budget = max(0, target_total - int(confirmed_old.size))
    rng = np.random.default_rng(seed)
    if n_other_budget >= other_idx.size:
        keep_other = other_idx
    elif n_other_budget == 0:
        keep_other = np.empty(0, dtype=np.int64)
    else:
        keep_other = rng.choice(other_idx, size=n_other_budget, replace=False)

    keep = np.concatenate([confirmed_old, keep_other])
    rng.shuffle(keep)

    new_idx_of_old = {int(old): int(new_i) for new_i, old in enumerate(keep)}
    new_sem: dict = {}
    for old, sem in point_semantics.items():
        new_i = new_idx_of_old.get(int(old))
        if new_i is not None:
            new_sem[new_i] = sem

    return pc_xyz[keep], pc_colors[keep], new_sem


def _write_anchor_set_ply(
    pc_xyz: np.ndarray,
    point_semantics: dict,
    out_path: Path,
    palette_rgb: dict = PART_RGB_SATURATED,
    gray_rgb: Tuple[int, int, int] = GRAY_RGB_UNCONFIRMED,
) -> Tuple[int, dict]:
    """Write a PLY where every point starts as `gray_rgb` and confirmed anchor
    points are overwritten with their part color from `palette_rgb`.

    Returns (n_confirmed_total, per_class_counts).
    """
    n = len(pc_xyz)
    colors = np.tile(np.asarray(gray_rgb, dtype=np.float32) / 255.0, (n, 1))
    per_class: dict = {}
    n_conf = 0
    for pidx, sem in point_semantics.items():
        pidx_i = int(pidx)
        if not (0 <= pidx_i < n):
            continue
        color = palette_rgb.get(sem)
        if color is None:
            # Semantic not in palette -- leave point gray so figure stays clean.
            continue
        colors[pidx_i] = np.asarray(color, dtype=np.float32) / 255.0
        per_class[sem] = per_class.get(sem, 0) + 1
        n_conf += 1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pc_xyz, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return n_conf, per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Chair",
                        help="Object category from PartNetE_meta.json (default: Chair)")
    parser.add_argument("--object-id", default="179",
                        help="Object instance id (default: 179)")
    parser.add_argument("--exp-suffix", default="MaskBackproj",
                        help="Suffix for vis_dir (default: MaskBackproj)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Pipeline config YAML")
    parser.add_argument("--frac-low", type=float, default=0.025,
                        help="Lower bound on mask area / object area (default 0.025)")
    parser.add_argument("--frac-high", type=float, default=0.10,
                        help="Upper bound on mask area / object area (default 0.10)")
    parser.add_argument("--min-unique-3d", type=int, default=200,
                        help="Minimum number of distinct back-projected 3D points "
                             "for a mask to be eligible (default 200)")
    parser.add_argument("--no-mitsuba", action="store_true",
                        help="Skip the Mitsuba ball-grain render of the PLY "
                             "(default: Mitsuba render is produced)")
    parser.add_argument("--mitsuba-elev", type=float, default=CANONICAL_ELEV,
                        help=f"Mitsuba camera elevation, deg (default {CANONICAL_ELEV})")
    parser.add_argument("--mitsuba-azim", type=float, default=CANONICAL_AZIM,
                        help=f"Mitsuba camera azimuth, deg (default {CANONICAL_AZIM})")
    parser.add_argument("--mitsuba-camera-distance", type=float,
                        default=MITSUBA_OVERVIEW_DEFAULTS["camera_distance"],
                        help="Mitsuba camera distance from origin (default 8.0; "
                             "larger = whole object fits with margin)")
    parser.add_argument("--mitsuba-fov", type=float,
                        default=MITSUBA_OVERVIEW_DEFAULTS["fov"],
                        help="Mitsuba camera FOV in degrees (default 18)")
    parser.add_argument("--aspect-min", type=float, default=0.55,
                        help="Minimum bbox W/H ratio for an eligible mask")
    parser.add_argument("--aspect-max", type=float, default=1.80,
                        help="Maximum bbox W/H ratio for an eligible mask")
    parser.add_argument("--extent-min", type=float, default=0.45,
                        help="Minimum mask_area / bbox_area for eligibility")
    parser.add_argument("--solidity-min", type=float, default=0.80,
                        help="Minimum mask_area / convex_hull_area for eligibility")
    parser.add_argument("--no-edge-enhance", action="store_true",
                        help="Skip the pipeline's RGB edge enhancement step "
                             "(produces views that look closer to the raw render)")
    parser.add_argument("--no-mask-process", action="store_true",
                        help="Skip the pipeline's mask de-overlap stage")
    parser.add_argument("--no-anchor-set", action="store_true",
                        help="Skip Stage B (first-pass MLLM + per-point voting "
                             "+ anchor-set render). Stage B costs API tokens "
                             "and a few minutes; disable it when iterating on "
                             "the single-mask Stage A artifacts only.")
    parser.add_argument("--anchor-conf-thresh", type=float,
                        default=ANCHOR_CONFIDENCE_THRESH,
                        help=f"Per-point voting confidence threshold "
                             f"(default {ANCHOR_CONFIDENCE_THRESH}; matches "
                             f"the value pipeline.py 790 passes to "
                             f"TopologyGraphBuilder.build_semantic_point_cloud)")
    parser.add_argument("--anchor-min-votes", type=int,
                        default=ANCHOR_MIN_VOTE_COUNT,
                        help=f"Minimum agreeing-vote count to confirm a 3D "
                             f"point (default {ANCHOR_MIN_VOTE_COUNT}; matches "
                             f"TopologyGraphBuilder default)")
    parser.add_argument("--anchor-mllm-workers", type=int, default=5,
                        help="Parallel views fed to the MLLM during the "
                             "first-pass tagging call (default 5; pipeline "
                             "default).")
    args = parser.parse_args()

    # ----- GPU -----
    physical = _resolve_gpu()
    physical_map = _get_physical_gpu_map()
    print("\n  Physical GPU status (nvidia-smi index):")
    for idx in sorted(physical_map.keys()):
        free_gb, total_gb = physical_map[idx]
        used_pct = (total_gb - free_gb) / total_gb * 100
        marker = "  <-- selected" if idx == physical else ""
        print(f"    GPU {idx}: {free_gb:.1f} / {total_gb:.1f} GB free "
              f"({used_pct:.0f}% used){marker}")
    print()

    import torch
    torch.cuda.set_device(physical)
    logical_gpu = physical

    # ----- Pipeline init -----
    pipeline = Arbitr3DPipeline(args.config, gpu_index=logical_gpu)

    # ----- Load + normalize PC -----
    pc_filename = f"{args.category}/{args.object_id}/pc.ply"
    pcd = pipeline.dataloader.load_point_cloud(pc_filename, num_points=0)
    norm_pcd = pipeline.dataloader.normalize_pc(pcd)
    pc_xyz = np.asarray(norm_pcd.points)
    pc_colors = pipeline.dataloader.extract_color(norm_pcd)
    print(f"[pc] {pc_filename}: {len(pc_xyz)} points, "
          f"colors range [{pc_colors.min():.3f}, {pc_colors.max():.3f}]")

    # ----- Render multi-view (renderer is the same one the pipeline uses) -----
    cam_positions, cam_rotations = generate_sphere_cameras(
        pipeline.config["render"]["num_cameras"],
        radius=pipeline.config["render"]["camera_distance"],
    )
    print(f"[render] {len(cam_positions)} views ...")
    render_output = pipeline.renderer.render(pc_xyz, pc_colors, cam_positions, cam_rotations)

    if not args.no_edge_enhance and pipeline.edge_enhancer.enable_enhancement:
        print("[render] applying RGB edge enhancement (matches pipeline default)")
        render_output.images = pipeline.edge_enhancer.enhance_batch(render_output.images)

    # ----- SAM auto segmentation -----
    print("[sam] generating per-view masks ...")
    masks_across_views = pipeline.sam_auto.generate_masks_automatically(render_output.images)

    # Optional de-overlap (matches pipeline default behavior).
    if not args.no_mask_process:
        algo_cfg = pipeline.config["algorithm"]
        if algo_cfg.get("enable_mask_processing", True):
            min_area = algo_cfg.get("mask_process_min_area", 200)
            cont_thr = algo_cfg.get("mask_process_containment_thresh", 0.95)
            print(f"[sam] de-overlap (min_area={min_area}, cont_thr={cont_thr}) ...")
            cleaned = []
            for masks in masks_across_views:
                if masks:
                    masks = process_masks(masks, containment_threshold=cont_thr,
                                          min_area=min_area)
                    masks = [m for m in masks if int(np.sum(m)) >= min_area]
                cleaned.append(masks)
            masks_across_views = cleaned
    print("[sam] mask counts per view: "
          + ", ".join(str(len(m)) for m in masks_across_views))

    # ----- Pick a medium mask -----
    print(f"[select] looking for mask with area frac in "
          f"[{args.frac_low:.3f}, {args.frac_high:.3f}] and "
          f">= {args.min_unique_3d} unique 3D points ...")
    chosen = _select_medium_mask(
        masks_across_views, render_output.index_maps,
        frac_low=args.frac_low, frac_high=args.frac_high,
        min_unique_3d=args.min_unique_3d,
        aspect_min=args.aspect_min, aspect_max=args.aspect_max,
        extent_min=args.extent_min, solidity_min=args.solidity_min,
    )
    if chosen is None:
        raise RuntimeError(
            "No mask passed all filters. Try widening --frac-low/--frac-high, "
            "--aspect-min/--aspect-max, or lowering --min-unique-3d / "
            "--extent-min / --solidity-min."
        )
    v_idx, m_idx, mask_on_obj, unique_pc, stats = chosen
    print(f"[select] view {v_idx}, mask {m_idx}: area={stats['area']} px "
          f"({stats['area_ratio']*100:.2f}% of object), "
          f"aspect={stats['aspect']:.2f}, extent={stats['extent']:.2f}, "
          f"solidity={stats['solidity']:.2f}, "
          f"unique_3d={stats['unique_3d']}")

    # ----- Outputs -----
    vis_dir = Path(f"visual_res/{args.object_id}_vis_{args.exp_suffix}")
    out_dir = vis_dir / "mask_backproj"
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_id = args.object_id
    cat_lower = args.category.lower()

    # 1. Full 2D view of the SAM image with ONLY the chosen mask highlighted.
    view_path = out_dir / f"{cat_lower}_{obj_id}_view{v_idx:02d}_mask{m_idx:02d}_2d_view.png"
    _render_2d_full_view(
        render_output.images[v_idx], mask_on_obj, view_path,
        color_rgb=HIGHLIGHT_RGB,
    )
    print(f"[out] 2D view -> {view_path}")

    # 2. 3D PLY: keep original PC colors, override only the back-projected
    #    mask points with the vivid highlight color. Pre-subsample with a
    #    stratified scheme so ALL maOPENAI_API_KEY points are kept (Mitsuba
    #    would otherwise random-thin them along with the rest, leaving visible
    #    gaps in the highlighted region).
    target_pc_total = int(MITSUBA_OVERVIEW_DEFAULTS["max_points"])
    pc_xyz_w, pc_colors_w, unique_pc_w = _stratified_subsample(
        pc_xyz, pc_colors, unique_pc, target_total=target_pc_total
    )
    print(f"[ply] stratified subsample: {len(pc_xyz)} -> {len(pc_xyz_w)} points "
          f"(highlight points kept: {len(unique_pc_w)} / {len(unique_pc)})")
    ply_path = out_dir / f"{cat_lower}_{obj_id}_view{v_idx:02d}_mask{m_idx:02d}_3d_color.ply"
    _write_highlighted_ply(pc_xyz_w, pc_colors_w, unique_pc_w, ply_path,
                           highlight_rgb=HIGHLIGHT_RGB)
    print(f"[out] 3D PLY -> {ply_path}")

    # 3. Mitsuba ball-grain render at the canonical (20, 45) view that matches
    #    the paper's overview_thumbs/01_input_pc.png composition. Camera is
    #    pulled back (distance=7.0, fov=18) so the whole chair fits in frame.
    elev = float(args.mitsuba_elev)
    azim = float(args.mitsuba_azim)
    mitsuba_png = None
    if not args.no_mitsuba:
        png_path = ply_path.with_suffix(".png")
        overrides = {
            "camera_distance": float(args.mitsuba_camera_distance),
            "fov": float(args.mitsuba_fov),
        }
        if _maybe_render_mitsuba(ply_path, png_path, elev=elev, azim=azim,
                                 overrides=overrides):
            mitsuba_png = png_path
            print(f"[out] 3D PNG (mitsuba, elev={elev:.1f}, azim={azim:.1f}, "
                  f"dist={overrides['camera_distance']:.1f}) -> {png_path}")
        else:
            print("[out] mitsuba render failed; PLY can still be opened in MeshLab/Blender")

    # 4. Tiny summary JSON for reproducibility
    sam_pos = np.asarray(cam_positions[v_idx], dtype=np.float64)
    sam_r = float(np.linalg.norm(sam_pos))
    sam_elev = float(np.degrees(np.arcsin(sam_pos[1] / sam_r))) if sam_r > 0 else 0.0
    sam_azim = float(np.degrees(np.arctan2(sam_pos[0], sam_pos[2]))) if sam_r > 0 else 0.0

    summary = {
        "category": args.category,
        "object_id": args.object_id,
        "view_idx": int(v_idx),
        "mask_idx": int(m_idx),
        "sam_view_elev_deg": sam_elev,
        "sam_view_azim_deg": sam_azim,
        "mask_stats": {
            "area_px": int(stats["area"]),
            "area_ratio": float(stats["area_ratio"]),
            "bbox_xyxy": list(map(int, stats["bbox"])),
            "aspect_w_over_h": float(stats["aspect"]),
            "extent": float(stats["extent"]),
            "solidity": float(stats["solidity"]),
            "unique_3d_points": int(stats["unique_3d"]),
            "compact_score": float(stats["compact_score"]),
            "centered_score": float(stats["centered_score"]),
            "total_score": float(stats["total_score"]),
        },
        "mitsuba_camera": {
            "elev_deg": elev,
            "azim_deg": azim,
            "distance": float(args.mitsuba_camera_distance),
            "fov": float(args.mitsuba_fov),
        },
        "highlight_rgb": list(HIGHLIGHT_RGB),
        "selection_thresholds": {
            "frac_low": args.frac_low,
            "frac_high": args.frac_high,
            "min_unique_3d": args.min_unique_3d,
            "aspect_min": args.aspect_min,
            "aspect_max": args.aspect_max,
            "extent_min": args.extent_min,
            "solidity_min": args.solidity_min,
        },
        "outputs": {
            "view_png": str(view_path),
            "ply": str(ply_path),
            "mitsuba_png": str(mitsuba_png) if mitsuba_png is not None else None,
        },
    }
    summary_path = out_dir / f"{cat_lower}_{obj_id}_view{v_idx:02d}_mask{m_idx:02d}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[out] summary -> {summary_path}")

    # ============================================================
    # Stage B: first-pass MLLM + per-point voting -> anchor set X^star
    # ============================================================
    if args.no_anchor_set:
        print("\n[anchor-set] skipped (--no-anchor-set)")
        print("\nDone.")
        return

    print("\n" + "=" * 60)
    print("Stage B: anchor-set construction (first-pass MLLM + voting)")
    print("=" * 60)

    # B.1 Prompt classes from PartNetE_meta.json (mirror test.py / batch_test.py).
    meta_path = Path(__file__).parent / "PartNetE_meta.json"
    prompt_classes = _load_prompt_classes(meta_path, args.category)
    print(f"[anchor-set] prompt_classes for {args.category}: {prompt_classes}")

    # B.2 Generalized knowledge K_u (filtered by current instance's parts).
    unified_knowledge = _load_unified_knowledge(args.category, prompt_classes)

    # B.3 view_angles_str (byte-identical to a real pipeline run).
    view_angles_str = _build_view_angles_str(cam_positions)

    # B.4 Initial instance knowledge K_0 (per-mask 3D geometric features).
    print("[anchor-set] building initial instance knowledge K_0 ...")
    initial_instance_knowledge = _build_initial_instance_knowledge(
        pc_xyz, render_output.index_maps, masks_across_views
    )

    # B.5 First-pass MLLM tagging across all views.
    #
    # `predict_soft_probabilities_all_views` has gained a few kwargs across
    # versions (e.g. `disable_height_prior` was added later). To keep the
    # script working on whichever copy the user has checked out, filter our
    # kwargs through `inspect.signature` and take only the first element of
    # the return tuple (older versions return a 3-tuple, newer versions
    # return a 4-tuple; we only need `soft_masks_across_views`).
    import inspect as _inspect

    n_masks_total = sum(len(m) for m in masks_across_views)
    print(f"[anchor-set] calling first-pass MLLM on "
          f"{n_masks_total} masks across {len(masks_across_views)} views "
          f"(this can take several minutes) ...")

    _mllm_fn = pipeline.mllm_classifier.predict_soft_probabilities_all_views
    _mllm_kwargs = {
        "images_rgb": render_output.images,
        "depth_maps": render_output.depth_maps,
        "masks_across_views": masks_across_views,
        "prompt_classes": prompt_classes,
        "object_category": args.category,
        "view_angles_str": view_angles_str,
        "unified_knowledge": unified_knowledge,
        "initial_instance_knowledge": initial_instance_knowledge,
        "pose_assessment": None,
        "max_workers": int(args.anchor_mllm_workers),
        "disable_height_prior": False,
    }
    try:
        _accepted = set(_inspect.signature(_mllm_fn).parameters.keys())
        _dropped = [k for k in _mllm_kwargs if k not in _accepted]
        if _dropped:
            print(f"  [compat] dropping kwargs not accepted by this version: "
                  f"{_dropped}")
        _mllm_kwargs = {k: v for k, v in _mllm_kwargs.items() if k in _accepted}
    except (TypeError, ValueError):
        # Fall back to passing everything if signature introspection fails.
        pass

    _mllm_result = _mllm_fn(**_mllm_kwargs)
    soft_masks_across_views = (
        _mllm_result[0] if isinstance(_mllm_result, (tuple, list)) else _mllm_result
    )

    # B.6 Per-point voting via the same TopologyGraphBuilder pipeline.py uses.
    print(f"[anchor-set] aggregating per-point votes "
          f"(conf>={args.anchor_conf_thresh:.2f}, votes>={args.anchor_min_votes}) ...")
    topo_builder = TopologyGraphBuilder(
        pc_xyz=pc_xyz,
        index_maps=render_output.index_maps,
        unified_knowledge=unified_knowledge,
        object_category=args.category,
    )
    point_semantics = topo_builder.build_semantic_point_cloud(
        soft_masks_across_views,
        confidence_threshold=float(args.anchor_conf_thresh),
        min_vote_count=int(args.anchor_min_votes),
    )

    # B.7 Write the full-resolution anchor PLY.
    #
    # We deliberately do NOT pre-subsample here. The anchor set typically
    # covers 50-70% of all points; if we kept "every anchor + budget remainder"
    # the rendered 10 k subsample would be ~100% colored and the figure would
    # lose its "only some points are highlighted" look. Instead we save the
    # full PC and let render_balls_mitsuba's internal `--max-points 10000`
    # uniform subsample preserve the anchor:gray ratio (matches how
    # 08_final_seg.png is rendered from `final_result.ply`).
    anchor_ply = out_dir / f"{cat_lower}_{obj_id}_anchor_set_3d.ply"
    n_conf_total, per_class_counts = _write_anchor_set_ply(
        pc_xyz, point_semantics, anchor_ply
    )
    pc_total = len(pc_xyz)
    print(f"[anchor-set] confirmed {n_conf_total} / {pc_total} pts "
          f"({100.0 * n_conf_total / max(1, pc_total):.1f}% of full PC; "
          f"mitsuba's --max-points 10000 will uniform-subsample at render time)")
    for cls in sorted(per_class_counts.keys()):
        print(f"    {cls:8s}: {per_class_counts[cls]}")
    print(f"[out] anchor PLY -> {anchor_ply}")

    # B.9 Mitsuba ball-grain render at the canonical (20, 45) view to match
    #     overview_thumbs/08_final_seg.png stylistically.
    anchor_png = anchor_ply.with_suffix(".png")
    mitsuba_anchor_png: Optional[Path] = None
    if not args.no_mitsuba:
        anchor_overrides = {
            "camera_distance": float(args.mitsuba_camera_distance),
            "fov": float(args.mitsuba_fov),
        }
        if _maybe_render_mitsuba(
            anchor_ply, anchor_png, elev=elev, azim=azim,
            overrides=anchor_overrides,
        ):
            mitsuba_anchor_png = anchor_png
            print(f"[out] anchor PNG (mitsuba, elev={elev:.1f}, "
                  f"azim={azim:.1f}) -> {anchor_png}")
        else:
            print("[out] mitsuba anchor render failed; PLY still saved.")
    else:
        print("[anchor-set] mitsuba render skipped (--no-mitsuba)")

    # B.10 Anchor-set summary JSON.
    anchor_summary = {
        "category": args.category,
        "object_id": args.object_id,
        "n_views": len(masks_across_views),
        "n_masks_total": int(n_masks_total),
        "n_points_pc": int(pc_total),
        "n_anchor_total": int(n_conf_total),
        "anchor_per_class": {k: int(v) for k, v in sorted(per_class_counts.items())},
        "anchor_thresholds": {
            "confidence_threshold": float(args.anchor_conf_thresh),
            "min_vote_count": int(args.anchor_min_votes),
        },
        "prompt_classes": prompt_classes,
        "palette_rgb_saturated": {k: list(v) for k, v in PART_RGB_SATURATED.items()},
        "gray_rgb_unconfirmed": list(GRAY_RGB_UNCONFIRMED),
        "mitsuba_camera": {
            "elev_deg": float(elev),
            "azim_deg": float(azim),
            "distance": float(args.mitsuba_camera_distance),
            "fov": float(args.mitsuba_fov),
        },
        "outputs": {
            "anchor_ply": str(anchor_ply),
            "anchor_png": str(mitsuba_anchor_png) if mitsuba_anchor_png else None,
        },
    }
    anchor_summary_path = out_dir / f"{cat_lower}_{obj_id}_anchor_set_summary.json"
    with open(anchor_summary_path, "w", encoding="utf-8") as f:
        json.dump(anchor_summary, f, indent=2, ensure_ascii=False)
    print(f"[out] anchor summary -> {anchor_summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

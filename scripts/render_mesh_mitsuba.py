"""Render PartObjaverse-Tiny meshes as continuous shaded surfaces (Mitsuba 3).

Drop-in companion to ``render_balls_mitsuba.py``: instead of rendering the
predicted point cloud as small spheres, this script loads the original GLB
mesh, transfers part labels onto the mesh faces (either from PartObjaverse-Tiny
GT ``.npy`` annotations or by voting from a predicted point cloud), splits the
mesh by label into per-part submeshes, and path-traces each submesh with a
saturated diffuse-plastic BSDF. Soft area lighting + a ground shadow catcher
reproduce the "silky surface" look standard in qualitative figures
(SAMPart3D, PartField, PartObjaverse-Tiny paper, ...).

Instance-selection interface
----------------------------
The CLI deliberately accepts multiple ways to specify which instances to
render so you can mix interactive single-instance work with batch jobs:

    # 1) Single instance (most common)
    python scripts/render_mesh_mitsuba.py \
        --label-source gt \
        --category Food --object-id 00aee5c2fef743d69421bb642d446a5b

    # 2) All instances of one category
    python scripts/render_mesh_mitsuba.py --label-source gt --category Food

    # 3) Multiple categories
    python scripts/render_mesh_mitsuba.py --label-source gt \
        --categories Food Animals "Daily-Used"

    # 4) Hand-picked list from a text file
    #    File format: one "<category>/<obj_id>" per line; lines starting with
    #    '#' or empty lines are ignored. Bare "<obj_id>" lines are also OK
    #    if --category is supplied as a fallback.
    python scripts/render_mesh_mitsuba.py --label-source gt \
        --instances-file scripts/render_mesh_picks.txt

    # 5) Render every instance of every category (slow!)
    python scripts/render_mesh_mitsuba.py --label-source gt --all

    # 6) Limit / shard for parallel jobs
    python scripts/render_mesh_mitsuba.py --label-source gt --all \
        --start-index 0 --limit 16

    # 7) Render predictions instead of GT (resolves
    #    {pred_root}/{category}/{obj_id}/step4_final/{final_result.ply,
    #    ins_sem_seg.npz}; auto-falls back to {pred_root}/{obj_id}_vis_*).
    python scripts/render_mesh_mitsuba.py \
        --label-source pred \
        --pred-root visual_res/batch_objaverse_eval \
        --categories Food Animals

Outputs are written to ``--output-root``, mirroring the input layout::

    {output_root}/{category}/{obj_id}__{label_source}_e{elev}_a{azim}.png
    {output_root}/{category}/{obj_id}__{label_source}.json   (metadata)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Reuse the Mitsuba helpers from render_balls_mitsuba.py (same scripts/ folder).
# Importing first selects the variant via ``_select_variant`` below.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# ---------------------------------------------------------------------------
# sys.path surgery
# ---------------------------------------------------------------------------
# When a user runs ``python scripts/render_mesh_mitsuba.py`` Python
# *automatically* prepends the script's parent directory (``scripts/``) to
# ``sys.path[0]`` before our code starts. That breaks the pipeline runner
# because ``scripts/config.py`` (a real module used by knowledge-generation
# scripts) silently shadows the project-root ``config/`` *namespace* package
# that ``pipeline.py`` / ``module_2d_fm`` depend on (
# ``from config.feature_mapping import ...``). Regular modules always win
# over namespace package portions, irrespective of sys.path order, so
# *just adding PROJECT_ROOT first is not enough* -- we must actively remove
# the auto-added scripts dir entry. Then we re-add PROJECT_ROOT and import
# render_balls_mitsuba via the ``scripts`` namespace package.
sys.path[:] = [
    p for p in sys.path
    if not p or Path(p).resolve() != SCRIPT_DIR
]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_balls_mitsuba import (  # noqa: E402
    _select_variant,
    _look_at,
    _rotate,
    _scale,
    _translate,
    PALETTES,
    PALETTE_GRAY,
    adjust_palette,
    cam_position,
    composite_over_bg,
    crop_to_content,
    render_view,
)

# These are deferred to keep import cheap; loaded lazily on first use.
import mitsuba as mi  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults (mirrors test_objaverse.py / batch_test_objaverse.py)
# ---------------------------------------------------------------------------
DEFAULT_BASE_PATH = "datasets/PartObjaverse-Tiny/PartObjaverse-Tiny"
DEFAULT_MESH_SUBDIR = "PartObjaverse-Tiny_mesh"
DEFAULT_GT_SUBDIR = "PartObjaverse-Tiny_semantic_gt"
DEFAULT_META_JSON = "PartObjaverse-Tiny_semantic.json"
DEFAULT_PIPELINE_VIS_TEMPLATE = "outputs/render_mesh_pipeline/{category}/{obj_id}"
DEFAULT_PIPELINE_NUM_POINTS = 30000

# ---- Module-level singletons used by the optional pipeline-runner path. ----
_PIPELINE_SINGLETON = None  # type: ignore[var-annotated]
_PIPELINE_TMP_CONFIG_PATH: Optional[Path] = None


# ===========================================================================
# Mesh + label loading
# ===========================================================================
def load_concatenated_mesh(path: Path):
    """Load a GLB/OBJ as a single ``trimesh.Trimesh`` (concatenate scene if any).

    Mirrors the loading convention used by ``test_objaverse._remap_gt`` so that
    GT face indices line up with the dumped mesh's face order.
    """
    import trimesh
    raw = trimesh.load(str(path))
    if isinstance(raw, trimesh.Scene):
        geom_list = [
            g for g in raw.geometry.values()
            if isinstance(g, trimesh.Trimesh) and g.faces is not None and len(g.faces) > 0
        ]
        if not geom_list:
            raise RuntimeError(f"No valid Trimesh geometry in {path}")
        return raw.dump(concatenate=True)
    if not hasattr(raw, "faces") or raw.faces is None or len(raw.faces) == 0:
        raise RuntimeError(f"mesh has no faces: {path}")
    return raw


def normalize_mesh_inplace(mesh) -> Tuple[np.ndarray, float]:
    """Center mesh on origin and scale max radius -> 1. Returns (centroid, scale)."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    radius = float(np.linalg.norm(centered, axis=1).max())
    if radius <= 1e-12:
        return centroid, 1.0
    mesh.vertices = centered / radius
    return centroid, radius


def load_gt_face_palette_idx(
    npy_path: Path,
    mesh,
    n_part_names: int,
) -> np.ndarray:
    """Load PartObjaverse-Tiny GT labels and return ``(n_faces,)`` palette indices.

    The official ``.npy`` files store either per-face or per-vertex integer
    labels indexing into the per-instance ``part_names`` list (i.e. the
    positional palette). We auto-detect length, fold per-vertex labels into
    per-face by majority of the 3 corner labels, and clip out-of-range / -1
    labels to ``-1`` (rendered as gray).
    """
    raw = np.load(str(npy_path))
    raw = np.asarray(raw).astype(np.int64).ravel()
    n_faces = int(len(mesh.faces))
    n_verts = int(len(mesh.vertices))

    if len(raw) == n_faces:
        face_labels = raw
    elif len(raw) == n_verts:
        # Vertex labels -> majority vote across the 3 corners of each face.
        v_labels = raw[mesh.faces]  # (n_faces, 3)
        face_labels = np.empty(n_faces, dtype=np.int64)
        for fid in range(n_faces):
            row = v_labels[fid]
            # Counter handles ties deterministically by first-seen order.
            face_labels[fid] = Counter(row.tolist()).most_common(1)[0][0]
    else:
        raise ValueError(
            f"GT length {len(raw)} matches neither n_faces={n_faces} nor "
            f"n_verts={n_verts}; cannot map to per-face labels."
        )

    # Treat anything outside [0, n_part_names) as background -> -1 (gray).
    face_labels = face_labels.copy()
    out_of_range = (face_labels < 0) | (face_labels >= n_part_names)
    face_labels[out_of_range] = -1
    return face_labels


def load_pred_pointcloud_with_labels(
    pred_dir: Path,
    part_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the predicted PC and per-point part-name-space labels.

    Searches ``pred_dir`` (and ``pred_dir/step4_final``) for the canonical
    ``ins_sem_seg.npz`` (containing ``sem_label`` + optional ``class_mapping``)
    plus the matching colored PLY. Falls back to color-distance matching
    against the saturated 'arbitr3d' palette if no labels file is found.

    Returns
    -------
    pred_xyz : (N, 3) float64
    pred_palette_idx : (N,) int64 in {-1, 0, ..., len(part_names)-1}
    """
    import open3d as o3d

    # ---- locate npz / ply ---------------------------------------------------
    candidates = [pred_dir, pred_dir / "step4_final"]
    ply_path = None
    npz_path = None
    for cand in candidates:
        if not cand.is_dir():
            continue
        for name in ("final_result.ply", "pred.ply"):
            p = cand / name
            if p.is_file():
                ply_path = p
                break
        if ply_path is None:
            # Any .ply that starts with "pred_" or just one .ply in dir
            plys = sorted(cand.glob("pred_*.ply"))
            if not plys:
                plys = sorted(cand.glob("*.ply"))
            if plys:
                ply_path = plys[0]
        for name in ("ins_sem_seg.npz",):
            p = cand / name
            if p.is_file():
                npz_path = p
                break
        if ply_path is not None:
            break

    if ply_path is None:
        raise FileNotFoundError(
            f"No predicted .ply found under {pred_dir} (looked for "
            f"final_result.ply / pred*.ply in dir and step4_final/)."
        )

    # ---- load PLY -----------------------------------------------------------
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pred_xyz = np.asarray(pcd.points, dtype=np.float64)
    pred_rgb = np.asarray(pcd.colors, dtype=np.float64)
    if pred_xyz.size == 0:
        raise RuntimeError(f"empty PLY: {ply_path}")

    # ---- map labels -> palette index in part-name space ---------------------
    palette_idx = np.full(pred_xyz.shape[0], -1, dtype=np.int64)
    name_to_palette = {n: i for i, n in enumerate(part_names)}

    if npz_path is not None:
        data = np.load(str(npz_path), allow_pickle=True)
        if "sem_label" not in data:
            raise KeyError(f"{npz_path} has no 'sem_label' key")
        sem_label = np.asarray(data["sem_label"], dtype=np.int64)
        if sem_label.shape[0] != pred_xyz.shape[0]:
            raise RuntimeError(
                f"sem_label size ({sem_label.shape[0]}) != PLY points "
                f"({pred_xyz.shape[0]}) for {ply_path}"
            )

        if "class_mapping" in data.files:
            try:
                cm = dict(data["class_mapping"].tolist())
            except Exception:
                cm = {}
            id_to_name = {int(v): str(k) for k, v in cm.items()}
        else:
            # Convention: pipeline emits the part_names first, then unlabeled,
            # then background, in positional order.
            id_to_name = {i: name for i, name in enumerate(part_names)}

        for raw_id in np.unique(sem_label):
            name = id_to_name.get(int(raw_id))
            if name is None:
                continue
            if name in name_to_palette:
                palette_idx[sem_label == raw_id] = name_to_palette[name]
            # else: 'unlabeled' / 'background' -> stay -1
        print(f"  [pred-load] {ply_path.name} + {npz_path.name}: "
              f"{int((palette_idx >= 0).sum())} labeled / "
              f"{int((palette_idx < 0).sum())} background points")
    else:
        # Fall back: match per-point RGB to the saturated 'arbitr3d' palette
        # in part-name positional order.
        pal = PALETTES["arbitr3d"][:len(part_names)]  # (K, 3) in [0, 1]
        if pred_rgb.size == 0:
            raise RuntimeError(
                f"PLY {ply_path} has no per-vertex colors and no "
                f"ins_sem_seg.npz to recover labels."
            )
        # Distance to each palette color; threshold at 0.15 (saturated palette
        # is well separated, so anything within 0.15 of a palette color is
        # confidently that part).
        dists = np.linalg.norm(
            pred_rgb[:, None, :] - pal[None, :, :], axis=-1
        )  # (N, K)
        nearest = np.argmin(dists, axis=1)
        nearest_dist = np.min(dists, axis=1)
        palette_idx = np.where(nearest_dist < 0.15, nearest, -1).astype(np.int64)
        print(f"  [pred-load] {ply_path.name}: no npz found, recovered labels "
              f"from PLY colors via palette match "
              f"({int((palette_idx >= 0).sum())} labeled / "
              f"{int((palette_idx < 0).sum())} background)")

    return pred_xyz, palette_idx


def transfer_pred_to_faces(
    mesh,
    pred_xyz: np.ndarray,
    pred_palette_idx: np.ndarray,
    samples_per_face: int = 3,
    min_total_samples: int = 80000,
) -> np.ndarray:
    """Transfer per-point palette indices onto the mesh's per-face labels.

    Densely samples the mesh proportional to face area (with face_idx tracked),
    KDTree-queries each sample against the prediction PC, then majority-votes
    per face.

    Both the mesh samples and the prediction cloud are normalized to a unit
    sphere independently before the KDTree query. This works because (a) the
    pipeline runs on a normalized PC sampled from the same mesh, so the two
    point sets share scale up to a tiny re-centering offset, and (b) the
    nearest-neighbor query is robust to that offset.
    """
    import open3d as o3d

    n_faces = int(len(mesh.faces))
    n_samples = max(samples_per_face * n_faces, min_total_samples)
    samples, face_idx = mesh.sample(n_samples, return_index=True)
    samples = np.asarray(samples, dtype=np.float64)
    face_idx = np.asarray(face_idx, dtype=np.int64)

    # Independent unit-sphere normalisation (matches pipeline.normalize_pc).
    def _unit_sphere(pts: np.ndarray) -> np.ndarray:
        c = pts.mean(axis=0)
        d = pts - c
        r = float(np.linalg.norm(d, axis=1).max())
        return d / r if r > 1e-12 else d

    samples_n = _unit_sphere(samples)
    pred_n = _unit_sphere(pred_xyz)

    # KDTree on prediction cloud; nearest-neighbor lookup for each sample.
    ref = o3d.geometry.PointCloud()
    ref.points = o3d.utility.Vector3dVector(pred_n)
    tree = o3d.geometry.KDTreeFlann(ref)
    sample_palette_idx = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        _, idx, _ = tree.search_knn_vector_3d(samples_n[i], 1)
        sample_palette_idx[i] = pred_palette_idx[idx[0]]

    # Majority vote per face (linear-time via sorted groupby).
    face_labels = np.full(n_faces, -1, dtype=np.int64)
    order = np.argsort(face_idx, kind="stable")
    sf = face_idx[order]
    sl = sample_palette_idx[order]
    if n_samples == 0:
        return face_labels
    boundaries = np.concatenate(([0], np.where(np.diff(sf) != 0)[0] + 1, [n_samples]))
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        fid = int(sf[b0])
        if 0 <= fid < n_faces:
            chunk = sl[b0:b1]
            face_labels[fid] = Counter(chunk.tolist()).most_common(1)[0][0]
    return face_labels


# ===========================================================================
# Optional pipeline runner: regenerate prediction outputs on the fly
# ---------------------------------------------------------------------------
# Older `batch_test_objaverse.py` runs only saved the colored
# ``step4_final/final_result.ply`` and forgot the matching
# ``ins_sem_seg.npz``. Without integer labels we have to fall back to
# color-distance matching, which is heuristic and tends to drop ~5% of
# points to 'unlabeled'. To get a clean render this script can re-invoke
# Arbitr3DPipeline on demand and persist both files itself.
# ===========================================================================
def _has_pred_npz(pred_dir: Path) -> bool:
    """True if ``pred_dir`` (or its ``step4_final/`` subdir) has ins_sem_seg.npz."""
    for sub in (".", "step4_final"):
        if (pred_dir / sub / "ins_sem_seg.npz").is_file():
            return True
    return False


def _save_pipeline_npz(
    vis_dir: Path,
    final_labels: np.ndarray,
    class_mapping: dict,
) -> Path:
    """Persist ``sem_label`` + ``class_mapping`` next to ``final_result.ply``.

    Mirrors the convention used by ``test.py`` for PartNetE so downstream
    rendering tools (this script + ``render_qualitative_4x6`` +
    ``render_balls_mitsuba``) can recolor by label without guessing.
    """
    out_dir = vis_dir / "step4_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "ins_sem_seg.npz"
    np.savez(
        npz_path,
        sem_label=np.asarray(final_labels, dtype=np.int64),
        class_mapping=np.array(list(class_mapping.items()), dtype=object),
    )
    return npz_path


def _init_pipeline(args, base_path: Path):
    """Lazily initialize ``Arbitr3DPipeline``; cached for subsequent calls.

    The pipeline init is heavy (loads SAM, BERT, MLLM clients, ...), so we
    keep one instance for the whole batch. Re-uses ``--config`` /
    ``config/config.yaml`` but force-overrides ``data.base_path`` to point at
    the resolved PartObjaverse-Tiny root.
    """
    global _PIPELINE_SINGLETON, _PIPELINE_TMP_CONFIG_PATH
    if _PIPELINE_SINGLETON is not None:
        return _PIPELINE_SINGLETON

    import yaml  # type: ignore
    config_path = (
        Path(args.config) if args.config
        else PROJECT_ROOT / "config" / "config.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Pipeline run requested but config not found at {config_path}. "
            f"Pass --config /path/to/config.yaml."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("data", {})
    cfg["data"]["base_path"] = str(base_path)
    cfg["data"]["dataset_name"] = "PartObjaverse-Tiny"

    tmp_path = PROJECT_ROOT / "config" / "_tmp_render_mesh_pipeline_config.yaml"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    _PIPELINE_TMP_CONFIG_PATH = tmp_path

    # GPU: a launcher that sets CUDA_VISIBLE_DEVICES gives us logical 0;
    # otherwise honour --gpu. Default -1 -> logical 0 too.
    gpu_index = int(args.gpu) if int(args.gpu) >= 0 else 0
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        gpu_index = 0
    try:
        import torch  # type: ignore
        torch.cuda.set_device(gpu_index)
    except Exception as e:  # noqa: BLE001
        print(f"[render-mesh] WARN: torch.cuda.set_device({gpu_index}) failed: {e}")

    print(f"[render-mesh] initializing Arbitr3DPipeline "
          f"(gpu={gpu_index}, config={tmp_path})")
    from pipeline import Arbitr3DPipeline  # type: ignore
    _PIPELINE_SINGLETON = Arbitr3DPipeline(str(tmp_path), gpu_index=gpu_index)
    return _PIPELINE_SINGLETON


def run_pipeline_for_instance(
    category: str,
    obj_id: str,
    part_names: List[str],
    args,
    base_path: Path,
) -> Path:
    """Run Arbitr3DPipeline on one instance and persist the rendering inputs.

    Writes ``{vis_dir}/step4_final/final_result.ply`` (via the pipeline's
    own visualizer) and ``{vis_dir}/step4_final/ins_sem_seg.npz`` (saved
    here). Returns ``vis_dir`` so it can be passed straight into
    ``load_pred_pointcloud_with_labels``.
    """
    pipeline = _init_pipeline(args, base_path)

    vis_dir = Path(
        args.pipeline_vis_template.format(category=category, obj_id=obj_id)
    )
    vis_dir.mkdir(parents=True, exist_ok=True)

    mesh_rel = f"{args.mesh_subdir}/{obj_id}.glb"  # relative to base_path
    prompt_classes_str = ", ".join(part_names)
    num_points = (
        int(args.num_points) if args.num_points is not None
        else int(DEFAULT_PIPELINE_NUM_POINTS)
    )

    print(f"  [pipeline] running on {category}/{obj_id} "
          f"(num_points={num_points}, vis_dir={vis_dir})")

    pcd, final_labels, class_mapping = pipeline.run(
        point_cloud_filename=mesh_rel,
        prompt_classes_str=prompt_classes_str,
        vis_dir=str(vis_dir),
        object_category=category,
        num_points=num_points,
    )

    # The pipeline already wrote step4_final/final_result.ply via its
    # internal Visualizer3D; we just add the matching label npz so
    # load_pred_pointcloud_with_labels can use exact integer labels.
    npz_path = _save_pipeline_npz(vis_dir, final_labels, class_mapping)
    print(f"  [pipeline] wrote {npz_path}")

    return vis_dir


# ===========================================================================
# Submesh export
# ===========================================================================

def export_per_part_submeshes(
    mesh,
    face_palette_idx: np.ndarray,
    palette: np.ndarray,
    gray: np.ndarray,
    out_dir: Path,
    show_unlabeled_as_gray: bool = True,
) -> List[Tuple[Path, Tuple[float, float, float]]]:
    """Split mesh by face label, write each submesh as PLY, return colors.

    Each unique label gets its own ``.ply`` so the Mitsuba scene can apply a
    flat per-part BSDF with no per-vertex attribute trickery. Faces labelled
    -1 are emitted as a single 'gray' submesh (so the chair's geometry stays
    visually complete) when ``show_unlabeled_as_gray`` is True; otherwise
    they are dropped.
    """
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[Path, Tuple[float, float, float]]] = []
    unique_labels = np.unique(face_palette_idx)
    for label in sorted(unique_labels.tolist()):
        face_mask = face_palette_idx == label
        if not face_mask.any():
            continue
        if label < 0:
            if not show_unlabeled_as_gray:
                continue
            color = tuple(float(c) for c in gray)
            tag = "unlabeled"
        else:
            if label >= len(palette):
                # Past palette length: cycle (rare, only if K > len(palette)).
                color = tuple(float(c) for c in palette[label % len(palette)])
            else:
                color = tuple(float(c) for c in palette[label])
            tag = f"part_{int(label):02d}"

        sub = mesh.submesh([np.where(face_mask)[0]], append=True)
        if sub is None or len(sub.faces) == 0:
            continue
        # Force vertex normals so Mitsuba does smooth (not flat) shading.
        _ = sub.vertex_normals  # trimesh computes lazily and caches.
        ply_path = out_dir / f"{tag}.ply"
        sub.export(str(ply_path))
        items.append((ply_path, color))
    return items


# ===========================================================================
# Mitsuba scene builder (mesh path)
# ===========================================================================
def build_mesh_scene_dict(
    submeshes: List[Tuple[Path, Tuple[float, float, float]]],
    cam_pos: List[float],
    *,
    width: int,
    height: int,
    spp: int,
    bg_color: Tuple[float, float, float],
    fov_deg: float = 22.0,
    ground_y: float = -1.05,
    light_strength: float = 6.0,
    fill_strength: float = 0.45,
    transparent_bg: bool = True,
    bsdf_alpha: float = 0.10,
    int_ior: float = 1.46,
) -> dict:
    """Build a Mitsuba 3 scene dict with one PLY shape per part submesh.

    Lighting / sensor / ground catcher mirror ``render_balls_mitsuba``'s look
    so the mesh-rendered figures sit comfortably next to the ball-rendered
    intermediates in a paper figure row.
    """
    scene: dict = {
        "type": "scene",
        "integrator": {
            "type": "path",
            "max_depth": 8,
            "rr_depth": 4,
            "hide_emitters": True,
        },
        "sensor": {
            "type": "perspective",
            "fov": fov_deg,
            "fov_axis": "smaller",
            "to_world": _look_at(cam_pos),
            "film": {
                "type": "hdrfilm",
                "width": int(width),
                "height": int(height),
                "rfilter": {"type": "gaussian"},
                "pixel_format": "rgba",
                "sample_border": True,
            },
            "sampler": {"type": "independent", "sample_count": int(spp)},
        },
        "key_light": {
            "type": "rectangle",
            "to_world": (
                _look_at((2.0, 3.0, 2.5))
                @ _scale((1.4, 1.4, 1.0))
            ),
            "bsdf": {"type": "null"},
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": [light_strength, light_strength, light_strength],
                },
            },
        },
    }

    if transparent_bg:
        scene["fill_light"] = {
            "type": "rectangle",
            "to_world": (
                _look_at((-2.0, 1.5, -1.5))
                @ _scale((1.5, 1.5, 1.0))
            ),
            "bsdf": {"type": "null"},
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": [
                        light_strength * 0.35,
                        light_strength * 0.35,
                        light_strength * 0.35,
                    ],
                },
            },
        }
    else:
        scene["envmap"] = {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": [
                    bg_color[0] * fill_strength,
                    bg_color[1] * fill_strength,
                    bg_color[2] * fill_strength,
                ],
            },
        }
        scene["ground"] = {
            "type": "rectangle",
            "to_world": (
                _translate((0.0, ground_y, 0.0))
                @ _rotate((1.0, 0.0, 0.0), -90.0)
                @ _scale((6.0, 6.0, 1.0))
            ),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": list(bg_color)},
            },
        }

    # One PLY shape per part submesh, each with a flat roughplastic BSDF.
    for i, (ply_path, color) in enumerate(submeshes):
        scene[f"part_{i}"] = {
            "type": "ply",
            "filename": str(ply_path),
            "face_normals": False,  # smooth shading via vertex normals
            "bsdf": {
                "type": "roughplastic",
                "diffuse_reflectance": {
                    "type": "rgb",
                    "value": [float(color[0]), float(color[1]), float(color[2])],
                },
                "alpha": float(bsdf_alpha),
                "int_ior": float(int_ior),
            },
        }

    return scene


# ===========================================================================
# Instance resolver
# ===========================================================================
def _load_partobjaverse_config(config_path: Path) -> dict:
    """Read the ``partobjaverse_tiny`` section out of ``config/config.yaml``.

    Returns an empty dict if the file is missing or unreadable; warns the
    user so they know whether their CLI flags actually overrode anything.
    """
    if not config_path.is_file():
        print(f"[render-mesh] config not found at {config_path}; "
              f"falling back to hardcoded defaults.")
        return {}
    try:
        import yaml  # type: ignore
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        pot = cfg.get("partobjaverse_tiny", {}) or {}
        if pot:
            print(f"[render-mesh] using partobjaverse_tiny config from "
                  f"{config_path}")
        return pot
    except Exception as e:  # noqa: BLE001
        print(f"[render-mesh] WARN: failed to read {config_path}: {e}")
        return {}


def load_meta(meta_rel: str, base_path: Path) -> Tuple[Path, dict]:
    """Locate and load the PartObjaverse-Tiny semantic JSON.

    Search order, mirroring ``test_objaverse.py`` /
    ``generate_objaverse_knowledge.py`` (which keep the JSON next to the code
    so the model card lives in the repo, not in /data/...):

      1. If ``meta_rel`` is absolute -> use as-is.
      2. ``PROJECT_ROOT / meta_rel``  (the canonical checked-in copy).
      3. ``base_path / meta_rel``     (a copy that ships with the dataset).
    """
    p = Path(meta_rel)
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [PROJECT_ROOT / p, base_path / p]
    for c in candidates:
        if c.is_file():
            with open(c, "r", encoding="utf-8") as f:
                return c, json.load(f)
    raise FileNotFoundError(
        "Could not locate PartObjaverse-Tiny_semantic.json. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nFix options:"
        + "\n  - copy the JSON into the project root (it is checked-in for the"
        + " Windows workspace), or"
        + "\n  - pass --meta-json /absolute/path/to/PartObjaverse-Tiny_semantic.json,"
        + " or"
        + "\n  - point partobjaverse_tiny.meta_json in config/config.yaml at"
        + " an existing file (relative to project root or to base_path)."
    )


def _read_instances_file(path: Path, fallback_category: Optional[str]) -> List[Tuple[str, str]]:
    """Parse a list-of-instances text file. See the module docstring."""
    out: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "/" in line:
                cat, oid = line.split("/", 1)
                cat, oid = cat.strip(), oid.strip()
                if not cat or not oid:
                    raise ValueError(f"{path}:{ln_no} malformed: {raw!r}")
                out.append((cat, oid))
            else:
                if not fallback_category:
                    raise ValueError(
                        f"{path}:{ln_no} has bare obj_id {line!r} but no "
                        f"--category fallback was given."
                    )
                out.append((fallback_category, line))
    return out


def resolve_instances(meta: dict, args) -> List[Tuple[str, str, List[str]]]:
    """Turn CLI flags into a concrete list of (category, obj_id, part_names).

    Resolution order:
      1. --instances-file (highest precedence)
      2. --object-id (requires --category)
      3. --all
      4. --categories / --category (all instances under given categor(ies))

    After resolution, applies --start-index and --limit as a slicing window.
    """
    pairs: List[Tuple[str, str]] = []
    if args.instances_file:
        pairs = _read_instances_file(Path(args.instances_file), args.category)
    elif args.object_id:
        if not args.category:
            raise ValueError("--object-id requires --category")
        pairs = [(args.category, args.object_id)]
    elif args.all:
        for cat, instances in meta.items():
            for oid in sorted(instances.keys()):
                pairs.append((cat, oid))
    elif args.categories or args.category:
        cats = list(args.categories) if args.categories else [args.category]
        for cat in cats:
            if cat not in meta:
                print(f"[resolve] WARN: category {cat!r} not in meta; skipping")
                continue
            for oid in sorted(meta[cat].keys()):
                pairs.append((cat, oid))
    else:
        raise ValueError(
            "No instance specified. Use one of: --object-id, --category, "
            "--categories, --instances-file, --all."
        )

    # Look up part_names for each pair, drop unknowns.
    resolved: List[Tuple[str, str, List[str]]] = []
    for cat, oid in pairs:
        if cat not in meta:
            print(f"[resolve] WARN: category {cat!r} not in meta; dropping {oid}")
            continue
        if oid not in meta[cat]:
            print(f"[resolve] WARN: {cat}/{oid} not in meta; dropping")
            continue
        resolved.append((cat, oid, list(meta[cat][oid])))

    # Apply windowing.
    s = max(0, int(args.start_index))
    if args.limit and args.limit > 0:
        resolved = resolved[s:s + int(args.limit)]
    elif s > 0:
        resolved = resolved[s:]
    return resolved


# ===========================================================================
# Per-instance render driver
# ===========================================================================
def render_one(
    category: str,
    obj_id: str,
    part_names: List[str],
    *,
    args,
    palette: np.ndarray,
    gray: np.ndarray,
    label_source: Optional[str] = None,
) -> Optional[Path]:
    """Render every requested view for one instance. Returns the output dir.

    ``label_source`` overrides ``args.label_source`` for this call (used when
    --label-source both expands one instance into two render passes). When
    None, falls back to args.label_source.
    """
    if label_source is None:
        label_source = args.label_source
    base_path = Path(args.base_path)
    mesh_path = base_path / args.mesh_subdir / f"{obj_id}.glb"
    if not mesh_path.is_file():
        print(f"  SKIP: mesh not found {mesh_path}")
        return None

    print(f"  mesh: {mesh_path}")
    mesh = load_concatenated_mesh(mesh_path)
    print(f"  -> {len(mesh.vertices)} verts / {len(mesh.faces)} faces")

    # ---------------------------------------------------------------- labels
    if label_source == "gt":
        npy_path = base_path / args.gt_subdir / f"{obj_id}.npy"
        if not npy_path.is_file():
            print(f"  SKIP: GT not found {npy_path}")
            return None
        face_palette_idx = load_gt_face_palette_idx(npy_path, mesh, len(part_names))
        label_tag = "gt"
    elif label_source == "pred":
        # Build the list of prediction-dir candidates, then take the first that
        # actually exists on disk. Order: every --pred-root in argv order,
        # then every --pred-template in argv order. Optionally fall back to
        # running the Arbitr3D pipeline on this instance.
        candidates: List[Path] = []
        for root in (args.pred_root or []):
            candidates.append(Path(root) / category / obj_id)
        for tpl in (args.pred_template or []):
            candidates.append(
                Path(tpl.format(category=category, obj_id=obj_id))
            )
        # Always also consider the pipeline-cache directory: if a previous
        # invocation already wrote ins_sem_seg.npz + final_result.ply there,
        # we'll reuse it under --run-pipeline if-missing without spinning
        # the pipeline back up.
        if args.pipeline_vis_template:
            cached = Path(
                args.pipeline_vis_template.format(
                    category=category, obj_id=obj_id,
                )
            )
            if cached not in candidates:
                candidates.append(cached)

        pred_dir: Optional[Path] = None
        if args.run_pipeline != "always":
            if args.run_pipeline == "if-missing":
                # Prefer existing dirs that already carry the label npz; an
                # existing dir without npz triggers a pipeline re-run below.
                pred_dir = next(
                    (c for c in candidates if c.is_dir() and _has_pred_npz(c)),
                    None,
                )
            else:  # never
                pred_dir = next((c for c in candidates if c.is_dir()), None)

        if pred_dir is None and args.run_pipeline in ("if-missing", "always"):
            try:
                pred_dir = run_pipeline_for_instance(
                    category, obj_id, part_names, args, base_path,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP: pipeline run failed ({type(e).__name__}: {e})")
                if args.fail_fast:
                    raise
                return None

        if pred_dir is None:
            if not candidates:
                print(f"  SKIP: --label-source pred needs --pred-root / "
                      f"--pred-template / --pipeline-vis-template, or "
                      f"--run-pipeline {{if-missing,always}}")
            else:
                print(f"  SKIP: pred dir not found. Tried:")
                for c in candidates:
                    print(f"    - {c}")
            return None

        if len(candidates) > 1 or args.run_pipeline != "never":
            print(f"  pred dir: {pred_dir}")
        try:
            pred_xyz, pred_palette_idx = load_pred_pointcloud_with_labels(
                pred_dir, part_names
            )
        except Exception as e:
            print(f"  SKIP: pred load failed: {e}")
            return None
        face_palette_idx = transfer_pred_to_faces(
            mesh, pred_xyz, pred_palette_idx,
            samples_per_face=int(args.pred_samples_per_face),
        )
        label_tag = "pred"
    else:
        raise ValueError(f"unknown label_source {label_source!r}")

    # ------------------------------------------------------------- statistics
    bins = np.bincount(
        np.where(face_palette_idx >= 0, face_palette_idx, len(part_names)),
        minlength=len(part_names) + 1,
    )
    label_dist = {
        part_names[i] if i < len(part_names) else "unlabeled": int(bins[i])
        for i in range(len(part_names) + 1)
    }
    print(f"  face label distribution: {label_dist}")

    # ----------------------------------------------------------- normalize
    centroid, scale = normalize_mesh_inplace(mesh)
    bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    bbox_diag = float(np.linalg.norm(bbox))
    ground_y = float(mesh.vertices[:, 1].min()) - 0.01 * bbox_diag

    # ----------------------------------------------------------- submeshes
    out_root = Path(args.output_root)
    out_dir = out_root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{obj_id}__{label_tag}"

    with tempfile.TemporaryDirectory(prefix=f"mesh_render_{obj_id}_") as tmpdir:
        submeshes = export_per_part_submeshes(
            mesh, face_palette_idx, palette, gray, Path(tmpdir),
            show_unlabeled_as_gray=not args.hide_unlabeled,
        )
        if not submeshes:
            print(f"  SKIP: no submeshes after split")
            return None

        # ------------------------------------------------------------- views
        H, W = args.resolution
        bg_rgb = args.background_color  # tuple in [0, 1]
        import cv2  # delayed (mitsuba already loaded)
        for elev, azim in args.views:
            cam_pos = cam_position(float(elev), float(azim), float(args.camera_distance))
            scene = build_mesh_scene_dict(
                submeshes, cam_pos,
                width=int(W), height=int(H), spp=int(args.spp),
                bg_color=bg_rgb, fov_deg=float(args.fov),
                ground_y=ground_y,
                light_strength=float(args.light_strength),
                transparent_bg=bool(args.transparent_bg),
                bsdf_alpha=float(args.bsdf_alpha),
            )
            rgba = render_view(scene)
            if args.crop:
                rgba = crop_to_content(rgba, pad=20)

            png_path = out_dir / (
                f"{base_name}_e{int(round(elev))}_a{int(round(azim))}.png"
            )
            if args.transparent_bg:
                alpha = rgba[..., 3]
                rgba[..., :3] = rgba[..., :3] * (alpha > 0)[..., None]
                cv2.imwrite(str(png_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            else:
                rgb_img = composite_over_bg(rgba, bg_rgb)
                cv2.imwrite(str(png_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            print(f"  -> {png_path}")

    # ------------------------------------------------------------- metadata
    meta_out = {
        "category": category,
        "object_id": obj_id,
        "part_names": part_names,
        "label_source": label_tag,
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "face_label_distribution": label_dist,
        "views": [{"elev": float(e), "azim": float(a)} for e, a in args.views],
        "render": {
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
            "spp": int(args.spp),
            "fov": float(args.fov),
            "camera_distance": float(args.camera_distance),
            "palette": args.palette,
            "saturation": float(args.saturation),
            "brightness": float(args.brightness),
            "transparent_bg": bool(args.transparent_bg),
            "bsdf_alpha": float(args.bsdf_alpha),
        },
        "mesh_normalize": {
            "centroid": [float(c) for c in centroid],
            "scale": float(scale),
        },
    }
    with open(out_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    return out_dir


# ===========================================================================
# CLI / main
# ===========================================================================
def _parse_view_pair(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Each view must be 'elev,azim'")
    return float(parts[0]), float(parts[1])


def _parse_view_list(text: str) -> List[Tuple[float, float]]:
    return [_parse_view_pair(c) for c in text.split(";") if c.strip()]


def _parse_color(text: str) -> Tuple[float, float, float]:
    v = text.strip()
    if v.startswith("#"):
        h = v[1:]
        if len(h) != 6:
            raise argparse.ArgumentTypeError("Hex color must be #RRGGBB")
        rgb_int = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    else:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Color must be #RRGGBB or R,G,B")
        rgb_int = tuple(int(p) for p in parts)
    if any(c < 0 or c > 255 for c in rgb_int):
        raise argparse.ArgumentTypeError("RGB values must be in [0, 255]")
    return tuple(c / 255.0 for c in rgb_int)


def _parse_resolution(text: str) -> Tuple[int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must be 'H,W'")
    h, w = int(parts[0]), int(parts[1])
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return h, w


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ---- instance selection
    sel = p.add_argument_group("instance selection")
    sel.add_argument("--category", "-c", default=None,
                     help="A single category (also used as fallback category for "
                          "bare obj_ids in --instances-file).")
    sel.add_argument("--object-id", "-o", default=None,
                     help="A single instance UID (requires --category).")
    sel.add_argument("--categories", nargs="*", default=None,
                     help="Multiple categories: render every instance in each.")
    sel.add_argument("--instances-file", default=None,
                     help="Text file: one '<category>/<obj_id>' per line "
                          "(or bare '<obj_id>' if --category is given).")
    sel.add_argument("--all", action="store_true",
                     help="Render every instance of every category.")
    sel.add_argument("--start-index", type=int, default=0,
                     help="Skip the first N instances after resolution.")
    sel.add_argument("--limit", type=int, default=0,
                     help="Render at most N instances (0 = no cap).")

    # ---- dataset paths (resolution order: CLI > config/config.yaml > hardcoded fallback)
    ds = p.add_argument_group("dataset paths")
    ds.add_argument("--config", default=None,
                    help="Path to config/config.yaml. Defaults to "
                         "<project_root>/config/config.yaml. The "
                         "partobjaverse_tiny section supplies "
                         "base_path / mesh_subdir / gt_subdir / meta_json "
                         "unless overridden on the CLI.")
    ds.add_argument("--base-path", default=None,
                    help=f"Root of PartObjaverse-Tiny. CLI > config > "
                         f"hardcoded ('{DEFAULT_BASE_PATH}').")
    ds.add_argument("--mesh-subdir", default=None,
                    help=f"Mesh subdir under base_path. CLI > config > "
                         f"'{DEFAULT_MESH_SUBDIR}'.")
    ds.add_argument("--gt-subdir", default=None,
                    help=f"GT (.npy) subdir under base_path. CLI > config > "
                         f"'{DEFAULT_GT_SUBDIR}'.")
    ds.add_argument("--meta-json", default=None,
                    help="Path to PartObjaverse-Tiny_semantic.json. Either "
                         "absolute, or relative (resolved against project root "
                         "first, then base_path). CLI > config > "
                         f"'{DEFAULT_META_JSON}'.")

    # ---- label source
    lbl = p.add_argument_group("label source")
    lbl.add_argument("--label-source", choices=["gt", "pred", "both"],
                     default="gt",
                     help="Use the official .npy GT, transfer a predicted "
                          "PC's labels onto the mesh faces, or do both for "
                          "every instance (renders <obj_id>__gt_*.png and "
                          "<obj_id>__pred_*.png side-by-side, useful for "
                          "reviewer figures comparing pred vs GT).")
    lbl.add_argument("--pred-root", default=None, action="append",
                     metavar="DIR",
                     help="Used with --label-source pred. Resolves "
                          "{pred_root}/{category}/{obj_id}/step4_final/. "
                          "May be repeated to search multiple roots in order; "
                          "the first existing match wins. Useful when results "
                          "are split across runs (e.g., "
                          "batch_objaverse_eval/ for Food, "
                          "batch_objaverse_animals_with_h/ for Animals).")
    lbl.add_argument("--pred-template", default=None, action="append",
                     metavar="TEMPLATE",
                     help="Alternative/extension to --pred-root: a Python "
                          "format string with {category} and {obj_id} "
                          "placeholders. May be repeated; first existing "
                          "match wins. Tried after all --pred-root entries.")
    lbl.add_argument("--pred-samples-per-face", type=int, default=3,
                     help="Density of mesh sampling for label transfer.")
    lbl.add_argument("--run-pipeline",
                     choices=["never", "if-missing", "always"],
                     default="if-missing",
                     help="When --label-source pred and the resolved pred dir "
                          "lacks ins_sem_seg.npz (e.g. older batch_objaverse "
                          "runs that only saved the colored PLY), this script "
                          "can invoke Arbitr3DPipeline on the instance to "
                          "regenerate the rendering inputs. "
                          "'never' = use only existing outputs (heuristic "
                          "color-match fallback when npz is missing); "
                          "'if-missing' (default) = run the pipeline only "
                          "when no candidate dir already has the npz; "
                          "'always' = re-run even if existing outputs are "
                          "already complete.")
    lbl.add_argument("--pipeline-vis-template",
                     default=DEFAULT_PIPELINE_VIS_TEMPLATE,
                     help=f"Where the pipeline writes intermediate outputs "
                          f"when --run-pipeline triggers. Format string with "
                          f"{{category}} and {{obj_id}} placeholders. "
                          f"Default: '{DEFAULT_PIPELINE_VIS_TEMPLATE}'.")
    lbl.add_argument("--num-points", type=int, default=None,
                     help="Mesh-sampling point count for the pipeline run. "
                          f"Default: partobjaverse_tiny.num_points from "
                          f"config (or {DEFAULT_PIPELINE_NUM_POINTS} if "
                          f"unset). 0 = use raw vertices.")
    lbl.add_argument("--gpu", type=int, default=-1,
                     help="GPU index for the pipeline (only used when "
                          "--run-pipeline triggers). -1 = honour "
                          "CUDA_VISIBLE_DEVICES (logical 0). Default: -1.")

    # ---- output
    out = p.add_argument_group("output")
    out.add_argument("--output-root", default="outputs/render_mesh",
                     help="Where to write {category}/{obj_id}__{src}_eN_aM.png.")
    out.add_argument("--fail-fast", action="store_true",
                     help="Stop at the first error instead of skipping.")
    out.add_argument("--list-only", action="store_true",
                     help="Resolve instances and print them, then exit. "
                          "Useful for sanity-checking --instances-file picks "
                          "before launching the actual render.")

    # ---- render style (mostly mirrors render_balls_mitsuba)
    r = p.add_argument_group("render style")
    r.add_argument("--views", type=_parse_view_list,
                   default=[(20.0, 45.0), (20.0, 135.0),
                            (20.0, 225.0), (20.0, 315.0)],
                   help='Semicolon-separated "elev,azim" pairs. Default '
                        '"20,45;20,135;20,225;20,315" -- four three-quarter '
                        'azimuthal corners at the same elevation, the standard '
                        '4-view layout reviewers expect for 3D part-seg '
                        'figures (front-3/4 right -> back-3/4 right -> '
                        'back-3/4 left -> front-3/4 left).')
    r.add_argument("--resolution", type=_parse_resolution, default=(1024, 1024))
    r.add_argument("--spp", type=int, default=256)
    r.add_argument("--fov", type=float, default=22.0)
    r.add_argument("--camera-distance", type=float, default=5.2,
                   help="Distance from camera to object centre. Larger "
                        "values pull the camera back, leaving more empty "
                        "margin around the model. Default 5.2 (the model "
                        "fills roughly the inner third of the frame, which "
                        "leaves comfortable padding for paper figures).")
    r.add_argument("--background-color", type=_parse_color,
                   default=_parse_color("236,242,248"))
    r.add_argument("--transparent-bg", action="store_true", default=True,
                   help="(default on) Output PNGs with a transparent BG.")
    r.add_argument("--no-transparent-bg", dest="transparent_bg",
                   action="store_false")
    r.add_argument("--crop", action="store_true", default=True,
                   help="(default on) Tightly crop each render to the object.")
    r.add_argument("--no-crop", dest="crop", action="store_false")
    r.add_argument("--light-strength", type=float, default=6.0)
    r.add_argument("--bsdf-alpha", type=float, default=0.10,
                   help="Roughness of the part BSDF (smaller = glossier).")
    r.add_argument("--palette",
                   choices=list(PALETTES.keys()), default="arbitr3d",
                   help="Named palette for part colors (default: arbitr3d, "
                        "matching ZeroPS / qualitative_4x6 saturated colors).")
    r.add_argument("--saturation", type=float, default=1.0)
    r.add_argument("--brightness", type=float, default=1.0)
    r.add_argument("--hide-unlabeled", action="store_true",
                   help="Drop -1 (background) faces instead of rendering them gray.")

    # ---- mitsuba
    p.add_argument("--variant",
                   choices=["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"],
                   default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- resolve dataset paths: CLI flag > config/config.yaml > hardcoded
    config_path = (
        Path(args.config) if args.config
        else PROJECT_ROOT / "config" / "config.yaml"
    )
    pot_cfg = _load_partobjaverse_config(config_path)

    eff_base = (
        args.base_path
        or pot_cfg.get("base_path")
        or DEFAULT_BASE_PATH
    )
    eff_mesh_subdir = (
        args.mesh_subdir
        or pot_cfg.get("mesh_subdir")
        or DEFAULT_MESH_SUBDIR
    )
    eff_gt_subdir = (
        args.gt_subdir
        or pot_cfg.get("gt_subdir")
        or DEFAULT_GT_SUBDIR
    )
    eff_meta_rel = (
        args.meta_json
        or pot_cfg.get("meta_json")
        or DEFAULT_META_JSON
    )
    # num_points: CLI > config > hardcoded. Stored back so render_one and
    # the pipeline runner share a single source of truth.
    if args.num_points is None:
        args.num_points = int(
            pot_cfg.get("num_points", DEFAULT_PIPELINE_NUM_POINTS)
        )
    # Write resolved values back so render_one() can keep reading args.*
    args.base_path = eff_base
    args.mesh_subdir = eff_mesh_subdir
    args.gt_subdir = eff_gt_subdir
    args.meta_json = eff_meta_rel

    base_path = Path(eff_base)
    print(f"[render-mesh] base_path   = {base_path}")
    print(f"[render-mesh] mesh_subdir = {eff_mesh_subdir}")
    print(f"[render-mesh] gt_subdir   = {eff_gt_subdir}")
    print(f"[render-mesh] meta_json   = {eff_meta_rel}")
    if not base_path.is_dir():
        print(f"[render-mesh] WARN: base_path does not exist on this machine: "
              f"{base_path}")

    # Meta + instance resolution happens before the (slow) Mitsuba variant
    # selection so that --list-only can return instantly.
    meta_path, meta = load_meta(eff_meta_rel, base_path)
    print(f"[render-mesh] meta loaded from: {meta_path}")

    instances = resolve_instances(meta, args)

    if args.list_only:
        print(f"[render-mesh] {len(instances)} instances resolved "
              f"(label-source={args.label_source}); --list-only set, exiting.")
        for i, (cat, oid, parts) in enumerate(instances, start=1):
            print(f"  [{i:>4d}] {cat}/{oid}  parts=({', '.join(parts)})")
        return

    if not instances:
        print(f"[render-mesh] no instances resolved; nothing to do.")
        return

    variant = _select_variant(preferred=args.variant)
    print(f"[render-mesh] mitsuba variant: {variant}")
    print(f"[render-mesh] {len(instances)} instances to render "
          f"(label-source={args.label_source}).")

    palette = PALETTES[args.palette]
    if args.saturation != 1.0 or args.brightness != 1.0:
        palette = adjust_palette(palette, args.saturation, args.brightness)
        print(f"[render-mesh] palette adjusted: "
              f"saturation={args.saturation}, brightness={args.brightness}")
    gray = PALETTE_GRAY[args.palette]

    # --label-source both -> render each instance twice (pred then gt) so the
    # output dir holds matching <obj_id>__pred_e*.png and <obj_id>__gt_e*.png
    # pairs for every view.
    sources = (
        ["pred", "gt"] if args.label_source == "both"
        else [args.label_source]
    )

    n_done, n_skipped, n_failed = 0, 0, 0
    total_renders = len(instances) * len(sources)
    for i, (cat, obj_id, parts) in enumerate(instances, start=1):
        print(f"\n[{i}/{len(instances)}] {cat}/{obj_id} "
              f"(parts: {', '.join(parts)})")
        for src in sources:
            if len(sources) > 1:
                print(f"  --- label_source={src} ---")
            try:
                out = render_one(
                    cat, obj_id, parts,
                    args=args, palette=palette, gray=gray,
                    label_source=src,
                )
                if out is None:
                    n_skipped += 1
                else:
                    n_done += 1
            except Exception as e:
                n_failed += 1
                print(f"  ERROR ({type(e).__name__}): {e}")
                traceback.print_exc()
                if args.fail_fast:
                    raise

    print(f"\n[render-mesh] done. rendered={n_done}, "
          f"skipped={n_skipped}, failed={n_failed}, "
          f"total={total_renders} ({len(instances)} instances x "
          f"{len(sources)} source(s)).")


if __name__ == "__main__":
    main()

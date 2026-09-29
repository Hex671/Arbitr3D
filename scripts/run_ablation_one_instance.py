"""Run a single PartNetE instance through Arbitr3D under five knowledge-ablation
configurations, then render each result as Mitsuba "balls" from four canonical
views.

Five ablations
--------------
  full   : all three knowledge bases enabled (K_c, K_g, K_t)
  no_kc  : disable K_c (category-wise part knowledge / unified_knowledge)
  no_kg  : disable K_g (patchlet-wise geometry / initial_instance_knowledge)
  no_kt  : disable K_t (part-level topology / updated_instance_knowledge,
                        also disables the 2nd-round MLLM refinement)
  none   : disable all three

Output layout (controlled by --output-root)
-------------------------------------------
  <output_root>/gt/
      gt_<category>_<obj_id>.ply
      ins_sem_seg.npz
      renders/                              <-- 4 GT mitsuba views + collage
  <output_root>/<ablation>/pipeline_output/   <-- Arbitr3D vis_dir
      knowledge_system/, step1_sam/, ..., step4_final/final_result.ply
  <output_root>/<ablation>/renders/           <-- 4 mitsuba views + collage
      final_result_e20_a45.png
      final_result_e20_a135.png
      final_result_e20_a225.png
      final_result_e20_a315.png
      final_result_collage.png

Example
-------
  python scripts/run_ablation_one_instance.py \
      --category Chair --obj-id 179 \
      --dataset-base datasets/PartNetE/test \
      --meta-json PartNetE_meta.json \
      --output-root visual_res/ablation_chair179

Notes
-----
- The 4 rendering views are fixed at elev=20, azim in {45, 135, 225, 315}.
- Pipeline initialization (loading SAM, MLLM clients, etc.) happens once and
  is reused across all five ablations to save time and GPU memory.
- If a particular ablation crashes mid-pipeline we log the traceback and move
  on to the next; rendering for the failed ablation is skipped.
- Use --skip-pipeline to only re-render an existing final_result.ply (handy
  for tweaking palette / spp / view angles without rerunning the pipeline).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from typing import Dict, List, Tuple

import numpy as np

# Make CUDA device numbering match nvidia-smi (must be set before any torch import).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
_LAUNCH_CWD = os.path.abspath(os.getcwd())

# ---------------------------------------------------------------------------
# sys.path surgery
# ---------------------------------------------------------------------------
# When a user runs ``python scripts/run_ablation_one_instance.py`` Python
# automatically prepends the script's parent directory (``scripts/``) to
# ``sys.path[0]`` before our code starts. That breaks the pipeline import
# because ``scripts/config.py`` (a real module used by knowledge-generation
# scripts) silently shadows the project-root ``config/`` *namespace* package
# that ``pipeline.py`` / ``module_2d_fm`` depend on
# (``from config.feature_mapping import ...``). Regular modules always win
# over namespace package portions, irrespective of sys.path order, so just
# putting PROJECT_ROOT first is not enough -- we must actively remove the
# auto-added scripts dir entry. We also insert the launch cwd to survive
# setups where the script mount (e.g. /file/...) and launch cwd mount
# (e.g. /data/...) differ.
def _realpath_safe(p: str) -> str:
    try:
        return os.path.realpath(p)
    except Exception:
        return ""

_script_dir_real = _realpath_safe(SCRIPT_DIR)
sys.path[:] = [
    p for p in sys.path
    if not p or _realpath_safe(p) != _script_dir_real
]
for _root in (PROJECT_ROOT, _LAUNCH_CWD):
    if _root not in sys.path:
        sys.path.insert(0, _root)


# ---------------------------------------------------------------------------
# Ablation table: name -> ablation switches dict consumed by pipeline.config.
# ---------------------------------------------------------------------------
ABLATIONS: Dict[str, Dict[str, bool]] = {
    "full":  {"use_unified_knowledge": True,  "use_initial_instance_knowledge": True,  "use_updated_instance_knowledge": True},
    "no_kc": {"use_unified_knowledge": False, "use_initial_instance_knowledge": True,  "use_updated_instance_knowledge": True},
    "no_kg": {"use_unified_knowledge": True,  "use_initial_instance_knowledge": False, "use_updated_instance_knowledge": True},
    "no_kt": {"use_unified_knowledge": True,  "use_initial_instance_knowledge": True,  "use_updated_instance_knowledge": False},
    "none":  {"use_unified_knowledge": False, "use_initial_instance_knowledge": False, "use_updated_instance_knowledge": False},
}

# Four canonical rendering views (elevation, azimuth) in degrees.
DEFAULT_VIEWS: List[Tuple[float, float]] = [
    (20.0, 45.0),
    (20.0, 135.0),
    (20.0, 225.0),
    (20.0, 315.0),
]

DEFAULT_LABEL_PALETTE = np.array([
    [1.00, 0.00, 0.00],   # red
    [0.00, 1.00, 0.00],   # green
    [0.00, 0.00, 1.00],   # blue
    [1.00, 1.00, 0.00],   # yellow
    [0.50, 0.00, 0.50],   # purple
    [0.00, 1.00, 1.00],   # cyan
    [1.00, 0.50, 0.00],   # orange
    [1.00, 0.75, 0.80],   # pink
    [0.60, 0.30, 0.00],   # brown
    [0.75, 1.00, 0.00],   # lime
], dtype=np.float32)
DEFAULT_BACKGROUND_COLOR = np.array([0.70, 0.70, 0.70], dtype=np.float32)


# ---------------------------------------------------------------------------
# GPU auto-selection (mirrors batch_test.py logic, slimmed down).
# ---------------------------------------------------------------------------
def _query_physical_gpus() -> Dict[int, Tuple[float, float]]:
    """Return {physical_id: (free_gb, total_gb)} via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        mapping: Dict[int, Tuple[float, float]] = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            mapping[int(parts[0])] = (float(parts[1]) / 1024.0, float(parts[2]) / 1024.0)
        return mapping
    except Exception:
        return {}


def _select_best_gpu(physical_map: Dict[int, Tuple[float, float]], threshold_gb: float) -> int:
    candidates = {idx: free for idx, (free, _) in physical_map.items() if free >= threshold_gb}
    if not candidates:
        raise RuntimeError(
            f"No GPU has >= {threshold_gb:.1f} GB free memory. "
            f"Status: {physical_map}"
        )
    return max(candidates, key=lambda i: candidates[i])


def setup_gpu(explicit_gpu: int = None) -> int:
    """Print GPU status, pick a logical device, and `torch.cuda.set_device` it.

    Returns the logical GPU index that callers should pass to Arbitr3DPipeline.
    """
    raw_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "auto").strip()
    externally_set = raw_cvd.lower() not in ("", "auto")

    physical_map = _query_physical_gpus()
    print("\n[gpu] Physical GPU status (nvidia-smi index -> free / total GB):")

    if explicit_gpu is not None:
        # User pinned a specific physical GPU; replicate batch_test.py masking.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(explicit_gpu)
        print(f"[gpu] CUDA_VISIBLE_DEVICES set to physical GPU {explicit_gpu} -> logical 0")
        logical = 0
    elif externally_set:
        visible_ids = [int(x) for x in raw_cvd.split(",") if x.strip().isdigit()]
        print(f"[gpu] CUDA_VISIBLE_DEVICES={raw_cvd}, physical {visible_ids} -> logical 0")
        logical = 0
    else:
        threshold = 4.0
        try:
            import yaml
            with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), "r", encoding="utf-8") as f:
                cfg_for_thresh = yaml.safe_load(f) or {}
            threshold = float(cfg_for_thresh.get("gpu", {}).get("min_free_gb", 4.0))
        except Exception:
            pass
        chosen = _select_best_gpu(physical_map, threshold)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
        print(f"[gpu] auto-selected physical GPU {chosen} (>= {threshold:.1f} GB free) -> logical 0")
        logical = 0

    for idx in sorted(physical_map.keys()):
        free_gb, total_gb = physical_map[idx]
        used_pct = (total_gb - free_gb) / total_gb * 100
        bar_len = int((total_gb - free_gb) / total_gb * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  GPU {idx}: [{bar}] {used_pct:5.1f}% used, free {free_gb:5.1f} / {total_gb:5.1f} GB")
    print()

    import torch  # noqa: E402  (delayed because CUDA_VISIBLE_DEVICES must be set first)
    torch.cuda.set_device(logical)
    return logical


# ---------------------------------------------------------------------------
# Pipeline / rendering helpers.
# ---------------------------------------------------------------------------
def load_prompt_classes(meta_path: str, category: str) -> str:
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if category not in meta:
        raise ValueError(f"Category '{category}' not found in {meta_path}")
    return ", ".join(meta[category])


def load_prompt_class_list(meta_path: str, category: str) -> List[str]:
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if category not in meta:
        raise ValueError(f"Category '{category}' not found in {meta_path}")
    return list(meta[category])


def _colors_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.tile(DEFAULT_BACKGROUND_COLOR, (labels.shape[0], 1))
    fg = labels >= 0
    if fg.any():
        colors[fg] = DEFAULT_LABEL_PALETTE[labels[fg] % len(DEFAULT_LABEL_PALETTE)]
    return colors.astype(np.float32)


def prepare_gt_outputs(
    *,
    pc_abs: str,
    label_path: str,
    out_dir: str,
    category: str,
    obj_id: str,
    part_names: List[str],
    force: bool = False,
) -> str:
    """Create a colored GT PLY + ins_sem_seg.npz for render_balls_mitsuba.py.

    Returns the GT PLY path. Raises if label.npy is missing or malformed.
    """
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"GT label.npy not found: {label_path}")

    os.makedirs(out_dir, exist_ok=True)
    cat_l = category.lower()
    gt_ply = os.path.join(out_dir, f"gt_{cat_l}_{obj_id}.ply")
    gt_npz = os.path.join(out_dir, "ins_sem_seg.npz")
    if os.path.isfile(gt_ply) and os.path.isfile(gt_npz) and not force:
        print(f"[gt] Reusing GT PLY/NPZ under {out_dir}")
        return gt_ply

    import open3d as o3d  # delayed import; only needed when writing GT assets

    gt_data = np.load(label_path, allow_pickle=True).item()
    gt_labels = gt_data.get("semantic_seg", None)
    if gt_labels is None:
        raise KeyError(f"{label_path} has no 'semantic_seg' key")
    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    pcd = o3d.io.read_point_cloud(pc_abs)
    xyz = np.asarray(pcd.points)
    if xyz.shape[0] != gt_labels.shape[0]:
        raise ValueError(
            f"GT label size ({gt_labels.shape[0]}) != point count ({xyz.shape[0]})"
        )

    pcd.colors = o3d.utility.Vector3dVector(_colors_from_labels(gt_labels))
    o3d.io.write_point_cloud(gt_ply, pcd)

    class_mapping = {name: idx for idx, name in enumerate(part_names)}
    np.savez(
        gt_npz,
        sem_label=gt_labels,
        class_mapping=np.array(list(class_mapping.items()), dtype=object),
    )
    print(f"[gt] wrote {gt_ply}")
    print(f"[gt] wrote {gt_npz}")
    return gt_ply


def run_pipeline_one_ablation(
    pipeline,
    *,
    pc_rel_path: str,
    prompt_classes_str: str,
    vis_dir: str,
    ablation_name: str,
    switches: Dict[str, bool],
) -> str:
    """Patch pipeline.config['ablation'] and call pipeline.run.

    Returns the absolute path of the rendered final_result.ply on success.
    Raises on failure (caller decides whether to skip rendering).
    """
    full_switches = dict(switches)
    full_switches.setdefault("disable_height_prior", False)  # never auto-trigger orthogonal ablation
    pipeline.config["ablation"] = full_switches

    print("\n" + "=" * 72)
    print(f"[run] Ablation '{ablation_name}' -> switches = {full_switches}")
    print(f"[run] vis_dir  = {vis_dir}")
    print("=" * 72)

    pipeline.run(
        point_cloud_filename=pc_rel_path,
        prompt_classes_str=prompt_classes_str,
        vis_dir=vis_dir,
    )

    ply_path = os.path.join(vis_dir, "step4_final", "final_result.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"final_result.ply not produced at {ply_path}")
    return ply_path


def render_four_views(
    *,
    render_script: str,
    ply_path: str,
    out_dir: str,
    views: List[Tuple[float, float]],
    palette: str,
    spp: int,
    resolution: str,
    camera_distance: float,
    transparent_bg: bool,
    extra_args: List[str],
) -> None:
    """subprocess-call render_balls_mitsuba.py for one PLY.

    Matches ``run_ablation_one_instance_01.py``: no render cache here; use
    ``render_balls_mitsuba`` defaults for ball density unless overridden via
    ``--render-extra`` (e.g. ``--max-points 5000 --ball-radius-ratio 0.014``).
    """
    os.makedirs(out_dir, exist_ok=True)
    views_arg = ";".join(f"{e:g},{a:g}" for e, a in views)
    cmd = [
        sys.executable, render_script,
        "--input-path", ply_path,
        "--views", views_arg,
        "--output-dir", out_dir,
        "--resolution", resolution,
        "--spp", str(spp),
        "--palette", palette,
        "--camera-distance", f"{camera_distance:g}",
        "--save-collage",
    ]
    if transparent_bg:
        cmd.append("--transparent-bg")
    cmd += list(extra_args)
    print("\n[render] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one PartNetE instance through 5 knowledge-ablation "
                    "configurations and render each from 4 mitsuba views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Instance selection
    parser.add_argument("--category", required=True,
                        help="PartNetE category, e.g. Chair / Display / Bottle")
    parser.add_argument("--obj-id", required=True,
                        help="Instance folder name under <dataset_base>/<category>/")
    parser.add_argument("--dataset-base", required=True,
                        help="Absolute path to PartNetE/test (each instance is "
                             "<dataset_base>/<category>/<obj_id>/pc.ply)")
    parser.add_argument("--meta-json", default=os.path.join(PROJECT_ROOT, "PartNetE_meta.json"),
                        help="Path to PartNetE_meta.json (used for prompt classes)")

    # Pipeline / output paths
    parser.add_argument("--config-path", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
                        help="Path to Arbitr3D config.yaml")
    parser.add_argument("--output-root", required=True,
                        help="Root directory for ablation outputs. "
                             "Per-ablation results go under <output_root>/<ablation>/")

    # GPU + ablations to run
    parser.add_argument("--gpu", type=int, default=None,
                        help="Physical GPU id to pin (overrides auto-select)")
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()),
                        choices=list(ABLATIONS.keys()),
                        help="Subset of the 5 ablations to run")

    # Rendering knobs
    parser.add_argument("--render-script",
                        default=os.path.join(PROJECT_ROOT, "scripts", "render_balls_mitsuba.py"),
                        help="Path to render_balls_mitsuba.py")
    parser.add_argument("--palette", default="as-is",
                        choices=["as-is", "arbitr3d", "warm", "muted"],
                        help="Color palette for the balls. 'as-is' uses PLY vertex colors; "
                             "named palettes require ins_sem_seg.npz next to the PLY.")
    parser.add_argument("--spp", type=int, default=128,
                        help="Mitsuba samples-per-pixel (higher = cleaner but slower)")
    parser.add_argument("--resolution", default="1024,1024",
                        help="Render resolution H,W")
    parser.add_argument("--views", type=str, default=None,
                        help="Custom rendering views as 'elev,azim' pairs "
                             "separated by semicolons. For example: "
                             "'30,45' for a single view, or "
                             "'20,45;20,135' for two views. "
                             "Defaults to the 4 canonical views "
                             "(e20a45, e20a135, e20a225, e20a315).")
    parser.add_argument("--camera-distance", type=float, default=5.0,
                        help="Camera distance from origin (object normalized to "
                             "unit sphere). render_balls_mitsuba's own default "
                             "(2.5) is too close for these 4-view paper renders.")
    parser.add_argument("--transparent-bg", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Render with a transparent (alpha=0) background "
                             "and no ground plane, suitable for paper figures. "
                             "Use --no-transparent-bg for solid background + "
                             "envmap (same option semantics as "
                             "run_ablation_one_instance_01.py).")
    parser.add_argument("--render-extra", nargs=argparse.REMAINDER, default=[],
                        help="Trailing args forwarded verbatim to render_balls_mitsuba.py "
                             "(must come last on the command line). "
                             "Example: ... --render-extra --light-strength 4.5")
    parser.add_argument("--render-gt", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Also create and render the GT segmentation from "
                             "<dataset_base>/<category>/<obj_id>/label.npy.")
    parser.add_argument("--force-gt", action="store_true",
                        help="Regenerate the GT PLY/NPZ even if they already exist.")

    parser.add_argument("--force-pipeline", action="store_true",
                        help="Force re-running the pipeline even if a previous "
                             "final_result.ply already exists. By default we "
                             "reuse any existing PLY and only re-render.")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Deprecated alias: skipping is now the default "
                             "whenever the PLY exists. Kept for backward "
                             "compatibility; has no effect.")

    args = parser.parse_args()

    # ----- Validate dataset paths -----
    pc_abs = os.path.join(args.dataset_base, args.category, args.obj_id, "pc.ply")
    if not os.path.isfile(pc_abs):
        raise FileNotFoundError(f"Point cloud not found: {pc_abs}")
    pc_rel = f"{args.category}/{args.obj_id}/pc.ply"

    # ----- GPU + Pipeline init (only if any ablation actually needs it) -----
    # Auto-skip: if every ablation already has a final_result.ply on disk,
    # we never load Arbitr3DPipeline at all. --force-pipeline overrides.
    will_run_pipeline = args.force_pipeline or any(
        not os.path.isfile(os.path.join(
            args.output_root, ab, "pipeline_output", "step4_final", "final_result.ply"))
        for ab in args.ablations
    )
    pipeline = None
    if will_run_pipeline:
        logical_gpu = setup_gpu(explicit_gpu=args.gpu)
        from pipeline import Arbitr3DPipeline  # noqa: E402  (after CUDA setup)
        print(f"[init] Creating Arbitr3DPipeline from {args.config_path} on logical GPU {logical_gpu} ...")
        pipeline = Arbitr3DPipeline(args.config_path, gpu_index=logical_gpu)

    # ----- Parse views -----
    if args.views is not None:
        views: List[Tuple[float, float]] = []
        for pair in args.views.split(";"):
            parts = pair.strip().split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid view spec '{pair}': expected 'elev,azim'")
            views.append((float(parts[0]), float(parts[1])))
        print(f"[views] Custom views: {views}")
    else:
        views = DEFAULT_VIEWS
        print(f"[views] Using default 4 canonical views")

    prompt_classes_str = load_prompt_classes(args.meta_json, args.category)
    prompt_class_list = load_prompt_class_list(args.meta_json, args.category)
    print(f"[meta] Prompt classes for {args.category}: {prompt_classes_str}")

    # ----- GT render -----
    summary: List[Tuple[str, str, str]] = []  # (ablation/gt, status, detail)
    if args.render_gt:
        gt_root = os.path.join(args.output_root, "gt")
        gt_render_dir = os.path.join(gt_root, "renders")
        gt_label_path = os.path.join(args.dataset_base, args.category, args.obj_id, "label.npy")
        try:
            gt_ply = prepare_gt_outputs(
                pc_abs=pc_abs,
                label_path=gt_label_path,
                out_dir=gt_root,
                category=args.category,
                obj_id=args.obj_id,
                part_names=prompt_class_list,
                force=args.force_gt,
            )
            render_four_views(
                render_script=args.render_script,
                ply_path=gt_ply,
                out_dir=gt_render_dir,
                views=views,
                palette=args.palette,
                spp=args.spp,
                resolution=args.resolution,
                camera_distance=args.camera_distance,
                transparent_bg=args.transparent_bg,
                extra_args=args.render_extra,
            )
            summary.append(("gt", "OK", gt_render_dir))
        except subprocess.CalledProcessError as e:
            print(f"\n[gt] ERROR: render_balls_mitsuba returned {e.returncode}")
            summary.append(("gt", "RENDER_FAIL", f"return code {e.returncode}"))
        except Exception as e:
            print(f"\n[gt] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary.append(("gt", "GT_FAIL", f"{type(e).__name__}: {e}"))

    # ----- Iterate ablations -----
    for ablation_name in args.ablations:
        switches = ABLATIONS[ablation_name]
        ablation_root = os.path.join(args.output_root, ablation_name)
        vis_dir = os.path.join(ablation_root, "pipeline_output")
        render_dir = os.path.join(ablation_root, "renders")
        ply_path = os.path.join(vis_dir, "step4_final", "final_result.ply")

        # ---- Pipeline ----
        ply_exists = os.path.isfile(ply_path)
        if ply_exists and not args.force_pipeline:
            print(f"\n[run] [{ablation_name}] PLY exists, reusing {ply_path} "
                  f"(pass --force-pipeline to rerun).")
        else:
            try:
                ply_path = run_pipeline_one_ablation(
                    pipeline,
                    pc_rel_path=pc_rel,
                    prompt_classes_str=prompt_classes_str,
                    vis_dir=vis_dir,
                    ablation_name=ablation_name,
                    switches=switches,
                )
            except Exception as e:
                print(f"\n[run] [{ablation_name}] ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                summary.append((ablation_name, "PIPELINE_FAIL", f"{type(e).__name__}: {e}"))
                continue

        # ---- Render ----
        try:
            render_four_views(
                render_script=args.render_script,
                ply_path=ply_path,
                out_dir=render_dir,
                views=views,
                palette=args.palette,
                spp=args.spp,
                resolution=args.resolution,
                camera_distance=args.camera_distance,
                transparent_bg=args.transparent_bg,
                extra_args=args.render_extra,
            )
            summary.append((ablation_name, "OK", render_dir))
        except subprocess.CalledProcessError as e:
            print(f"\n[render] [{ablation_name}] ERROR: render_balls_mitsuba returned {e.returncode}")
            summary.append((ablation_name, "RENDER_FAIL", f"return code {e.returncode}"))
        except Exception as e:
            print(f"\n[render] [{ablation_name}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary.append((ablation_name, "RENDER_FAIL", f"{type(e).__name__}: {e}"))

    # ----- Final report -----
    print("\n" + "=" * 72)
    print(f"Summary for {args.category}/{args.obj_id}")
    print("=" * 72)
    for name, status, detail in summary:
        print(f"  [{status:14s}] {name:6s}  {detail}")
    print(f"\nAll outputs under: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()

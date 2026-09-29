"""Run the Arbitr3D pipeline on a single PartNetE instance and render its
segmentation result with the granular-ball look defined by
``scripts/render_balls_mitsuba.py``.

Typical usage
-------------
Run pipeline + render the default view ``(elev=20, azim=45)``::

    python scripts/render_partnete_segment.py \
        --category Eyeglasses --instance-id 101284

Render multiple views, save a horizontal collage::

    python scripts/render_partnete_segment.py \
        --category Chair --instance-id 179 \
        --view 20,45 --view 10,135 --view 20,225 --view 10,315

Iterate on render parameters only (reuses the already-saved pred/gt PLYs)::

    python scripts/render_partnete_segment.py \
        --category Eyeglasses --instance-id 101284 \
        --skip-pipeline --exp-suffix Eyeglasses_render \
        --saturation 1.8 --brightness 1.3 --view 25,30

Output layout
-------------
- ``visual_res/<id>_vis_<exp_suffix>/``        : pipeline outputs (pred/gt ply+npz, step dirs)
- ``outputs/render_partnete/<Cat>_<id>/pred/`` : per-view PNGs of predicted segmentation
- ``outputs/render_partnete/<Cat>_<id>/gt/``   : per-view PNGs of GT (if GT available)
- ``outputs/render_partnete/<Cat>_<id>/*_collage.png`` : horizontal collage if >1 view
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure CUDA device order matches nvidia-smi BEFORE any torch import.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# ---------------------------------------------------------------------------
# sys.path surgery
# ---------------------------------------------------------------------------
# When the user runs ``python scripts/render_partnete_segment.py``, Python
# automatically prepends ``scripts/`` to ``sys.path[0]``. That breaks the
# pipeline runner because ``scripts/config.py`` (a real module used by
# knowledge-generation scripts) silently shadows the project-root
# ``config/`` *namespace* package that ``pipeline.py`` / ``module_2d_fm``
# depend on (``from config.feature_mapping import ...``). Regular modules
# always win over namespace package portions, regardless of sys.path order,
# so *just adding REPO_ROOT first is not enough* -- we must actively remove
# the auto-added scripts entry. Then we re-add REPO_ROOT and import
# ``render_balls_mitsuba`` through the ``scripts`` namespace package.
sys.path[:] = [
    p for p in sys.path
    if not p or Path(p).resolve() != SCRIPT_DIR
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------
def parse_view(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--view must be 'elev,azim' (deg); got {text!r}"
        )
    return float(parts[0]), float(parts[1])


def parse_resolution(text: str) -> Tuple[int, int]:
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--resolution must be 'H,W'; got {text!r}"
        )
    return parts[0], parts[1]


def parse_bg_color(text: str) -> Tuple[float, float, float]:
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--background-color must be 'R,G,B' in 0-255; got {text!r}"
        )
    return tuple(c / 255.0 for c in parts)  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run Arbitr3D on a single PartNetE instance and render its "
            "segmentation result in the granular-ball style of "
            "scripts/render_balls_mitsuba.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----- Instance selection -----
    p.add_argument("--category", required=True,
                   help='PartNetE category (e.g. "Chair", "Eyeglasses")')
    p.add_argument("--instance-id", required=True,
                   help='PartNetE instance number (e.g. "179")')
    p.add_argument("--exp-suffix", default=None,
                   help="Suffix for visual_res/<id>_vis_<suffix>. "
                        "Defaults to '<category>_render'.")

    # ----- Pipeline control -----
    p.add_argument("--config-path", default="config/config.yaml",
                   help="Path to Arbitr3D pipeline config "
                        "(absolute or relative to repo root)")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Don't re-run Arbitr3D; only render existing outputs "
                        "in visual_res/<id>_vis_<exp_suffix>.")
    p.add_argument("--prompt-classes", default=None,
                   help="Override comma-separated part class string. "
                        "Default reads from PartNetE_meta.json.")

    # ----- View -----
    p.add_argument(
        "--view", action="append", type=parse_view, metavar="ELEV,AZIM",
        help="Camera view 'elev,azim' (deg). Pass multiple times for "
             "multi-view collage. Default: 20,45.",
    )

    # ----- Label source -----
    p.add_argument("--label-source", choices=["pred", "gt", "both"], default="pred",
                   help="Render predicted seg / GT / both side by side.")

    # ----- Render params (forwarded to render_balls_mitsuba) -----
    p.add_argument("--palette", default="warm",
                   choices=["warm", "muted", "arbitr3d", "as-is"])
    p.add_argument("--saturation", type=float, default=1.5)
    p.add_argument("--brightness", type=float, default=1.5)
    p.add_argument("--light-strength", type=float, default=6.0)
    p.add_argument("--resolution", type=parse_resolution, default=(1024, 1024),
                   help="Render H,W")
    p.add_argument("--spp", type=int, default=256, help="Samples per pixel")
    p.add_argument("--ball-radius-ratio", type=float, default=0.014,
                   help="Ball radius / bbox diag")
    p.add_argument("--camera-distance", type=float, default=5.0,
                   help="Camera distance from origin "
                        "(object normalized to unit sphere)")
    p.add_argument("--fov", type=float, default=22.0, help="Camera FOV (deg)")
    p.add_argument("--max-points", type=int, default=5000,
                   help="Subsample if PLY has more points")
    p.add_argument("--transparent-bg", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Render with transparent background (use "
                        "--no-transparent-bg for solid bg + ground plane)")
    p.add_argument("--background-color", default="236,242,248",
                   help="Solid bg 'R,G,B' (only with --no-transparent-bg)")
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True,
                   help="Crop each rendered image to its non-empty bbox")
    p.add_argument("--save-collage", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Save a horizontal collage when >1 view is rendered")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for point subsampling")
    p.add_argument("--variant", default=None,
                   choices=["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"],
                   help="Force a specific Mitsuba variant "
                        "(defaults to first available cuda > llvm > scalar)")

    # ----- Output -----
    p.add_argument("--output-dir", default=None,
                   help="Where to write rendered PNGs. "
                        "Default: outputs/render_partnete/<Cat>_<id>")

    return p


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.exp_suffix is None:
        args.exp_suffix = f"{args.category}_render"
    if not args.view:
        args.view = [(20.0, 45.0)]
    if args.output_dir is None:
        args.output_dir = str(
            REPO_ROOT / "outputs" / "render_partnete"
            / f"{args.category}_{args.instance_id}"
        )
    args.bg_color = parse_bg_color(args.background_color)
    return args


# ---------------------------------------------------------------------------
# GPU bootstrap (mirrors test.py)
# ---------------------------------------------------------------------------
def _query_nvidia_smi() -> dict:
    """Return {physical_idx: (free_gb, total_gb)} or {} if nvidia-smi missing."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        out: dict = {}
        for line in result.stdout.strip().splitlines():
            idx, free, total = [p.strip() for p in line.split(",")]
            out[int(idx)] = (float(free) / 1024.0, float(total) / 1024.0)
        return out
    except Exception:
        return {}


def _auto_pick_gpu() -> int:
    physical_map = _query_nvidia_smi()
    if not physical_map:
        print("[render_partnete] nvidia-smi unavailable; falling back to GPU 0")
        return 0
    threshold = 4.0
    try:
        import yaml
        cfg_path = REPO_ROOT / "config" / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        threshold = float(cfg.get("gpu", {}).get("min_free_gb", 4.0))
    except Exception:
        pass
    candidates = {idx: free for idx, (free, _) in physical_map.items()
                  if free >= threshold}
    if not candidates:
        raise RuntimeError(
            f"No GPU with >= {threshold:.1f} GB free memory available."
        )
    return max(candidates, key=lambda i: candidates[i])


def setup_gpu() -> int:
    """Return the logical GPU index PyTorch should use."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "auto").strip()
    externally_set = raw.lower() not in ("", "auto")
    if externally_set:
        # CUDA_VISIBLE_DEVICES masks the device list; first masked card is logical 0.
        print(f"[render_partnete] CUDA_VISIBLE_DEVICES={raw}; using logical GPU 0")
        return 0
    selected = _auto_pick_gpu()
    print(f"[render_partnete] auto-selected physical GPU {selected}")
    return selected


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def load_prompt_classes(meta_path: Path, category: str) -> str:
    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if category not in meta:
        raise ValueError(
            f"Category '{category}' not in {meta_path.name}. "
            f"Available examples: {sorted(meta.keys())[:8]}..."
        )
    return ", ".join(meta[category])


def run_pipeline(args: argparse.Namespace, gpu_index: int) -> Path:
    """Run Arbitr3D on the requested instance and return the vis_dir."""
    import torch
    torch.cuda.set_device(gpu_index)

    from pipeline import Arbitr3DPipeline  # heavy import; defer until needed

    cfg = args.config_path
    if not Path(cfg).is_absolute():
        cfg = str(REPO_ROOT / cfg)
    pipeline = Arbitr3DPipeline(cfg, gpu_index=gpu_index)

    point_cloud_filename = f"{args.category}/{args.instance_id}/pc.ply"
    if args.prompt_classes is None:
        prompt_str = load_prompt_classes(
            REPO_ROOT / "PartNetE_meta.json", args.category
        )
    else:
        prompt_str = args.prompt_classes
    print(f"[render_partnete] prompt classes: {prompt_str}")

    vis_dir = REPO_ROOT / "visual_res" / f"{args.instance_id}_vis_{args.exp_suffix}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render_partnete] running pipeline on "
          f"{args.category}/{args.instance_id} (vis_dir={vis_dir})")
    pcd, final_labels, class_mapping = pipeline.run(
        point_cloud_filename=point_cloud_filename,
        prompt_classes_str=prompt_str,
        vis_dir=str(vis_dir),
    )

    cat_l = args.category.lower()
    pred_ply = vis_dir / f"pred_{cat_l}_{args.instance_id}.ply"
    pipeline.visualizer.visualize_3d_result(
        pcd, final_labels, save_path=str(pred_ply), class_to_id=class_mapping
    )
    np.savez(
        vis_dir / f"pred_{cat_l}_{args.instance_id}_sem.npz",
        sem_label=np.asarray(final_labels, dtype=np.int64),
        class_mapping=np.array(list(class_mapping.items()), dtype=object),
    )
    print(f"[render_partnete] pred ply -> {pred_ply}")

    # save GT view if available
    base_path = pipeline.config["data"]["base_path"]
    gt_label_path = Path(base_path) / args.category / args.instance_id / "label.npy"
    if gt_label_path.is_file():
        try:
            gt_data = np.load(str(gt_label_path), allow_pickle=True).item()
            gt_labels = gt_data.get("semantic_seg", None)
            if gt_labels is not None:
                gt_ply = vis_dir / f"gt_{cat_l}_{args.instance_id}.ply"
                pipeline.visualizer.visualize_3d_result(
                    pcd, gt_labels, save_path=str(gt_ply), class_to_id=class_mapping
                )
                np.savez(
                    vis_dir / f"gt_{cat_l}_{args.instance_id}_sem.npz",
                    sem_label=np.asarray(gt_labels, dtype=np.int64),
                    class_mapping=np.array(
                        list(class_mapping.items()), dtype=object
                    ),
                )
                print(f"[render_partnete] gt ply   -> {gt_ply}")
        except Exception as e:
            print(f"[render_partnete] failed to save GT: {e}")
    else:
        print(f"[render_partnete] no GT label at {gt_label_path}; skipping GT.")

    return vis_dir


# ---------------------------------------------------------------------------
# Rendering wrapper (uses helpers from render_balls_mitsuba.py)
# ---------------------------------------------------------------------------
_RBM_VARIANT_READY = False


def _ensure_mitsuba(args: argparse.Namespace):
    """Lazy-import render_balls_mitsuba and pick a working Mitsuba variant."""
    global _RBM_VARIANT_READY
    from scripts import render_balls_mitsuba as rbm
    if not _RBM_VARIANT_READY:
        chosen = rbm._select_variant(preferred=args.variant)
        print(f"[render_partnete] mitsuba variant: {chosen}")
        _RBM_VARIANT_READY = True
    return rbm


def _load_sem_label(label_npz: Optional[Path], ply_path: Path) -> Optional[np.ndarray]:
    npz_path = label_npz if (label_npz and label_npz.is_file()) else None
    if npz_path is None:
        sibling = ply_path.parent / "ins_sem_seg.npz"
        if sibling.is_file():
            npz_path = sibling
    if npz_path is None:
        return None
    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception as e:
        print(f"[render_partnete] failed to read {npz_path}: {e}")
        return None
    if "sem_label" not in data:
        print(f"[render_partnete] {npz_path} has no 'sem_label' key")
        return None
    return np.asarray(data["sem_label"], dtype=np.int64)


def render_one_ply(
    ply_path: Path,
    label_npz: Optional[Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[np.ndarray]:
    """Render a single PLY at every requested view; write PNGs and return list."""
    rbm = _ensure_mitsuba(args)
    import cv2

    xyz, rgb = rbm.load_colored_pc(str(ply_path))
    print(f"[render_partnete] {ply_path.name}: {xyz.shape[0]} points")

    sem_label: Optional[np.ndarray] = None
    if args.palette != "as-is":
        sem_label = _load_sem_label(label_npz, ply_path)
        if sem_label is None:
            raise RuntimeError(
                f"--palette={args.palette} requires sem_label .npz, "
                f"but none was found next to {ply_path} or via {label_npz}."
            )
        if sem_label.shape[0] != xyz.shape[0]:
            raise RuntimeError(
                f"sem_label size ({sem_label.shape[0]}) != "
                f"point count ({xyz.shape[0]})"
            )

    if args.max_points and xyz.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
        if sem_label is not None:
            sem_label = sem_label[sel]
        print(f"[render_partnete] subsampled to {xyz.shape[0]} points")

    if sem_label is not None:
        base_palette = rbm.PALETTES[args.palette]
        if args.saturation != 1.0 or args.brightness != 1.0:
            base_palette = rbm.adjust_palette(
                base_palette, args.saturation, args.brightness
            )
        gray = rbm.PALETTE_GRAY[args.palette]
        rgb = rbm.recolor_by_label(sem_label, base_palette, gray)

    xyz, _ = rbm.normalize_to_unit(xyz)
    bbox_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    ball_radius = max(args.ball_radius_ratio * bbox_diag, 5e-3)
    ground_y = float(xyz[:, 1].min()) - ball_radius * 1.5

    H, W = args.resolution
    output_dir.mkdir(parents=True, exist_ok=True)
    images: List[np.ndarray] = []
    stem = ply_path.stem
    for elev, azim in args.view:
        cam_pos = rbm.cam_position(elev, azim, args.camera_distance)
        scene_dict = rbm.build_scene_dict(
            xyz, rgb, cam_pos,
            width=W, height=H, spp=args.spp,
            ball_radius=ball_radius,
            bg_color=args.bg_color,
            fov_deg=args.fov,
            ground_y=ground_y,
            light_strength=args.light_strength,
            transparent_bg=args.transparent_bg,
        )
        rgba = rbm.render_view(scene_dict)
        if args.crop:
            rgba = rbm.crop_to_content(rgba, pad=20)
        out_png = output_dir / f"{stem}_e{int(round(elev))}_a{int(round(azim))}.png"
        if args.transparent_bg:
            alpha = rgba[..., 3]
            opaque = alpha > 0
            rgba[..., :3] = rgba[..., :3] * opaque[..., None]
            cv2.imwrite(str(out_png), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            images.append(rgba)
        else:
            rgb_img = rbm.composite_over_bg(rgba, args.bg_color)
            cv2.imwrite(str(out_png), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            images.append(rgb_img)
        print(f"[render_partnete] wrote {out_png}")
    return images


def maybe_save_collage(
    images: List[np.ndarray], out_path: Path, args: argparse.Namespace
) -> None:
    if not args.save_collage or len(images) < 2:
        return
    rbm = _ensure_mitsuba(args)
    import cv2
    collage = rbm.make_collage(images)
    if collage.shape[2] == 4:
        cv2.imwrite(str(out_path), cv2.cvtColor(collage, cv2.COLOR_RGBA2BGRA))
    else:
        cv2.imwrite(str(out_path), cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))
    print(f"[render_partnete] wrote collage {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = finalize_args(build_arg_parser().parse_args())

    cat_l = args.category.lower()
    vis_dir = REPO_ROOT / "visual_res" / f"{args.instance_id}_vis_{args.exp_suffix}"

    if args.skip_pipeline:
        print(f"[render_partnete] --skip-pipeline: reusing {vis_dir}")
        if not vis_dir.is_dir():
            raise FileNotFoundError(
                f"vis_dir {vis_dir} does not exist; run without --skip-pipeline "
                f"first or pass --exp-suffix matching a previous run."
            )
    else:
        gpu_idx = setup_gpu()
        vis_dir = run_pipeline(args, gpu_idx)

    pred_ply = vis_dir / f"pred_{cat_l}_{args.instance_id}.ply"
    pred_npz = vis_dir / f"pred_{cat_l}_{args.instance_id}_sem.npz"
    gt_ply = vis_dir / f"gt_{cat_l}_{args.instance_id}.ply"
    gt_npz = vis_dir / f"gt_{cat_l}_{args.instance_id}_sem.npz"
    out_dir = Path(args.output_dir)

    if args.label_source in ("pred", "both"):
        if not pred_ply.is_file():
            raise FileNotFoundError(f"pred ply missing: {pred_ply}")
        print(f"\n[render_partnete] rendering pred views -> {out_dir / 'pred'}")
        imgs = render_one_ply(pred_ply, pred_npz, out_dir / "pred", args)
        maybe_save_collage(
            imgs, out_dir / f"pred_{cat_l}_{args.instance_id}_collage.png", args
        )

    if args.label_source in ("gt", "both"):
        if not gt_ply.is_file():
            print(f"[render_partnete] gt ply missing: {gt_ply}; skipping GT")
        else:
            print(f"\n[render_partnete] rendering gt views -> {out_dir / 'gt'}")
            imgs = render_one_ply(gt_ply, gt_npz, out_dir / "gt", args)
            maybe_save_collage(
                imgs, out_dir / f"gt_{cat_l}_{args.instance_id}_collage.png", args
            )

    print(f"\n[render_partnete] done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

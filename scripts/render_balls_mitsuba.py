"""Render a colored point cloud as 3D balls with soft shadows (Mitsuba 3).

Drop-in replacement for `render_pointcloud_views.py`'s look-and-feel, but uses
path tracing -> each point becomes a small shaded sphere. This produces the
"granular" look standard in 3D part-segmentation papers (PointFlow, PartSLIP,
PointCLIPv2, COPS, Find3D, ...).

Example
-------
  python scripts/render_balls_mitsuba.py \
      --input-path /path/to/zerops_sem.ply \
      --views "10,0;10,90;10,180;10,270" \
      --output-dir ./outputs/render_balls/Chair_37825 \
      --save-collage --spp 256

Notes
-----
- Mitsuba variant is auto-selected: cuda_ad_rgb -> llvm_ad_rgb -> scalar_rgb.
- Background tint is set via the constant emitter (light-source color).
- Object is centered at origin and rescaled into a unit-sphere before rendering;
  the ground plane is offset just below the lowest point of the cloud.
- Ball radius defaults to 1.2% of the bbox diagonal, which matches the look in
  the PointFlow renderer for ~5k-10k points. Tune via --ball-radius-ratio.

Dependencies
------------
  pip install mitsuba  # 50 MB, ships its own LLVM/CUDA backends
  pip install open3d numpy pillow opencv-python
"""
from __future__ import annotations

import argparse
import colorsys
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Mitsuba variant selection (cuda > llvm > scalar). Must happen before any
# call to mi.load_dict / mi.render.
# ---------------------------------------------------------------------------
try:
    import mitsuba as mi  # type: ignore
except ImportError as e:
    sys.stderr.write(
        "[render_balls_mitsuba] mitsuba is not installed. "
        "Install it with: pip install mitsuba\n"
    )
    raise


def _scalar_point3(value: Tuple[float, float, float]) -> "mi.ScalarPoint3f":
    return mi.ScalarPoint3f(float(value[0]), float(value[1]), float(value[2]))


def _look_at(
    origin: Tuple[float, float, float],
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> "mi.ScalarTransform4f":
    return mi.ScalarTransform4f().look_at(
        _scalar_point3(origin),
        _scalar_point3(target),
        _scalar_point3(up),
    )


def _scale(value: Tuple[float, float, float]) -> "mi.ScalarTransform4f":
    return mi.ScalarTransform4f().scale(_scalar_point3(value))


def _translate(value: Tuple[float, float, float]) -> "mi.ScalarTransform4f":
    return mi.ScalarTransform4f().translate(_scalar_point3(value))


def _rotate(axis: Tuple[float, float, float], angle: float) -> "mi.ScalarTransform4f":
    return mi.ScalarTransform4f().rotate(_scalar_point3(axis), float(angle))


def _select_variant(preferred: Optional[str] = None) -> str:
    """Pick the best usable Mitsuba variant.

    If `preferred` is given, try it first; otherwise try cuda -> llvm -> scalar.
    Each candidate is validated by loading a tiny scene, because some backends
    (e.g. cuda_ad_rgb on driver versions 565-569) accept set_variant() but then
    crash inside load_dict() with OptiX / LLVM init errors. We fall back rather
    than raising so a misconfigured GPU does not block CPU rendering.
    """
    available = set(mi.variants())
    default_order = ["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"]
    if preferred:
        candidates = [preferred] + [v for v in default_order if v != preferred]
    else:
        candidates = default_order
    for v in candidates:
        if v not in available:
            continue
        try:
            mi.set_variant(v)
            # Smallest legal scene: just an integrator. This forces the JIT/OptiX
            # backend to initialise; if the backend is broken it raises here.
            mi.load_dict({"type": "scene", "integrator": {"type": "path"}})
            return v
        except Exception as e:
            sys.stderr.write(
                f"[render_balls_mitsuba] variant {v!r} unavailable: "
                f"{type(e).__name__}: {e}\n"
            )
            continue
    raise RuntimeError(
        f"No usable Mitsuba variant. Tried: {candidates}; available: {sorted(available)}"
    )


# Postpone open3d import until after variant is set (open3d also pulls torch
# on some systems and can interfere with Mitsuba's C++ symbols on rare setups).
import open3d as o3d  # noqa: E402


# ---------------------------------------------------------------------------
# Palettes for optional render-time recoloring
# ---------------------------------------------------------------------------
# 'arbitr3d' : original vibrant primaries used by Arbitr3D's Visualizer3D /
#              ZeroPS demo. Saturated, high contrast.
# 'warm'     : warm + low-saturation, paper-figure friendly. Inspired by the
#              palettes seen in PointCLIPv2 / COPS / PartSLIP figures.
# 'muted'    : even more pastel; useful when you want the part colors to
#              recede a bit (good for instance-level visuals with many ids).
PALETTES = {
    "arbitr3d": np.array([
        [1.00, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00],
        [1.00, 1.00, 0.00],
        [0.50, 0.00, 0.50],
        [0.00, 1.00, 1.00],
        [1.00, 0.50, 0.00],
        [1.00, 0.75, 0.80],
        [0.60, 0.30, 0.00],
        [0.75, 1.00, 0.00],
    ], dtype=np.float32),
    "warm": np.array([
        [0.88, 0.50, 0.40],   # 0  terracotta (was 0.86,0.56,0.45)
        [0.68, 0.80, 0.45],   # 1  sage green  (was 0.78,0.82,0.55)
        [0.45, 0.62, 0.85],   # 2  dusty blue  (was 0.62,0.72,0.82)
        [0.95, 0.75, 0.35],   # 3  mustard     (was 0.93,0.80,0.50)
        [0.78, 0.50, 0.65],   # 4  mauve       (was 0.76,0.58,0.68)
        [0.40, 0.78, 0.78],   # 5  teal        (was 0.55,0.78,0.78)
        [0.95, 0.55, 0.30],   # 6  burnt orange(was 0.92,0.66,0.42)
        [0.90, 0.62, 0.52],   # 7  peach       (was 0.85,0.72,0.66)
        [0.58, 0.42, 0.32],   # 8  warm taupe  (was 0.62,0.52,0.42)
        [0.82, 0.85, 0.45],   # 9  chartreuse  (was 0.86,0.86,0.58)
    ], dtype=np.float32),
    "muted": np.array([
        [0.78, 0.62, 0.55],
        [0.72, 0.78, 0.62],
        [0.62, 0.70, 0.78],
        [0.85, 0.78, 0.58],
        [0.74, 0.62, 0.70],
        [0.62, 0.74, 0.74],
        [0.85, 0.70, 0.55],
        [0.82, 0.74, 0.70],
        [0.65, 0.58, 0.50],
        [0.80, 0.80, 0.62],
    ], dtype=np.float32),
}

# Background color for points whose label is -1 ("unlabeled" / background).
# Slightly warmer than the previous neutral 0.7 gray to match the warm palette.
PALETTE_GRAY = {
    "arbitr3d": np.array([0.70, 0.70, 0.70], dtype=np.float32),
    "warm":     np.array([0.78, 0.76, 0.72], dtype=np.float32),
    "muted":    np.array([0.78, 0.76, 0.74], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_colored_pc(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a PLY, return (N,3) xyz and (N,3) rgb in [0, 1]."""
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    rgb = np.asarray(pcd.colors, dtype=np.float64)
    if xyz.size == 0:
        raise RuntimeError(f"Empty point cloud: {ply_path}")
    if rgb.size == 0:
        # If the PLY had no per-vertex colors, fall back to a neutral gray.
        rgb = np.full_like(xyz, 0.65)
    return xyz, rgb


def maybe_load_sem_labels(ply_path: Path, override: Optional[str]) -> Optional[np.ndarray]:
    """Load (N,) sem_label from ins_sem_seg.npz next to the PLY, or from override."""
    if override is not None:
        npz_path = Path(override)
    else:
        npz_path = ply_path.parent / "ins_sem_seg.npz"
    if not npz_path.is_file():
        return None
    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception as e:
        print(f"[render_balls_mitsuba] failed to read {npz_path}: {e}")
        return None
    if "sem_label" not in data:
        print(f"[render_balls_mitsuba] {npz_path} has no 'sem_label' key")
        return None
    return np.asarray(data["sem_label"], dtype=np.int64)


def recolor_by_label(
    sem_label: np.ndarray, palette: np.ndarray, gray: np.ndarray
) -> np.ndarray:
    """label==-1 -> gray, else palette[label % len(palette)]."""
    rgb = np.tile(gray, (sem_label.shape[0], 1)).astype(np.float32)
    fg = sem_label >= 0
    if fg.any():
        rgb[fg] = palette[sem_label[fg] % len(palette)]
    return rgb


def adjust_palette(
    palette: np.ndarray, saturation: float = 1.0, brightness: float = 1.0
) -> np.ndarray:
    """Scale palette saturation (S) and brightness (V) in HSV space.

    saturation > 1 makes colors more vivid (clipped at 1.0).
    saturation < 1 pulls colors toward gray.
    brightness > 1 makes colors lighter (clipped at 1.0).
    brightness < 1 makes colors darker.
    """
    if abs(saturation - 1.0) < 1e-6 and abs(brightness - 1.0) < 1e-6:
        return palette
    out = np.empty_like(palette)
    for i, (r, g, b) in enumerate(palette):
        h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        s = float(np.clip(s * saturation, 0.0, 1.0))
        v = float(np.clip(v * brightness, 0.0, 1.0))
        out[i] = colorsys.hsv_to_rgb(h, s, v)
    return out.astype(np.float32)


def normalize_to_unit(xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center on origin, scale so max radius = 1. Returns (xyz_norm, scale)."""
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(centered, axis=1).max())
    if radius <= 1e-12:
        return centered, 1.0
    return centered / radius, radius


# ---------------------------------------------------------------------------
# CLI parsers (kept compatible with render_pointcloud_views.py)
# ---------------------------------------------------------------------------
def parse_resolution(text: str) -> Tuple[int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must be in the form H,W")
    h, w = int(parts[0]), int(parts[1])
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return h, w


def parse_color(text: str) -> Tuple[float, float, float]:
    value = text.strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if len(hex_value) != 6:
            raise argparse.ArgumentTypeError("Hex color must be #RRGGBB")
        rgb_int = tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))
    else:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Color must be #RRGGBB or R,G,B")
        rgb_int = tuple(int(p) for p in parts)
    if any(c < 0 or c > 255 for c in rgb_int):
        raise argparse.ArgumentTypeError("RGB values must be in [0, 255]")
    return tuple(c / 255.0 for c in rgb_int)  # to [0, 1]


def parse_view_pair(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Each view must be in the form elev,azim")
    return float(parts[0]), float(parts[1])


def parse_view_list(text: str) -> List[Tuple[float, float]]:
    views = [parse_view_pair(chunk) for chunk in text.split(";") if chunk.strip()]
    if not views:
        raise argparse.ArgumentTypeError("At least one valid view must be provided")
    return views


# ---------------------------------------------------------------------------
# Camera math
# ---------------------------------------------------------------------------
def cam_position(elev_deg: float, azim_deg: float, distance: float) -> List[float]:
    """elev/azim follow the convention used by render_pointcloud_views.py.

    elev: angle above the horizontal (XZ) plane.
    azim: angle around the vertical (Y) axis, measured from +Z.
    """
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    x = distance * math.cos(elev) * math.sin(azim)
    y = distance * math.sin(elev)
    z = distance * math.cos(elev) * math.cos(azim)
    return [x, y, z]


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------
def build_scene_dict(
    xyz: np.ndarray,
    rgb: np.ndarray,
    cam_pos: List[float],
    *,
    width: int,
    height: int,
    spp: int,
    ball_radius: float,
    bg_color: Tuple[float, float, float],
    fov_deg: float = 25.0,
    ground_y: float = -1.05,
    light_strength: float = 5.0,
    fill_strength: float = 0.45,
    transparent_bg: bool = False,
) -> dict:
    """Construct a Mitsuba 3 scene dict with one diffuse sphere per point.

    If `transparent_bg` is True, skip the ground plane and use a neutral white
    constant emitter so that the alpha channel is fully transparent everywhere
    except where balls are.
    """
    scene = {
        "type": "scene",
        "integrator": {
            "type": "path",
            "max_depth": 6,
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
                "width": width,
                "height": height,
                "rfilter": {"type": "gaussian"},
                "pixel_format": "rgba",
                "sample_border": True,
            },
            "sampler": {"type": "independent", "sample_count": int(spp)},
        },
        # Key light: area emitter from upper-right. The `null` BSDF makes
        # primary camera rays pass straight through the rectangle so the
        # white-lit emitter shape doesn't appear as a bright box in the
        # rendered image. (`hide_emitters: True` on the integrator only
        # suppresses the emitted radiance, not the shape's diffuse-white
        # default scattering, so by itself it's not enough.)
        "key_light": {
            "type": "rectangle",
            "to_world": (
                _look_at((2.0, 3.0, 2.5))
                @ _scale((1.2, 1.2, 1.0))
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
        # No envmap -> rays that miss geometry contribute nothing, so the film's
        # alpha channel is genuinely 0 at background pixels. To keep the spheres
        # from going too dark without ambient fill, we add a softer counter-fill
        # area light on the opposite side of the key.
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
        # Solid background: keep a constant envmap (acts as ambient fill) plus a
        # diffuse ground plane so the chair has a soft contact shadow.
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

    # Insert one sphere per point.
    rgb = np.clip(rgb, 0.0, 1.0)
    for i, (p, c) in enumerate(zip(xyz, rgb)):
        scene[f"p{i}"] = {
            "type": "sphere",
            "center": [float(p[0]), float(p[1]), float(p[2])],
            "radius": float(ball_radius),
            "bsdf": {
                "type": "roughplastic",
                "diffuse_reflectance": {
                    "type": "rgb",
                    "value": [float(c[0]), float(c[1]), float(c[2])],
                },
                "alpha": 0.10,
                "int_ior": 1.46,
            },
        }
    return scene


# ---------------------------------------------------------------------------
# Render + post-process
# ---------------------------------------------------------------------------
def render_view(scene_dict: dict) -> np.ndarray:
    """Render a Mitsuba scene dict, return (H, W, 4) uint8 RGBA image."""
    scene = mi.load_dict(scene_dict)
    img = mi.render(scene)
    # mi.Bitmap conversion to sRGB uint8
    bmp = mi.Bitmap(img).convert(
        mi.Bitmap.PixelFormat.RGBA,
        mi.Struct.Type.UInt8,
        srgb_gamma=True,
    )
    arr = np.array(bmp, copy=True)
    return arr  # (H, W, 4) uint8


def crop_to_content(img: np.ndarray, pad: int = 16) -> np.ndarray:
    """Crop an RGBA image down to the bbox of non-transparent pixels."""
    if img.shape[2] < 4:
        return img
    alpha = img[..., 3]
    ys, xs = np.where(alpha > 0)
    if ys.size == 0 or xs.size == 0:
        return img
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, img.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def composite_over_bg(img: np.ndarray, bg_color: Tuple[float, float, float]) -> np.ndarray:
    """Alpha-composite RGBA image over a solid background color, return RGB uint8."""
    if img.shape[2] == 3:
        return img
    rgb = img[..., :3].astype(np.float32) / 255.0
    alpha = img[..., 3:4].astype(np.float32) / 255.0
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    out = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def make_collage(images: List[np.ndarray]) -> np.ndarray:
    """Stack RGB or RGBA images into a horizontal strip.

    Separator color: white for RGB inputs, transparent for RGBA inputs.
    """
    if not images:
        raise RuntimeError("No images to assemble into collage.")
    h = max(img.shape[0] for img in images)
    n_ch = images[0].shape[2]
    sep = 8
    pad_value = (255,) * 3 + (0,) if n_ch == 4 else 255
    panels: List[np.ndarray] = []
    for i, img in enumerate(images):
        pad_top = (h - img.shape[0]) // 2
        pad_bot = h - img.shape[0] - pad_top
        if n_ch == 4:
            padded = np.pad(
                img,
                ((pad_top, pad_bot), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,  # transparent
            )
        else:
            padded = np.pad(
                img,
                ((pad_top, pad_bot), (0, 0), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        panels.append(padded)
        if i < len(images) - 1:
            sep_block = np.zeros((h, sep, n_ch), dtype=np.uint8)
            if n_ch == 3:
                sep_block[:] = 255
            # for RGBA leave it transparent (all zeros)
            panels.append(sep_block)
    return np.concatenate(panels, axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a colored PLY as 3D balls (Mitsuba 3 path tracing)."
    )
    parser.add_argument("--input-path", required=True, help="Input colored .ply")
    parser.add_argument(
        "--views",
        type=parse_view_list,
        default=[(10.0, 0.0), (10.0, 90.0), (10.0, 180.0), (10.0, 270.0)],
        help='Semicolon-separated "elev,azim" pairs',
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the rendered PNG(s) into",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Base filename for outputs (no extension). Each view becomes "
        "{stem}_e{elev}_a{azim}.png; collage is {stem}_collage.png. "
        "Default: stem of --input-path (e.g. zerops_sem from zerops_sem.ply).",
    )
    parser.add_argument(
        "--background-color",
        type=parse_color,
        default=parse_color("236,242,248"),
        help="Background color: '#RRGGBB' or 'R,G,B' (0-255). Default: 236,242,248",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(1024, 1024),
        help="Render resolution H,W (default 1024,1024)",
    )
    parser.add_argument("--spp", type=int, default=256, help="Samples per pixel")
    parser.add_argument(
        "--ball-radius-ratio",
        type=float,
        default=0.012,
        help="Ball radius as a fraction of bbox diagonal (default 0.012)",
    )
    parser.add_argument(
        "--ball-radius",
        type=float,
        default=None,
        help="Override absolute ball radius (after normalization to unit sphere)",
    )
    parser.add_argument(
        "--camera-distance", type=float, default=2.5,
        help="Camera distance from origin (object is normalized to unit sphere)",
    )
    parser.add_argument("--fov", type=float, default=25.0, help="Camera FOV (deg)")
    parser.add_argument("--crop", action="store_true", help="Crop to object bbox")
    parser.add_argument(
        "--save-collage",
        action="store_true",
        help="Also write a horizontal strip combining all views",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10000,
        help="If the cloud has more points, randomly subsample down to this many "
        "(too many spheres slow Mitsuba and blur the granular look). Default 10000.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Subsample RNG seed for reproducibility"
    )
    parser.add_argument(
        "--palette",
        choices=list(PALETTES.keys()) + ["as-is"],
        default="as-is",
        help="Override per-point colors using a named palette. 'as-is' (default) "
        "uses colors from the PLY. Non-default palettes need ins_sem_seg.npz "
        "next to the PLY (or pass --label-npz).",
    )
    parser.add_argument(
        "--label-npz", default=None,
        help="Path to a .npz file containing 'sem_label' (N,). Auto-detected as "
        "<ply_dir>/ins_sem_seg.npz when --palette != as-is.",
    )
    parser.add_argument(
        "--saturation", type=float, default=1.0,
        help="Multiplier on palette saturation (HSV S). 1.0 = palette as defined; "
        "0.5 = halfway to gray; 1.5 = bumped (clipped at full saturation). "
        "Only affects named palettes (warm/muted/arbitr3d).",
    )
    parser.add_argument(
        "--brightness", type=float, default=1.0,
        help="Multiplier on palette value (HSV V). >1 brightens colors, <1 darkens. "
        "Stacks with --light-strength which controls scene illumination.",
    )
    parser.add_argument(
        "--light-strength", type=float, default=5.0,
        help="Radiance of the area lights. Higher = brighter image with stronger "
        "highlights. Default 5.0.",
    )
    parser.add_argument(
        "--transparent-bg", action="store_true",
        help="Output PNG with a transparent background (RGBA). Removes the "
        "ground plane, so there is no ground shadow.",
    )
    parser.add_argument(
        "--variant",
        choices=["cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"],
        default=None,
        help="Force a specific Mitsuba variant. Defaults to first available "
        "(cuda > llvm > scalar). Use scalar_rgb if cuda OptiX or LLVM JIT "
        "fail (e.g. NVIDIA driver 565-569 OptiX bug).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------- variant
    variant = _select_variant(preferred=args.variant)
    print(f"[render_balls_mitsuba] using mitsuba variant: {variant}")

    # ------------------------------------------------------------- input
    in_path = Path(args.input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input not found: {in_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render_balls_mitsuba] reading {in_path}")
    xyz, rgb = load_colored_pc(str(in_path))
    print(f"[render_balls_mitsuba] {xyz.shape[0]} points")

    # Optional: load semantic labels for render-time recoloring.
    sem_label: Optional[np.ndarray] = None
    if args.palette != "as-is":
        sem_label = maybe_load_sem_labels(in_path, args.label_npz)
        if sem_label is None:
            raise RuntimeError(
                f"--palette {args.palette} requires sem_label; "
                f"could not find ins_sem_seg.npz next to {in_path} "
                f"and no --label-npz was provided."
            )
        if sem_label.shape[0] != xyz.shape[0]:
            raise RuntimeError(
                f"sem_label size ({sem_label.shape[0]}) != point count ({xyz.shape[0]})"
            )
        print(
            f"[render_balls_mitsuba] palette='{args.palette}', "
            f"recoloring {int((sem_label >= 0).sum())} foreground / "
            f"{int((sem_label < 0).sum())} background points"
        )

    # subsample if too dense
    if args.max_points and xyz.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz = xyz[sel]
        rgb = rgb[sel]
        if sem_label is not None:
            sem_label = sem_label[sel]
        print(f"[render_balls_mitsuba] subsampled to {xyz.shape[0]} points")

    # apply palette recoloring after subsampling so indexing matches
    if sem_label is not None:
        base_palette = PALETTES[args.palette]
        if args.saturation != 1.0 or args.brightness != 1.0:
            base_palette = adjust_palette(
                base_palette, args.saturation, args.brightness
            )
            print(
                f"[render_balls_mitsuba] palette adjusted: "
                f"saturation={args.saturation}, brightness={args.brightness}"
            )
        gray = PALETTE_GRAY[args.palette]
        rgb = recolor_by_label(sem_label, base_palette, gray)

    # normalize so the longest semi-axis is 1
    xyz, _ = normalize_to_unit(xyz)
    bbox_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    if args.ball_radius is not None:
        ball_radius = float(args.ball_radius)
    else:
        ball_radius = max(args.ball_radius_ratio * bbox_diag, 5e-3)
    print(f"[render_balls_mitsuba] bbox diag={bbox_diag:.3f}, ball_radius={ball_radius:.4f}")

    # ground plane just below lowest point
    ground_y = float(xyz[:, 1].min()) - ball_radius * 1.5

    H, W = args.resolution
    bg_rgb = args.background_color  # tuple in [0, 1]

    # ------------------------------------------------------------- render
    out_imgs: List[np.ndarray] = []
    stem = in_path.stem
    import cv2  # delayed import (numpy/o3d/mitsuba already loaded by now)
    for elev, azim in args.views:
        cam_pos = cam_position(elev, azim, args.camera_distance)
        print(
            f"[render_balls_mitsuba] view (elev={elev}, azim={azim}) "
            f"cam={cam_pos}"
        )
        scene_dict = build_scene_dict(
            xyz, rgb, cam_pos,
            width=W, height=H, spp=args.spp,
            ball_radius=ball_radius,
            bg_color=bg_rgb,
            fov_deg=args.fov,
            ground_y=ground_y,
            light_strength=args.light_strength,
            transparent_bg=args.transparent_bg,
        )
        rgba = render_view(scene_dict)
        if args.crop:
            rgba = crop_to_content(rgba, pad=20)
        out_path = out_dir / f"{stem}_e{int(round(elev))}_a{int(round(azim))}.png"

        if args.transparent_bg:
            # Premultiply-clear: zero RGB on transparent pixels so viewers that
            # ignore alpha do not show stray illumination as a gray fill.
            alpha = rgba[..., 3]
            opaque_mask = alpha > 0
            transparent_count = int((alpha == 0).sum())
            print(
                f"  alpha range: [{int(alpha.min())}, {int(alpha.max())}], "
                f"transparent pixels: {transparent_count}/{alpha.size}"
            )
            rgba[..., :3] = rgba[..., :3] * opaque_mask[..., None]
            cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            out_imgs.append(rgba)  # collage helper handles 4-channel below
        else:
            rgb_img = composite_over_bg(rgba, bg_rgb)
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            out_imgs.append(rgb_img)
        print(f"  wrote {out_path}")

    if args.save_collage and len(out_imgs) > 1:
        collage = make_collage(out_imgs)
        coll_path = out_dir / f"{stem}_collage.png"
        if collage.shape[2] == 4:
            cv2.imwrite(str(coll_path), cv2.cvtColor(collage, cv2.COLOR_RGBA2BGRA))
        else:
            cv2.imwrite(str(coll_path), cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))
        print(f"[render_balls_mitsuba] wrote {coll_path}")


if __name__ == "__main__":
    main()

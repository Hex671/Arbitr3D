import argparse
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_prep.dataloader import PointCloudDataset
from module_render.camera_poses import generate_cameras_from_views, generate_sphere_cameras
from module_render.renderer import MultiViewRenderer


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "rendered_views"
POSE_CHECK_VIEWS = [(10.0, 0.0), (10.0, 90.0), (10.0, 180.0), (10.0, 270.0)]


def parse_resolution(text: str) -> Tuple[int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must be in the form H,W")
    h, w = int(parts[0]), int(parts[1])
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return h, w


def parse_color(text: str) -> Tuple[int, int, int]:
    value = text.strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if len(hex_value) != 6:
            raise argparse.ArgumentTypeError("Hex color must be #RRGGBB")
        return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))

    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be #RRGGBB or R,G,B")

    rgb = tuple(int(p) for p in parts)
    if any(c < 0 or c > 255 for c in rgb):
        raise argparse.ArgumentTypeError("RGB values must be in [0, 255]")
    return rgb


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


def infer_views_from_positions(cam_positions: np.ndarray) -> List[Tuple[float, float]]:
    views: List[Tuple[float, float]] = []
    for pos in cam_positions:
        radius = float(np.linalg.norm(pos))
        if radius <= 0:
            views.append((0.0, 0.0))
            continue
        elev = float(np.degrees(np.arcsin(pos[1] / radius)))
        azim = float(np.degrees(np.arctan2(pos[0], pos[2])))
        views.append((elev, azim))
    return views


def resolve_cameras(
    single_view: str,
    multi_views: str,
    preset: str,
    camera_distance: float,
) -> Tuple[List[Tuple[float, float]], np.ndarray, np.ndarray]:
    if multi_views:
        views = parse_view_list(multi_views)
        cam_positions, cam_rotations = generate_cameras_from_views(views, radius=camera_distance)
        return views, cam_positions, cam_rotations

    if single_view:
        view = parse_view_pair(single_view)
        cam_positions, cam_rotations = generate_cameras_from_views([view], radius=camera_distance)
        return [view], cam_positions, cam_rotations

    if preset == "pose_check4":
        views = list(POSE_CHECK_VIEWS)
        cam_positions, cam_rotations = generate_cameras_from_views(views, radius=camera_distance)
        return views, cam_positions, cam_rotations

    cam_positions, cam_rotations = generate_sphere_cameras(10, radius=camera_distance)
    views = infer_views_from_positions(cam_positions)
    return views, cam_positions, cam_rotations


def load_object_as_arrays(input_path: Path, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    dataset = PointCloudDataset("")
    pcd = dataset.load_point_cloud(str(input_path), num_points=num_points)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"Loaded object contains no points: {input_path}")
    normalized = dataset.normalize_pc(pcd)
    return np.asarray(normalized.points), dataset.extract_color(normalized)


def make_foreground_mask(image: np.ndarray, background_threshold: int) -> np.ndarray:
    return np.any(image < background_threshold, axis=2)


def apply_background_color(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    background_color: Tuple[int, int, int],
) -> np.ndarray:
    output = image.copy()
    bg = np.array(background_color, dtype=np.uint8)
    output[~foreground_mask] = bg
    return output


def crop_to_foreground(image: np.ndarray, foreground_mask: np.ndarray, padding: int) -> np.ndarray:
    if not np.any(foreground_mask):
        return image
    ys, xs = np.where(foreground_mask)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, image.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def make_collage(images: Sequence[np.ndarray], background_color: Tuple[int, int, int]) -> np.ndarray:
    if not images:
        raise ValueError("No images to compose")

    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    bg = np.array(background_color, dtype=np.uint8)
    collage = np.full((rows * max_h, cols * max_w, 3), bg, dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        y = row * max_h + (max_h - image.shape[0]) // 2
        x = col * max_w + (max_w - image.shape[1]) // 2
        collage[y : y + image.shape[0], x : x + image.shape[1]] = image

    return collage


def render_views(
    xyz: np.ndarray,
    colors: np.ndarray,
    cam_positions: np.ndarray,
    cam_rotations: np.ndarray,
    resolution: Tuple[int, int],
    device: str,
    use_open3d: bool,
):
    renderer = MultiViewRenderer(resolution)
    if use_open3d:
        renderer.has_pytorch3d = False
    elif renderer.has_pytorch3d and device != "auto":
        import torch

        renderer.device = torch.device(device)

    try:
        return renderer.render(xyz, colors, cam_positions, cam_rotations)
    except RuntimeError as exc:
        if use_open3d or not getattr(renderer, "has_pytorch3d", False) or "out of memory" not in str(exc).lower():
            raise
        import torch

        torch.cuda.empty_cache()
        print("CUDA out of memory during rendering. Retrying with PyTorch3D on CPU...")
        renderer.device = torch.device("cpu")
        return renderer.render(xyz, colors, cam_positions, cam_rotations)


def save_outputs(
    images: Sequence[np.ndarray],
    views: Sequence[Tuple[float, float]],
    output_dir: Path,
    save_collage: bool,
    background_color: Tuple[int, int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    for idx, (image, view) in enumerate(zip(images, views)):
        elev, azim = view
        image_name = f"view_{idx:02d}_e{elev:g}_a{azim:g}.png"
        image_path = output_dir / image_name
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        lines.append(f"{image_name}: elev={elev:.2f}, azim={azim:.2f}")

    (output_dir / "view_angles.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if save_collage and len(images) > 1:
        collage = make_collage(images, background_color)
        cv2.imwrite(str(output_dir / "collage.png"), cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render clean viewpoint images from a point cloud or mesh using Arbitr3D's renderer.")
    parser.add_argument("--input-path", required=True, help="Path to a .ply/.pcd/.xyz point cloud or .glb/.obj/.stl mesh")
    parser.add_argument("--output-dir", default=None, help="Directory to save rendered images")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument("--num-points", type=int, default=30000, help="Sampling count for mesh inputs; ignored for point-cloud files")
    parser.add_argument("--view", default=None, help="Single view in the form elev,azim")
    parser.add_argument("--views", default=None, help="Multiple views in the form e1,a1;e2,a2;...")
    parser.add_argument("--preset", choices=["sphere10", "pose_check4"], default="sphere10")
    parser.add_argument("--resolution", type=parse_resolution, default=None, help="Override render resolution as H,W")
    parser.add_argument("--camera-distance", type=float, default=None, help="Override camera distance")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--use-open3d", action="store_true")
    parser.add_argument("--crop", action="store_true", help="Crop each image to the foreground object")
    parser.add_argument("--crop-padding", type=int, default=24)
    parser.add_argument("--background-color", type=parse_color, default=(255, 255, 255), help="Background color as #RRGGBB or R,G,B")
    parser.add_argument("--background-threshold", type=int, default=245, help="Pixels brighter than this threshold are treated as background")
    parser.add_argument("--save-collage", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolution = args.resolution or tuple(config.get("render", {}).get("resolution", [512, 512]))
    camera_distance = float(args.camera_distance or config.get("render", {}).get("camera_distance", 2.0))

    default_output_dir = DEFAULT_OUTPUT_ROOT / input_path.stem
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir

    xyz, colors = load_object_as_arrays(input_path, args.num_points)
    views, cam_positions, cam_rotations = resolve_cameras(args.view, args.views, args.preset, camera_distance)
    render_output = render_views(xyz, colors, cam_positions, cam_rotations, resolution, args.device, args.use_open3d)

    processed_images: List[np.ndarray] = []
    for image in render_output.images:
        foreground_mask = make_foreground_mask(image, args.background_threshold)
        processed = apply_background_color(image, foreground_mask, args.background_color)
        if args.crop:
            processed = crop_to_foreground(processed, foreground_mask, args.crop_padding)
        processed_images.append(processed)

    save_outputs(processed_images, views, output_dir, args.save_collage, args.background_color)
    print(f"Saved {len(processed_images)} rendered view(s) to: {output_dir}")


if __name__ == "__main__":
    main()

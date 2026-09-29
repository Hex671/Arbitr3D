import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_prep.dataloader import PointCloudDataset
from module_render.camera_poses import generate_sphere_cameras
from module_render.renderer import MultiViewRenderer


DEFAULT_MESH_PATH = "datasets/PartObjaverse-Tiny/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/00aee5c2fef743d69421bb642d446a5b.glb"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "partobjaverse_tiny_00aee5c2fef743d69421bb642d446a5b_10views"


def load_mesh_as_point_cloud(mesh_path: str, num_points: int) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {mesh_path}")
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def save_rendered_views(images, output_dir: Path, cam_positions: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    angle_lines = []
    for idx, image in enumerate(images):
        out_path = output_dir / f"view_{idx:02d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        pos = cam_positions[idx]
        radius = np.linalg.norm(pos)
        elev = np.degrees(np.arcsin(pos[1] / radius)) if radius > 0 else 0.0
        azim = np.degrees(np.arctan2(pos[0], pos[2])) if radius > 0 else 0.0
        angle_lines.append(f"view_{idx:02d}.png: elev={elev:.1f}, azim={azim:.1f}")
    (output_dir / "view_angles.txt").write_text("\n".join(angle_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a GLB mesh with Arbitr3D's 10-view camera setup.")
    parser.add_argument("--mesh-path", default=DEFAULT_MESH_PATH)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--num-points", type=int, default=30000)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--use-open3d", action="store_true")
    args = parser.parse_args()

    mesh_path = Path(args.mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolution = tuple(config.get("render", {}).get("resolution", [512, 512]))
    camera_distance = float(config.get("render", {}).get("camera_distance", 2.0))

    pcd = load_mesh_as_point_cloud(str(mesh_path), args.num_points)
    dataset = PointCloudDataset("")
    norm_pcd = dataset.normalize_pc(pcd)
    pc_xyz = np.asarray(norm_pcd.points)
    pc_colors = dataset.extract_color(norm_pcd)

    cam_positions, cam_rotations = generate_sphere_cameras(10, radius=camera_distance)
    renderer = MultiViewRenderer(resolution)
    if args.use_open3d:
        renderer.has_pytorch3d = False
    elif renderer.has_pytorch3d and args.device != "auto":
        import torch
        renderer.device = torch.device(args.device)

    try:
        render_output = renderer.render(pc_xyz, pc_colors, cam_positions, cam_rotations)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or not renderer.has_pytorch3d:
            raise
        import torch
        torch.cuda.empty_cache()
        print("CUDA out of memory during rendering. Retrying with PyTorch3D on CPU...")
        renderer.device = torch.device("cpu")
        render_output = renderer.render(pc_xyz, pc_colors, cam_positions, cam_rotations)

    output_dir = Path(args.output_dir)
    save_rendered_views(render_output.images, output_dir, cam_positions)
    print(f"Saved 10 rendered views to: {output_dir}")


if __name__ == "__main__":
    main()

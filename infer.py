"""Run Arbitr3D on one point cloud or mesh from the repository root."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Absolute path, or path relative to data.base_path")
    parser.add_argument("--category", required=True, help="Category used to select the knowledge files")
    parser.add_argument("--classes", help="Comma-separated part names; defaults to PartNetE_meta.json")
    parser.add_argument("--config", default=os.environ.get("CONFIG_PATH", "config/config.yaml"))
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--gpu", type=int, default=0, help="Logical GPU index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num-points", type=int, default=0, help="Mesh sample count; 0 uses mesh vertices")
    args = parser.parse_args()

    import yaml
    from utils.runtime_config import resolve_api_key, resolve_model_name

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    resolve_api_key(config["mllm"])
    resolve_model_name(config["mllm"])
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(config["data"]["base_path"]) / input_path
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if not Path(config["model"]["sam_weight_path"]).is_file():
        parser.error("SAM checkpoint is missing; set model.sam_weight_path in your config.")
    if args.num_points < 0:
        parser.error("--num-points must be nonnegative")

    classes = args.classes
    if not classes:
        with open(Path(__file__).with_name("PartNetE_meta.json"), encoding="utf-8") as stream:
            metadata = json.load(stream)
        if args.category not in metadata:
            parser.error("Unknown category; supply --classes with the desired part names.")
        classes = ", ".join(metadata[args.category])
    if not any(part.strip() for part in classes.split(",")):
        parser.error("--classes must contain at least one part name")

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the supported inference setup.")
    if not 0 <= args.gpu < torch.cuda.device_count():
        parser.error("--gpu must be a valid logical index within CUDA_VISIBLE_DEVICES")
    try:
        from pytorch3d import _C  # noqa: F401 -- also check the compiled extension
    except ImportError:
        parser.error("Install PyTorch3D with its native extensions; see docs/installation.md.")

    from pipeline import Arbitr3DPipeline
    output = Path(args.output)
    pipeline = Arbitr3DPipeline(args.config, gpu_index=args.gpu)
    pcd, labels, class_mapping = pipeline.run(
        point_cloud_filename=str(input_path.resolve()),
        prompt_classes_str=classes,
        vis_dir=str(output),
        object_category=args.category,
        num_points=args.num_points,
    )
    pipeline.visualizer.visualize_3d_result(
        pcd, labels, save_path=str(output / "prediction.ply"), class_to_id=class_mapping,
    )
    np.save(output / "labels.npy", np.asarray(labels, dtype=np.int64))
    (output / "classes.json").write_text(
        json.dumps(class_mapping, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved predictions to {output.resolve()}")


if __name__ == "__main__":
    main()

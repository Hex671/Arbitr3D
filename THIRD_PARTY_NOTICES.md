# Third-party code and resources

The root MIT license applies to Arbitr3D-authored code. Existing third-party licenses and notices continue to apply to their respective files.

## Bundled code

| Files | Upstream | License / attribution |
| --- | --- | --- |
| `partition/graphs.py` | [SuperPoint Graph](https://github.com/loicland/superpoint_graph) | Loic Landrieu; MIT text in `partition/SUPERPOINT_GRAPH_LICENSE` |
| `partition/cut-pursuit/` | [Cut-Pursuit](https://github.com/loicland/cut-pursuit) | Loic Landrieu; original MIT text retained in `partition/cut-pursuit/LICENSE` |

These files were retained from the local research implementation. They are not claimed to match a particular upstream commit. The Cut-Pursuit README retains its original algorithm references.

## External dependencies

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything): Meta's segmentation model. Installed as a dependency; its source and weights are not vendored.
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d): multi-view rendering. Installed separately; its source is not vendored.
- [Open3D](https://github.com/isl-org/Open3D), [Trimesh](https://github.com/mikedh/trimesh), and [Mitsuba](https://github.com/mitsuba-renderer/mitsuba3): geometry and visualization dependencies.
- The OpenAI Python SDK provides the client interface to the configured MLLM service; model/service terms are separate from this code's license.

## Datasets and related implementations

- [PartSTAD](https://github.com/KAIST-Visual-AI-Group/PartSTAD) and the PartNet-Ensembled dataset: data preparation context and camera-view conventions. `PartNetE_meta.json` supplies the category/part-name lookup used by the evaluation scripts.
- [SAMPart3D / PartObjaverse-Tiny](https://github.com/Pointcept/SAMPart3D): mesh dataset and semantic annotations, downloaded separately under the dataset provider's terms.
- [CoPart / PartVerse](https://huggingface.co/datasets/dscdyc/partverse/tree/main): mesh and caption data for the optional RES path, downloaded separately.

Consult upstream repositories and dataset cards for their licenses and citation instructions. No dataset, third-party model weight, or commercial API access is licensed by the root MIT file.

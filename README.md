# Arbitr3D

MLLM-guided 3D part segmentation with multi-view masks, geometric evidence, and topology-aware review.

[中文说明](README_zh-CN.md) · [Installation](docs/installation.md) · [Data and evaluation](docs/data-and-evaluation.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

Arbitr3D renders a point cloud or mesh into multiple views, extracts masks with SAM, assigns part semantics with a vision-language model, reviews inconsistent predictions using 3D geometry and topology, and fuses the results back into a 3D segmentation. The repository also includes a separate PartVerse referring-expression segmentation (RES) path.

## Release scope

This is a source-code release of the research implementation. It includes inference, knowledge generation, evaluation, ablation, and visualization utilities. Datasets, SAM weights, experiment outputs, manuscript files, and local credentials are not distributed. Only a few example category knowledge files are included; generate the knowledge for other categories before evaluation.

The intended inference environment is **Linux, Python 3.10, an NVIDIA CUDA GPU, PyTorch3D, and a vision-capable OpenAI-compatible API**. The original environment used PyTorch 2.3.1 with CUDA 11.8. See the installation guide for native dependencies and the optional Cut-Pursuit extension.

## Quick start

```bash
git clone https://github.com/Hex671/Arbitr3D.git
cd Arbitr3D
# Complete docs/installation.md before running inference.
cp config/config.yaml config/config.local.yaml
```

Edit `config/config.local.yaml` to set the dataset paths, `model.sam_weight_path`, and your MLLM endpoint/model. All relative paths are resolved from the repository root; run commands there.

Set credentials through environment variables. `mllm.api_key_env` is the **name of an environment variable**, never the key itself. For a local server without authentication:

```bash
export OPENAI_API_KEY=EMPTY
export MLLM_BASE_URL=http://localhost:8000/v1
export MLLM_MODEL_NAME=your-vision-model
export CONFIG_PATH=config/config.local.yaml
```

For a hosted endpoint, set `OPENAI_API_KEY` securely in your shell and use that endpoint's base URL and model identifier. The model must accept image inputs. `.env.example` documents the variables; `.env` files are **not loaded automatically**.

Run one PartNetE point cloud:

```bash
python infer.py --config "$CONFIG_PATH" \
  --input Chair/179/pc.ply --category Chair --output outputs/chair_179
```

Or provide your own mesh and part names:

```bash
python infer.py --config "$CONFIG_PATH" \
  --input /path/to/chair.glb --category Chair \
  --classes "back,seat,leg,arm" --num-points 30000 \
  --output outputs/my_chair --gpu 0
```

`--gpu` is a logical CUDA index: with `CUDA_VISIBLE_DEVICES=2`, use `--gpu 0`. The output contains `prediction.ply`, `labels.npy`, `classes.json`, and intermediate renderings/knowledge. Labels correspond to the normalized points saved in the predicted point cloud; mesh sampling does not produce a label for every original mesh face.

## Knowledge and evaluation

```bash
# Uses data.knowledge_base_path (the separate PartNetE few-shot split).
python scripts/generate_unified_knowledge.py --category Chair

# PartNetE batch evaluation; reads CONFIG_PATH.
TARGET_CATEGORY=Chair python batch_test.py

# PartObjaverse-Tiny batch evaluation.
python batch_test_objaverse.py --config "$CONFIG_PATH" \
  --categories Animals --max-instances 1

# Independent PartVerse RES path.
python eval_partverse_res.py --config "$CONFIG_PATH" \
  --uid YOUR_INSTANCE_UID --max-queries 2
```

Dataset layouts, knowledge-file expectations, output locations, and additional commands are documented in [Data and evaluation](docs/data-and-evaluation.md). Knowledge generation and inference call the configured API and may incur provider charges.

## Code map

| Path | Purpose |
| --- | --- |
| `infer.py`, `pipeline.py` | Single-object CLI and main segmentation pipeline |
| `res_pipeline.py`, `eval_partverse_res.py` | PartVerse referring-expression segmentation |
| `module_render/` | Multi-view point-cloud/mesh rendering and camera poses |
| `module_2d_fm/` | SAM masks, MLLM classification, mask refinement |
| `module_3d_lift/`, `module_3d_optim/` | Projection, superpoints, geometry, topology, fusion |
| `module_mllm/` | Category/instance knowledge and second-round review |
| `data_prep/`, `evaluation/` | Data loaders, metrics, result visualization |
| `config/` | Portable defaults, prompts, and example knowledge |
| `scripts/` | Knowledge generation, ablations, and rendering tools |
| `partition/` | SuperPoint Graph utilities and Cut-Pursuit source |
| `tests/` | Configuration tests and existing regression checks |

## Validation

```bash
python -m compileall -q .
python -m unittest discover -s tests -p 'test_*.py' -v
# Requires the Python runtime dependencies, but no datasets, weights, or API calls:
python tests/regression_zero_intrusion.py
```

Release preparation passed Python syntax checks, five credential/configuration tests, and the six groups in the existing regression script. These checks cover module imports, interfaces, configuration, RES isolation, synthetic metric calculations, and loader query hashing. **They do not establish end-to-end inference or benchmark reproduction.** The release machine did not have PyTorch3D or the native Cut-Pursuit extension available for a full GPU run. GitHub Actions runs the lightweight syntax/configuration checks only.

MLLM outputs, model versions, mesh sampling, and generated knowledge can affect results. Record these settings when comparing experiments. The pure Python superpoint fallback differs from Cut-Pursuit; use the compiled extension for experiments intended to match that path.

## License and acknowledgments

Arbitr3D-authored code is released under the [MIT License](LICENSE). Bundled third-party code retains its own license notices; external datasets, models, and services retain their respective terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution to SAM, PyTorch3D, Cut-Pursuit, SuperPoint Graph, and dataset providers.

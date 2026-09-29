# Installation

Run all commands from the repository root unless stated otherwise. The target inference setup is Linux with an NVIDIA GPU. Windows can run the lightweight checks; the Open3D renderer fallback and native builds have different behavior and are not the reference inference path.

## Python and PyTorch

```bash
conda env create -f environment.yml
conda activate arbitr3d
python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

The Conda file installs Python libraries and build dependencies; PyTorch and PyTorch3D are installed separately because their CUDA/native builds must match your system. The requirements file provides compatibility bounds, not a fully resolved lockfile. Record `pip freeze` and `conda list` for reproducibility.

## PyTorch3D

Install a CUDA toolkit/compiler compatible with the chosen PyTorch build before building native extensions. The local source snapshot identifies itself as PyTorch3D 0.7.9, while the old environment notes named 0.7.8. The example below uses the available upstream 0.7.9 tag, whose [requirements list PyTorch 2.3.1](https://github.com/facebookresearch/pytorch3d/blob/v0.7.9/INSTALL.md). This native build was not executed during release preparation.

```bash
python -m pip install fvcore iopath
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
python -c "from pytorch3d import _C; from pytorch3d.renderer import PointsRasterizer"
```

See the [official PyTorch3D installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for compiler and platform requirements. Use a different compatible combination if your platform cannot build the original stack; such a combination needs separate validation.

## SAM

`requirements.txt` installs SAM directly from the official repository at commit `dca509fe793f601edb92606367a655c15ac00fdf`; Git must be available. The unmodified third-party source tree is not vendored. Download the SAM ViT-H checkpoint through the [official SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints), place it at `checkpoints/sam_vit_h_4b8939.pth`, or set `model.sam_weight_path` to its actual location. Keep `model.sam_model_type: vit_h` matched to that checkpoint.

## Cut-Pursuit

The supplied source under `partition/cut-pursuit/` needs a C++ compiler with OpenMP, CMake, Boost.Python/NumPy, and the active Python/NumPy development headers. The Conda environment includes Boost and Eigen. Build against the same Python environment used for inference:

```bash
cmake -S partition/cut-pursuit -B partition/cut-pursuit/build \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPYTHON_LIBRARY="$CONDA_PREFIX/lib/libpython3.10.so" \
  -DPYTHON_INCLUDE_DIR="$CONDA_PREFIX/include/python3.10" \
  -DBOOST_INCLUDEDIR="$CONDA_PREFIX/include" \
  -DBOOST_LIBRARYDIR="$CONDA_PREFIX/lib" \
  -DEIGEN3_INCLUDE_DIR="$CONDA_PREFIX/include/eigen3"
cmake --build partition/cut-pursuit/build --parallel
PYTHONPATH=partition/cut-pursuit/build/src python -c "import libcp"
```

Adjust the library names if your Python/Boost packaging differs. The pipeline looks for `libcp` in `partition/cut-pursuit/build/src`. A NetworkX-based fallback exists if compilation/import fails; it uses a different partitioning algorithm and should not be treated as numerically equivalent.

## MLLM endpoint

Arbitr3D uses the OpenAI Python SDK as an HTTP client for a vision-capable compatible endpoint. Provision a service separately, then configure `mllm.base_url` and `mllm.model_name` or the `MLLM_BASE_URL` and `MLLM_MODEL_NAME` environment variables. Set the environment variable named by `mllm.api_key_env`; use `EMPTY` only when your local server does not require authentication. No credentials are embedded in the distributed configuration.

## Optional rendering tools

```bash
python -m pip install -r requirements-visualization.txt
python scripts/render_pointcloud_views.py --help
python scripts/render_balls_mitsuba.py --help
```

Mitsuba is needed only by the corresponding visualization scripts. These rendering utilities can have additional display/GPU requirements distinct from the core pipeline.

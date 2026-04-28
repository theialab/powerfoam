# Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization

## Shrisudhan Govindarajan\*, Daniel Rebain\*, Dor Verbin, Kwang Moo Yi, Anish Prabhu, Andrea Tagliasacchi

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=githubpages&logoColor=white)](https://powerfoam.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.24994)

</div>

This repository contains the official implementation of **Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization**. It provides scripts for training, evaluation and benchmarking, plus an interactive viewer that can be used both to inspect trained checkpoints and to watch a model converge live during training. The code includes scripts for training and evaluation, as well as a real-time viewer that can be used to visualize trained models, or optionally to observe the progression of models as they train. Everything in this repository is non-final and subject to change as the project is still being actively developed.

Warning: this is an organic, free-range research codebase, and should be treated with the appropriate care when integrating it into any other software.

## Getting started

Start by cloning the repository:

```bash
git clone https://github.com/theialab/powerfoam.git
cd powerfoam
```

You will need a Linux environment with **Python 3.10**, **CUDA 12.x** and a CUDA-compatible GPU of **Compute Capability 7.0 or higher**. The reference build is `torch==2.9.1+cu128`, `warp-lang==1.10.0` on CUDA 12.8.

After creating your Python virtual environment, install PyTorch matching your CUDA build (cu128 below):

```bash
conda create -n powerfoam python=3.11 -y
conda activate powerfoam

pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Then install the rest of the Python dependencies:

```bash
pip install -r requirements.txt
```

### Optional: Metric3D normal/depth supervision

If you plan to use Metric3D-derived normals (`--use_metric3d True`, optionally combined with `--normal_supervision`), install the extra dependencies on top of the base requirements:

```bash
pip install -r metric3d_requirements.txt
```

The Metric3D ViT-Large checkpoint is downloaded from `torch.hub` on first use (`yvanyin/metric3d`).

### Optional: Viewer

The interactive viewer (`view.py`, and `--viewer` during training) needs `imgui-bundle`, `glfw` and `PyOpenGL`. They are listed in `requirements.txt` and installed by the command above.

## Dataset layout

Power Foam supports two dataset types, selected by the `dataset:` field in the yaml config:

| Config value | Reader                                                   |
| ------------ | -------------------------------------------------------- |
| `colmap`     | COLMAP `images/` + `sparse/0/` (DTU, MipNeRF-360, DL3DV) |
| `blender`    | Synthetic Blender (`transforms_*.json` + `*/r_*.png`)    |

We have tested our model on the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and [DL3DV](https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample) datasets.

Place each scene under `<data_path>/<scene>/`, e.g. `data/mipnerf360/garden/{images,sparse}/`. If your raw data is just a folder of images, you can run COLMAP on it via:

```bash
python prepare_colmap_data.py --data_dir data/your_own_data
```

This creates `data/your_own_data/sparse/0/{cameras,images,points3D}.bin` so the COLMAP loader can pick it up.

When `--use_metric3d True` is passed, the dataset reader also expects (or auto-creates on first run) a sibling `metric3d/` directory containing per-image `.pt` files with predicted depth, normals and confidence.

## Configuration files

Per-dataset defaults live under `configs/`:

```
configs/dl3dv.yaml
configs/mipnerf360_indoor.yaml
configs/mipnerf360_outdoor.yaml
configs/dtu.yaml
```

Add your own `configs/<dataset>.yaml` for new datasets — copy one of the shipped files and edit `dataset:`, `data_path:`, `scene:` and any optimisation overrides.

Every entry can also be overridden from the command line, e.g. `--scene garden`, `--iterations 30000`. The full schema is defined in `configs/__init__.py` (`Params` dataclass).

> **Camera model default**
> For the standard datasets shipped in `configs/`, `is_pinhole: true`. The rasterizer therefore uses the fast pinhole pipeline. **If your dataset has a non-pinhole camera (fisheye, distortion, etc.), pass `is_pinhole: false`** in your config file so the renderer switches to the generic precomputed-cone pipeline.

## Training

Training is launched with:

```bash
python train.py -c configs/<config_file>.yaml
```

where `<config_file>` is one of the shipped files in `configs/` or your own. You can optionally include the `--viewer` flag to train interactively, or use `view.py` to inspect saved checkpoints.

```bash
# Train MipNeRF-360 garden with the defaults from configs/mipnerf360_outdoor.yaml
python train.py -c configs/mipnerf360_outdoor.yaml

# Train with the live viewer attached
python train.py -c configs/mipnerf360_outdoor.yaml --viewer
```

Outputs are written to `output/<experiment_name>/` and contain:

- `config.yaml` — frozen copy of all CLI args
- `model.pt`    — periodic checkpoint
- `points.ply`  — point cloud snapshot
- `test/`       — composed RGB / normal / depth previews
- TensorBoard event files (`tensorboard --logdir output/<experiment>`)

## Evaluation

The standard test metrics can be computed with:

```bash
python test.py -c output/<checkpoint_directory>/config.yaml
```

## Benchmarking

Rendering speed can be computed with:

```bash
python benchmark.py -c output/<checkpoint_directory>/config.yaml
```

`benchmark.py` supports both the rasterizer and the ray tracer over the same set of power-diagram primitives:

```bash
# Rasterizer benchmark (default)
python benchmark.py -c output/<checkpoint_directory>/config.yaml

# Ray tracer benchmark (uses adjacency-walk ray traversal)
python benchmark.py -c output/<checkpoint_directory>/config.yaml --render_type raytrace
```

## Viewer

```bash
python view.py -c output/<checkpoint_directory>/config.yaml
```

The viewer can also be attached *during* training by passing `--viewer` to `train.py` — the training loop pauses cleanly while the renderer holds the state lock so you can inspect convergence in real time.

## Important flags reference

| Flag                    | Default       | Description                                                                                                                                                |
| ----------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--is_pinhole`          | `False`*      | When `True`, use the fast pinhole rasterizer. The standard configs ship with `is_pinhole: true`. **Set to `False` for fisheye / distorted cameras.**       |
| `--use_metric3d`        | `False`       | Pre-compute / load Metric3D depth + normals into the dataset. Required to use Metric3D normals as supervision.                                             |
| `--normal_supervision`  | `False`       | Enable external normal supervision. Combine with `--use_metric3d True` for Metric3D normals; otherwise uses finite-difference normals from rendered depth. |
| `--eval`                | `False`       | Use `train`/`test` splits. When `False`, the loader uses all views as `all`.                                                                               |
| `--viewer`              | `False`       | Launch the interactive viewer alongside training.                                                                                                          |
| `--dry_run`             | `False`       | Don't write any checkpoints or TensorBoard events.                                                                                                         |
| `--experiment_name`     | random uuid   | Subdirectory name under `output/`.                                                                                                                         |
| `--render_type`         | `rasterize`   | (`benchmark.py` only) `rasterize` or `raytrace`.                                                                                                           |

*`Params.is_pinhole` defaults to `False` in code, but every shipped yaml overrides it to `True` because the standard datasets use pinhole models. Override on the CLI if you need otherwise.

### Flag combinations for normal supervision

| `normal_supervision` | `use_metric3d` | What happens                                                                        |
| -------------------- | -------------- | ----------------------------------------------------------------------------------- |
| False (default)      | any            | Only the renderer's internal `normal_err` term — no external normals are consulted. |
| True                 | False          | External target = finite-difference normals from the rendered median-depth map.     |
| True                 | True           | External target = Metric3D predicted normals.                                       |

When `--normal_supervision` is on, training also evaluates the rendered depth map at the median quantile (`0.5`) so the bilateral / FD path or the validity mask can be computed.

## BibTeX

```bibtex
@article{govindarajan2026powerfoam,
  title   = {Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization},
  author  = {Govindarajan, Shrisudhan and Rebain, Daniel and Verbin, Dor and
             Yi, Kwang Moo and Prabhu, Anish and Tagliasacchi, Andrea},
  journal = {arXiv},
  year    = {2026},
}
```

# SurfaceVAE

Implementation of [**Variational Auto-Encoders without graph coarsening for fine mesh learning**](https://ieeexplore.ieee.org/document/9191133), ICIP 2020.

This repository is a full rewrite of the original implementation, with minor improvements and bug fixes. As a result, the performance is slightly better than the one of the original paper.
---

## Overview

Standard graph-based VAEs for 3D mesh reconstruction rely on progressive graph coarsening
(down-sampling in the encoder, up-sampling in the decoder). This work shows that coarsening is
unnecessary when the dataset shares a fixed graph topology. The key contributions are:

- **Generative Surface Networks** — a variational auto-encoder for 3D mesh based on graph spectral operators from [Surface Networks](https://arxiv.org/abs/1705.10819) (Kostrikov et al., CVPR 2018).
- **Mean-shape decoder initialisation** — the decoder is primed with the per-vertex mean of the
  training set, giving the network an informative smooth starting point and removing the need for
  complex deformation parametrisations.
- **Mean operator** — for geometry-dependent operators, the operator is computed once from the mean training shape and shared across all samples, enabling unconditional generation from the latent space without requiring any input mesh.
- **Graph Dirac operator** — a topology-only Dirac operator defined purely from the adjacency matrix, extending the geometry-dependent [Dirac operator of Crane et al.](https://www.cs.cmu.edu/~kmcrane/Projects/SpinTransformations/) to any chordal graph.

The model is evaluated on the **CoMA dataset** of extreme facial expressions and approaches the
reconstruction accuracy of CoMA (Ranjan et al., ECCV 2018) while using fewer parameters and no
graph coarsening.

---

## Dataset — CoMA

The [CoMA dataset](https://coma.is.tue.mpg.de/) contains ~20,000 triangular meshes of 12 subjects
performing extreme facial expressions, consistently registered on a shared graph of 5,023 vertices.

### Download

1. Register at [https://coma.is.tue.mpg.de/](https://coma.is.tue.mpg.de/) and accept the license.
2. Download the **mesh data** archive (`COMA_data.zip` or similar) from the Downloads page.
3. Extract the archive. The expected directory layout is:
   ```
   <data_dir>/
   └── COMA_data/
       ├── FaceTalk_170725_00137_vc/
       │   ├── bareteeth/
       │   │   ├── 000001.ply
       │   │   └── ...
       │   └── ...
       ├── FaceTalk_170728_03272_vc/
       └── ...
   ```

### Configure the data path

Edit `configs/experiment/user/user_settings.yaml` and set `data_dir` to point to the folder where `COMA_data` is located, or set the environment variable `DATASET_DIR` in a `.env` file at the project root:

```dotenv
DATASET_DIR=/path/to/data_parent_folder
```

The first run will automatically preprocess the `.ply` files into `.npy` arrays and cache the graph operators.
---

## Installation

The project requires **Python 3.13** and uses [uv](https://github.com/astral-sh/uv) for dependency
management.

```bash
# Install all core dependencies
uv sync

# Optional extras
uv sync --extra track_all   # TensorBoard, W&B, SQLAlchemy logging
uv sync --extra finetune    # Optuna hyperparameter tuning
uv sync --extra visualize   # PyVista + Seaborn for visualisation
```

---

## Usage

### Training

The default configuration trains with the **graph Laplacian** operator (`lap_graph_norm`),
8 latent dimensions, and 1000 epochs:

```bash
uv run train_vae.py
```

To train with the **graph Dirac** operator instead:

```bash
uv run train_vae.py model=dir_vae
```

Common overrides (Hydra syntax):

```bash
# Change number of epochs
uv run train_vae.py train.n_epochs=500

# Resume from a checkpoint (epoch number)
uv run train_vae.py user.load_checkpoint=500

# Disable GPU (force CPU)
uv run train_vae.py user.cpu=true

# Change the latent space dimension
uv run train_vae.py model.dim_latent=16
```


---

## Available operators

| Config key | Operator | Description |
|---|---|---|
| `lap_graph_norm` | Adjacency Laplacian | Normalised graph Laplacian from adjacency matrix only (default) |
| `dirac_graph_norm` | Adjacency Dirac | Topology-only Dirac operator, generalises to chordal graphs |
| `lap_beltrami` | Laplace-Beltrami | Cotangent-weighted Laplacian (geometry-dependent) |
| `dirac_norm` | Dirac (Crane et al.) | Original continuous Dirac operator (high memory cost) |

---

## Project structure

```
SurfaceVAE/
├── configs/              # Hydra configuration files
│   ├── experiment/       # Model, training, data, user settings
│   └── tuning/           # Optuna study settings
├── src/
│   ├── config/           # Experiment configuration and Hydra integration
│   ├── data/             # CoMA dataset loading, preprocessing, operators
│   ├── module/           # VAE encoder, decoder, layers
│   ├── train/            # Training loop, loss, learning schema, hooks
│   └── utils/            # Sparse utilities, indexable helpers, tuning tools
├── train_vae.py          # Main training entry point
├── tune_vae.py           # Hyperparameter tuning entry point
└── pyproject.toml
```

---

## Results on CoMA

### Interpolation Experiment (Generalisation to unseen frames)

The table below shows the performance of the model on the interpolation split (trained on 1000 epochs) using different operators, compared with the reference CoMA paper. Our unweighted topological graph operators achieve comparable performance with fewer parameters.

| Operator / Model | Error (mm) | % nodes < 1 mm | # Weights |
|---|---|---|---|
| **CoMA** (Ranjan et al., ECCV 2018) | 0.845 ± 0.99 | 72.6 | 33,856 |
| **SurfaceVAE** (`lap_graph_norm`) | 0.948 ± 0.89 | 67.4% | **29,427** |
| **SurfaceVAE** (`lap_beltrami`) | 1.023 ± 0.97 | 63.3% | **29,427** |
| **SurfaceVAE** (`dirac_graph_norm`) | 1.220 ± 1.13 | 56.1% | **29,427** |
| **SurfaceVAE** (`lap_beltrami_norm`) | 1.322 ± 1.71 | 59.4% | **29,427** |
| **SurfaceVAE** (`none` - no operator) | 1.550 ± 5.00 | 57.5% | 29,747 |
| **SurfaceVAE** (`lap_graph_norm`, no mean shape) | 2.250 ± 1.90 | 30.1% | **29,427** |

### Extrapolation Experiment (Generalisation to unseen expressions)

In this cross-validation experiment, one expression is left out during training and the model is evaluated on its reconstruction. Below is a comparison of the mean reconstruction error (mm) and median error (mm) per left-out expression between the reference **CoMA** (Mesh Autoencoder) and our **SurfaceVAE** (`lap_graph_norm` operator).

*Note: `mouth_extreme` was not run for SurfaceVAE.*

| Expression | CoMA Mean Error (mm) | CoMA Median (mm) | SurfaceVAE Mean Error (mm) | SurfaceVAE Median (mm) |
|---|---|---|---|---|
| **bareteeth** | 1.376 ± 1.536 | 0.856 | **1.363 ± 1.310** | 0.919 |
| **cheeks_in** | 1.288 ± 1.501 | 0.794 | **1.278 ± 1.323** | 0.857 |
| **eyebrow** | 1.053 ± 1.088 | 0.706 | **1.018 ± 0.884** | 0.756 |
| **high_smile** | **1.205 ± 1.252** | **0.772** | 1.220 ± 1.128 | 0.843 |
| **lips_back** | **1.193 ± 1.476** | **0.708** | 1.218 ± 1.231 | 0.826 |
| **lips_up** | 1.081 ± 1.192 | 0.656 | **1.060 ± 0.981** | 0.744 |
| **mouth_down** | 1.050 ± 1.183 | 0.654 | **1.035 ± 0.963** | 0.760 |
| **mouth_extreme** | 1.336 ± 1.820 | 0.738 | - | - |
| **mouth_middle** | **1.017 ± 1.192** | **0.610** | 1.022 ± 1.000 | 0.705 |
| **mouth_open** | 0.961 ± 1.127 | 0.583 | **0.955 ± 0.980** | 0.655 |
| **mouth_side** | 1.264 ± 1.611 | 0.730 | **1.217 ± 1.372** | 0.791 |
| **mouth_up** | 1.097 ± 1.212 | 0.683 | **1.063 ± 1.001** | 0.748 |
| **Average (11 runs)** | 1.144 | 0.725 | **1.114** | 0.782 |

---


## Citation

```bibtex
@inproceedings{8664135,
  abstract     = {{In this paper, we propose a Variational Auto-Encoder able to correctly reconstruct a fine mesh from a very low-dimensional latent space. The architecture avoids the usual coarsening of the graph and relies on pooling layers for the decoding phase and on the mean values of the training set for the up-sampling phase. We select new operators compared to previous work, and in particular, we define a new Dirac operator which can be extended to different types of graph structured data. We show the improvements over the previous operators and compare the results with the current benchmark on the Coma Dataset.}},
  author       = {{Vercheval, Nicolas and De Bie, Hendrik and Pizurica, Aleksandra}},
  booktitle    = {{IEEE International Conference on Image Processing (ICIP 2020), Proceedings}},
  isbn         = {{9781728163956}},
  issn         = {{1522-4880}},
  keywords     = {{Variational Autoencoder,Geometric Deep Learning,Mesh processing}},
  language     = {{eng}},
  location     = {{Abu Dhabi, United Arab Emirates}},
  pages        = {{2681--2685}},
  publisher    = {{IEEE}},
  title        = {{Variational auto-encoders without graph coarsening for fine mesh learning}},
  url          = {{http://doi.org/10.1109/ICIP40778.2020.9191189}},
  year         = {{2020}},
}
```

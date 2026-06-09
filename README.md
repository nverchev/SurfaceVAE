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

| Operator | Error (mm) | % nodes < 1 mm | # Weights |
|---|---|---|---|
| **CoMA** (Ranjan et al., ECCV 2018) | 0.845 ± 0.99 | 72.6 | 33,856 |

(NEW RESULTS WILL BE ADDED SOON)

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

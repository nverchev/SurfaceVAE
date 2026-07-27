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
The default configuration trains with the **graph Laplacian** operator (`lap_graph_norm` — the Laplacian with uniform weights defined from the adjacency matrix only) and 8 latent dimensions.

### Replicating the experiments

#### 1. Interpolation Experiment (Generalisation to unseen frames)
Predicting unseen frames:

##### Graph Laplacian with mean-shape decoder initialization
```bash
uv run train_vae.py
```

##### Graph Laplacian without mean-shape decoder initialization
```bash
uv run train_vae.py model.use_mean_shape=false
```

#### 2. Extrapolation Experiment (Generalisation to unseen expressions)
Predicting unseen expressions:

##### Leaving out bareteeth expression
```bash
uv run train_vae.py data/dataset=bareteeth
```

##### Leaving out cheeks_in expression
```bash
uv run train_vae.py data/dataset=cheeks_in
```

##### Leaving out eyebrow expression
```bash
uv run train_vae.py data/dataset=eyebrow
```

##### Leaving out high_smile expression
```bash
uv run train_vae.py data/dataset=high_smile
```

##### Leaving out lips_back expression
```bash
uv run train_vae.py data/dataset=lips_back
```

##### Leaving out lips_up expression
```bash
uv run train_vae.py data/dataset=lips_up
```

##### Leaving out mouth_down expression
```bash
uv run train_vae.py data/dataset=mouth_down
```

##### Leaving out mouth_extreme expression
```bash
uv run train_vae.py data/dataset=mouth_extreme
```

##### Leaving out mouth_middle expression
```bash
uv run train_vae.py data/dataset=mouth_middle
```

##### Leaving out mouth_open expression
```bash
uv run train_vae.py data/dataset=mouth_open
```

##### Leaving out mouth_side expression
```bash
uv run train_vae.py data/dataset=mouth_side
```

##### Leaving out mouth_up expression
```bash
uv run train_vae.py data/dataset=mouth_up
```

#### 3. Identity Experiment (Generalisation to unseen subject identities)
Predicting unseen subjects:

##### Graph Laplacian (default)
```bash
uv run train_vae.py data/dataset=identity train.objective.beta=1 final=true
```

##### Laplace-Beltrami Stiff (geometry-dependent)
```bash
uv run train_vae.py data/dataset=identity train.objective.beta=1 final=true model.operator=lap_stiff
```
*Run reference:*
`[2026-07-25 19:04:14] - Experiment: main_data_dataset=identity_train.objective.beta=1_model.operator=lap_stiff_final - Stopping run: 2026-07-23@20h52m39s`

##### Laplace-Beltrami (geometry-dependent, normalized)
```bash
uv run train_vae.py data/dataset=identity train.objective.beta=1 final=true model.operator=lap_beltrami
```


##### Discrete Dirac operator with stiffness scaling
```bash
uv run train_vae.py data/dataset=identity train.objective.beta=1 final=true model.operator=dirac_stiff
```


---

### Visualisation and Evaluation

You can generate, reconstruct, or interpolate mesh samples using the visualization scripts below.

> [!IMPORTANT]
> **Use `user.on_the_fly=true` to avoid precomputing all operators:**
> When running evaluation or visualization scripts (especially for geometry-dependent operators like `lap_stiff`, `lap_beltrami`, `dirac`, `dirac_stiff`), pass `user.on_the_fly=true`. This calculates operators dynamically on-the-fly for the evaluation batch, completely avoiding the lengthy processing of precomputing and storing operators for all ~20,000 meshes in the dataset.

- **Generate random samples:**
  ```bash
  uv run generate_samples.py final=true user.on_the_fly=true
  ```

- **Reconstruct dataset samples:**
  ```bash
  uv run reconstruct_samples.py final=true user.on_the_fly=true
  ```

- **Interpolate latent vectors:**
  ```bash
  uv run interpolate_samples.py final=true user.on_the_fly=true
  ```

- **Tour latent space along a closed trajectory:**
  ```bash
  uv run tour_latent_space.py final=true user.on_the_fly=true
  ```

- **Visualise operator eigenvectors:**
  ```bash
  uv run visualise_eigenvectors.py final=true user.on_the_fly=true
  ```

- **Analyze operator eigenvalue spectrum across dataset:**
  ```bash
  uv run analyze_operator_spectrum.py final=true user.on_the_fly=true model.operator=dirac
  ```

---

## Available operators

| Config key | Operator | Description |
|---|---|---|
| `lap_graph_norm` | Adjacency Laplacian | Symmetric normalised graph Laplacian with uniform weights (from adjacency matrix only, default) |
| `lap_stiff` | Laplace-Beltrami Stiff | Cotangent stiffness matrix (geometry-dependent, unnormalized) |
| `lap_beltrami` | Laplace-Beltrami | Mass-normalised cotangent Laplace-Beltrami operator (geometry-dependent) |
| `dirac_graph_norm` | Adjacency Dirac | Topology-only Dirac operator, generalises to chordal graphs |
| `dirac` | Continuous Dirac | Continuous area-normalized Dirac operator (geometry-dependent) |
| `dirac_stiff` | Dirac Stiff | Continuous Dirac operator with stiffness scaling (geometry-dependent) |

---

## Project structure

```
SurfaceVAE/
├── configs/                      # Hydra configuration files
│   ├── experiment/               # Model, training, data, user settings
│   └── tuning/                   # Optuna study settings
├── src/
│   ├── config/                   # Experiment configuration and Hydra integration
│   ├── data/                     # CoMA dataset loading, preprocessing, operators
│   ├── module/                   # VAE encoder, decoder, layers
│   ├── train/                    # Training loop, loss, learning schema, hooks
│   └── utils/                    # Sparse utilities, indexable helpers, tuning tools
├── train_vae.py                  # Main training entry point
├── tune_vae.py                   # Hyperparameter tuning entry point
├── analyze_operator_spectrum.py  # Operator spectrum analysis & CSV export entry point
└── pyproject.toml
```

---

## Results on CoMA

### Interpolation Experiment (Generalisation to unseen frames)

The table below shows the performance of the model on the interpolation split using the Graph Laplacian operator (`lap_graph_norm`), compared with the official CoMA repository implementation (retrained on the clean sorted split).

> [!NOTE]
> **CoMA Settings & File Ordering:**
> The CoMA baseline was trained using the official CoMA repository code adapted to match the original architecture parameters (decoder filter sequence `dec_F = [32, 32, 16, 16]`, totaling 34,680 weights / ~33.8k parameters). All dataset splits are generated by sorting raw `.ply` file paths deterministically (`sorted(glob.glob('/scratch/dataset/COMA_data/*/*/*.ply'))`) prior to splitting, eliminating data leakage and ensuring 100% alignment with SurfaceVAE.

| Model | Operator | Error (mm) | % nodes < 1 mm | # Weights |
|---|---|---|---|---|
| **CoMA** (Official Repository) | Graph Laplacian (with coarsening) | 0.893 ± 1.03 | 71.7 | 34,680 |
| **SurfaceVAE (no mean shape)** | Graph Laplacian (`lap_graph_norm`) | 2.031 ± 1.862 | 36.5 | **33,779** |
| **SurfaceVAE** | Laplace-Beltrami (`lap_beltrami`) | 0.838 ± 0.832 | 72.4 | **33,779** |
| **SurfaceVAE** | Laplace-Beltrami Stiff (`lap_stiff`) | 0.814 ± 0.831 | 73.6 | **33,779** |
| **SurfaceVAE** | Continuous Dirac (`dirac`) | TBD | TBD | **33,779** |
| **SurfaceVAE** | Dirac Stiff (`dirac_stiff`) | TBD | TBD | **33,779** |
| **SurfaceVAE** | Graph Dirac (`dirac_graph_norm`) | 0.988 ± 0.948 | 66.6 | **33,779** |
| **SurfaceVAE** | Graph Laplacian (`lap_graph_norm`) | **0.768 ± 0.764** | **75.8** | **33,779** |

### Extrapolation Experiment (Generalisation to unseen expressions)

In this cross-validation experiment, one expression is left out during training and the model is evaluated on its reconstruction. Below is a comparison of the mean reconstruction error (mm) and median error (mm) per left-out expression between the official **CoMA** implementation and our **SurfaceVAE** (`lap_graph_norm` operator).

| Expression | CoMA Mean Error (mm) | CoMA Median (mm) | SurfaceVAE Mean Error (mm) | SurfaceVAE Median (mm) |
|---|---|---|---|---|
| **bareteeth** | 1.609 ± 2.053 | 0.991 | **1.169 ± 1.184** | **0.768** |
| **cheeks_in** | 1.314 ± 1.557 | 0.812 | **1.176 ± 1.325** | **0.725** |
| **eyebrow** | 1.045 ± 1.057 | 0.709 | **0.924 ± 0.871** | **0.663** |
| **high_smile** | 1.349 ± 1.598 | 0.803 | **1.065 ± 0.991** | **0.744** |
| **lips_back** | 1.192 ± 1.563 | 0.671 | **1.052 ± 1.186** | **0.657** |
| **lips_up** | 1.096 ± 1.269 | 0.640 | **0.989 ± 1.005** | **0.643** |
| **mouth_down** | 1.026 ± 1.200 | 0.651 | **0.880 ± 0.923** | **0.592** |
| **mouth_extreme** | 1.496 ± 2.457 | **0.796** | **1.296 ± 1.508** | 0.814 |
| **mouth_middle** | 1.041 ± 1.221 | 0.618 | **0.876 ± 0.896** | **0.583** |
| **mouth_open** | 0.942 ± 1.199 | 0.560 | **0.839 ± 0.981** | **0.532** |
| **mouth_side** | 1.360 ± 2.084 | 0.726 | **1.106 ± 1.290** | **0.689** |
| **mouth_up** | 1.094 ± 1.218 | 0.690 | **0.943 ± 0.935** | **0.644** |
| **Average** | 1.214 (12 runs) | 0.722 (12 runs) | **1.026 (12 runs)** | **0.671 (12 runs)** |

### Identity Experiment (Generalisation to unseen subject identities)

In this experiment, one subject identity is left out during training and the model is evaluated on its reconstruction of that unseen subject. Below is a comparison of the graph-based (topology-only) operators (Graph Laplacian and Graph Dirac) trained with $\beta = 1.0$.

We focus on log-likelihood metrics—specifically, the Reconstruction negative log-likelihood (NLL) and KL Divergence (KLD)—along with the mean reconstruction error (mm).

| Operator / Model | Latent Dim | Mean Error (mm) | Reconstruction NLL | KL Divergence (KLD) |
|---|---|---|---|---|
| **SurfaceVAE** (`lap_graph_norm`) | 8 | 3.167 | 325.0 | **17.3** |
| **SurfaceVAE** (`lap_stiff`) | 8 | **2.602** | **267.0** | 18.1 |



#### Likelihood Assumptions
The Reconstruction NLL is computed assuming a **spherical Laplace likelihood** on the normalized 3D vertex coordinates:
$$p(x \mid \mu, b) \propto \exp(-\|x - \mu\|_2 / b)$$
yielding a Reconstruction NLL of $3 \log(b) + \frac{\|x - \mu\|_2}{b}$ per vertex, where $\mu$ is the reconstructed vertex coordinate and $b = \exp(\text{logvar})$ is the scale parameter. Under this assumption, minimizing the reconstruction loss is equivalent to minimizing the Mean Absolute Error (MAE) on the per-vertex Euclidean distances in 3D coordinate space.

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

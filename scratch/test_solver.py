import time
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from src.config import get_config_all, Experiment
from src.data import get_active_dataset
from src.data.operators import (
    build_laplace_beltrami,
    compute_mesh_geometry,
    compute_vertex_dual_areas,
)
from src.data.split import Partitions

cfg = get_config_all(["model=lap_vae"])
exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
with exp.create_run(resume=True):
    dataset = get_active_dataset().get_split(Partitions.train)
    faces = dataset.faces
    vertices = dataset[36][0].x.numpy()

    print("Building operators...")
    L_cot = build_laplace_beltrami(vertices, faces)
    _, face_areas = compute_mesh_geometry(vertices, faces)
    vertex_dual_areas = compute_vertex_dual_areas(faces, face_areas, vertices.shape[0])

    d_inv_sqrt = 1.0 / np.sqrt(vertex_dual_areas)
    D_inv_sqrt_mat = sparse.diags(d_inv_sqrt, 0)
    L_sym = D_inv_sqrt_mat @ L_cot @ D_inv_sqrt_mat

    print("Timing L_cot with sigma=1e-2...")
    try:
        t0 = time.time()
        val, vec = spla.eigsh(L_cot.tocsc(), k=6, sigma=1e-2, which="LM")
        print("L_cot shift-invert took:", time.time() - t0)
        print("Eigenvalues:", val)
    except Exception as e:
        print("L_cot shift-invert failed:", e)

    print("Timing L_sym with sigma=1e-2...")
    try:
        t0 = time.time()
        val, vec = spla.eigsh(L_sym.tocsc(), k=6, sigma=1e-2, which="LM")
        print("L_sym shift-invert took:", time.time() - t0)
        print("Eigenvalues:", val)
    except Exception as e:
        print("L_sym shift-invert failed:", e)

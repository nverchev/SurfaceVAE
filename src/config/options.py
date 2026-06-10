"""Defines all the options in the specifications."""

import enum


class Datasets(enum.StrEnum):
    """Dataset choices."""

    COMA = enum.auto()
    COMA_EXTRAPOLATION = enum.auto()


class Expressions(enum.StrEnum):
    """Available facial expressions in the COMA dataset."""

    bareteeth = enum.auto()
    cheeks_in = enum.auto()
    eyebrow = enum.auto()
    high_smile = enum.auto()
    lips_back = enum.auto()
    lips_up = enum.auto()
    mouth_down = enum.auto()
    mouth_extreme = enum.auto()
    mouth_middle = enum.auto()
    mouth_open = enum.auto()
    mouth_side = enum.auto()
    mouth_up = enum.auto()


class ModelOperators(enum.StrEnum):
    """Available graph VAE operators."""

    none = enum.auto()
    """Pointwise baseline, no operators/graph structure used."""

    lap_beltrami = enum.auto()
    """Stiffness matrix from cotangent-weighted Laplace-Beltrami operator."""

    lap_beltrami_norm = enum.auto()
    """Area-normalized cotangent-weighted Laplace-Beltrami operator (LBO)."""

    lap_graph_norm = enum.auto()
    """Symmetric normalized graph Laplacian (unweighted, purely topological, computed from fixed connectivity)."""

    dirac_norm = enum.auto()
    """Continuous coordinate-dependent area-normalized Dirac operator (D)."""

    dirac_graph_norm = enum.auto()
    """Combinatorial/topological unweighted normalized graph Dirac operator (D_tilde)."""


class LossTypes(enum.StrEnum):
    """Loss function settings."""

    elbo = enum.auto()
    """Standard Evidence Lower Bound (ELBO)."""


class Schedulers(enum.StrEnum):
    """Scheduler choices."""

    constant = enum.auto()
    cosine = enum.auto()
    exponential = enum.auto()

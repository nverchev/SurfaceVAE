"""Defines all the options in the specifications."""

import enum


class Datasets(enum.StrEnum):
    """Dataset choices."""

    COMA_INTERPOLATION = enum.auto()
    COMA_EXTRAPOLATION = enum.auto()
    COMA_IDENTITY = enum.auto()


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


class Identities(enum.StrEnum):
    """Available subject identities in the COMA dataset."""

    FaceTalk_170725_00137_TA = "FaceTalk_170725_00137_TA"
    FaceTalk_170728_03272_TA = "FaceTalk_170728_03272_TA"
    FaceTalk_170731_00024_TA = "FaceTalk_170731_00024_TA"
    FaceTalk_170809_00138_TA = "FaceTalk_170809_00138_TA"
    FaceTalk_170811_03274_TA = "FaceTalk_170811_03274_TA"
    FaceTalk_170811_03275_TA = "FaceTalk_170811_03275_TA"
    FaceTalk_170904_00128_TA = "FaceTalk_170904_00128_TA"
    FaceTalk_170904_03276_TA = "FaceTalk_170904_03276_TA"
    FaceTalk_170908_03277_TA = "FaceTalk_170908_03277_TA"
    FaceTalk_170912_03278_TA = "FaceTalk_170912_03278_TA"
    FaceTalk_170913_03279_TA = "FaceTalk_170913_03279_TA"
    FaceTalk_170915_00223_TA = "FaceTalk_170915_00223_TA"


class ModelOperators(enum.StrEnum):
    """Available graph VAE operators."""

    none = enum.auto()
    """Pointwise baseline, no operators/graph structure used."""

    lap_beltrami = enum.auto()
    """Stiffness matrix from cotangent-weighted Laplace-Beltrami operator."""

    lap_beltrami_norm = enum.auto()
    """Area-normalized cotangent-weighted Laplace-Beltrami operator."""

    lap_graph_norm = enum.auto()
    """Symmetric normalized graph Laplacian."""

    dirac = enum.auto()
    """Continuous coordinate-dependent area-normalized Dirac operator."""

    dirac_norm = enum.auto()
    """Continuous coordinate-dependent right-normalized Dirac operator."""

    dirac_graph_norm = enum.auto()
    """Combinatorial/topological unweighted normalized graph Dirac operator."""


class LossTypes(enum.StrEnum):
    """Loss function settings."""

    elbo = enum.auto()
    """Standard Evidence Lower Bound (ELBO)."""


class Schedulers(enum.StrEnum):
    """Scheduler choices."""

    constant = enum.auto()
    cosine = enum.auto()
    exponential = enum.auto()

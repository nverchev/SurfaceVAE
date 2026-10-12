"""Defines all the options in the specifications."""

import enum


class Datasets(enum.StrEnum):
    """Dataset choices."""

    COMA_INTERPOLATION = enum.auto()
    COMA_EXTRAPOLATION = enum.auto()
    COMA_IDENTITY = enum.auto()


class Expressions(enum.StrEnum):
    """Available facial expressions in the COMA dataset."""

    BARETEETH = enum.auto()
    CHEEKS_IN = enum.auto()
    EYEBROW = enum.auto()
    HIGH_SMILE = enum.auto()
    LIPS_BACK = enum.auto()
    LIPS_UP = enum.auto()
    MOUTH_DOWN = enum.auto()
    MOUTH_EXTREME = enum.auto()
    MOUTH_MIDDLE = enum.auto()
    MOUTH_OPEN = enum.auto()
    MOUTH_SIDE = enum.auto()
    MOUTH_UP = enum.auto()


class Identities(enum.StrEnum):
    """Available subject identities in the COMA dataset."""

    FaceTalk_170725_00137_TA = 'FaceTalk_170725_00137_TA'
    FaceTalk_170728_03272_TA = 'FaceTalk_170728_03272_TA'
    FaceTalk_170731_00024_TA = 'FaceTalk_170731_00024_TA'
    FaceTalk_170809_00138_TA = 'FaceTalk_170809_00138_TA'
    FaceTalk_170811_03274_TA = 'FaceTalk_170811_03274_TA'
    FaceTalk_170811_03275_TA = 'FaceTalk_170811_03275_TA'
    FaceTalk_170904_00128_TA = 'FaceTalk_170904_00128_TA'
    FaceTalk_170904_03276_TA = 'FaceTalk_170904_03276_TA'
    FaceTalk_170908_03277_TA = 'FaceTalk_170908_03277_TA'
    FaceTalk_170912_03278_TA = 'FaceTalk_170912_03278_TA'
    FaceTalk_170913_03279_TA = 'FaceTalk_170913_03279_TA'
    FaceTalk_170915_00223_TA = 'FaceTalk_170915_00223_TA'


class ModelOperators(enum.StrEnum):
    """Available graph VAE operators."""

    LAP_STIFF = enum.auto()
    """Stiffness matrix S from cotangent-weighted Laplace-Beltrami."""

    LAP_BELTRAMI = enum.auto()
    """Discrete Laplace-Beltrami operator M^{-1}S (mass-normalised cotangent Laplacian)."""

    LAP_GRAPH_NORM = enum.auto()
    """Symmetric normalised graph Laplacian L_norm = D^{-1/2} L_comb D^{-1/2}."""

    DIRAC = enum.auto()
    """Dirac pair; DiA @ Di = -M^{-1}S (real part)."""

    DIRAC_STIFF = enum.auto()
    """Dirac pair with D_A^{1/2} row-scaling; DiA_stiff @ Di_stiff = -S (real part)."""

    DIRAC_GRAPH_NORM = enum.auto()
    """Combinatorial Dirac pair; DiA @ Di = -L_norm (real part)."""


class LossTypes(enum.StrEnum):
    """Loss function settings."""

    ELBO = enum.auto()
    """Standard Evidence Lower Bound (ELBO)."""


class Schedulers(enum.StrEnum):
    """Scheduler choices."""

    CONSTANT = enum.auto()
    COSINE = enum.auto()
    EXPONENTIAL = enum.auto()

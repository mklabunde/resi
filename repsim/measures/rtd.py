import numpy.typing as npt

try:
    import rtd
except ImportError:
    rtd = None
import torch
from loguru import logger
from repsim.measures.utils import align_spatial_dimensions
from repsim.measures.utils import flatten
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.measures.utils import SHAPE_TYPE
from repsim.measures.utils import to_numpy_if_needed
import numpy as np


def representation_topology_divergence(
    R: torch.Tensor | npt.NDArray,
    Rp: torch.Tensor | npt.NDArray,
    shape: SHAPE_TYPE,
    pdist_device: str,
    trials: int = 10,
    batch: int = 500,
) -> float:
    """
    Compute the Representation Topology Divergence (RTD) [Barannikov et al., 2021] between two representations.
    Because RTD computes distances on a subset of the representations, subsequent runs can yield different results.
    """
    if pdist_device:
        assert pdist_device.startswith("cuda") or pdist_device.startswith(
            "cpu"
        ), "device must be a CUDA or CPU device"

    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    return rtd.rtd(R, Rp, pdist_device, trials=trials, batch=batch)


class RTD(RepresentationalSimilarityMeasure):
    def __init__(self, trials: int = 10, batch: int = 500, pdist_device: str = "cuda"):
        super().__init__(
            sim_func=representation_topology_divergence,  # type:ignore
            larger_is_more_similar=False,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )
        self.trials = trials
        self.batch = batch
        self.pdist_device = pdist_device
        logger.info(
            "RTD will use cuda devices to compute barcodes. It is not possible to specify which GPU directly. "
            "Use the CUDA_VISIBLE_DEVICES environment variable to specify which GPU to use."
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if not torch.cuda.is_available():
            return np.nan

        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)  # type:ignore
            shape = "nd"

        return self.sim_func(
            R, Rp, shape, pdist_device=self.pdist_device, trials=self.trials, batch=self.batch  # type:ignore
        )

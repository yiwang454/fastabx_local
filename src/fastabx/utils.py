"""Various utilities."""

import os
import sys

import torch.version


def load_dtw_extension() -> None:
    """Load the DTW extension.

    On Linux and Windows, we check that PyTorch has been installed with the correct CUDA version.
    """
    cuda_version = "12.4"
    if sys.platform in ["linux", "win32"] and torch.version.cuda != cuda_version:
        msg = (
            f"On Linux and Windows, the DTW extension requires PyTorch with CUDA {cuda_version}. "
            "It it not compatible with other CUDA versions, or with the CPU only version of PyTorch, "
            "even if you wanted to only use the CPU backend of the DTW. "
        )
        raise ImportError(msg)
    from . import _C  # type: ignore[attr-defined] # noqa: F401


def with_librilight_bug() -> bool:
    """Whether to reproduce the results from LibriLight ABX or not."""
    return os.getenv("FASTABX_WITH_LIBRILIGHT_BUG", "0") == "1"

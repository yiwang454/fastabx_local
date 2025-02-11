"""Various utilities."""

from dataclasses import dataclass, field
from typing import Literal

import polars as pl
import torch


def default_engine() -> Literal["cpu", "gpu"]:
    """Engine is 'gpu' if available, else 'cpu'."""
    try:
        pl.LazyFrame().collect(engine="gpu")
    except (ModuleNotFoundError, pl.exceptions.ComputeError):
        return "cpu"
    else:
        return "gpu"


@dataclass(frozen=True)
class Environment:
    """Store global environment variables."""

    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    engine: Literal["cpu", "gpu"] = field(default_factory=default_engine)

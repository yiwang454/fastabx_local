"""Safety checks for the public API."""

from itertools import chain

import polars as pl
import torch

PATTERN_INVALID_VALUE_IN_CELL = "[-_]"
INVALID_COLUMN_SUFFIX = ("_a", "_b", "_x")
INVALID_COLUMN_NAMES = {
    "group",
    "score",
    "length",
    "right",
    "left",
    "start",
    "end",
    "index",
    "lookup",
}

NDIM = 3


class ColumnNameError(ValueError):
    """Invalid name for a condition."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Invalid column name: {name}")


class ValueInColumnError(ValueError):
    """Invalid value in a column."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Invalid value in column: {name}")


def verify_task_conditions(conditions: list[str]) -> None:
    """Conditions should be unique strings."""
    if len(conditions) != len(set(conditions)):
        raise ValueError("Conditions should not contain duplicates")
    for cond in conditions:
        if not isinstance(cond, str):
            raise TypeError("Conditions should be strings")


def verify_dataset_labels(df: pl.DataFrame) -> None:
    """Check the columns: both the names and the values. Only for the columns used in the task."""
    for col, dtype in df.schema.items():
        if col in INVALID_COLUMN_NAMES or col.endswith(INVALID_COLUMN_SUFFIX):
            raise ColumnNameError(col)
        if dtype != pl.String:
            continue
        if df[col].str.contains(PATTERN_INVALID_VALUE_IN_CELL).any():
            raise ValueInColumnError(col)


def verify_subsampler_params(*sizes: int, seed: int) -> None:
    """All sizes must be positive integers."""
    if not all(isinstance(s, int) and s > 1 for s in sizes):
        raise TypeError("sizes should be positive integers")
    if not isinstance(seed, int):
        raise TypeError("seed should be an integer")


def verify_cell_integrity(
    a_sa: tuple[torch.Tensor, torch.Tensor],
    b_sb: tuple[torch.Tensor, torch.Tensor],
    x_sx: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Assert the integrity of a cell."""
    (a, sa), (b, sb), (x, sx) = a_sa, b_sb, x_sx
    if not (
        (a.ndim == b.ndim == x.ndim == NDIM)
        and (a.size(2) == b.size(2) == x.size(2))
        and (a.size(0) == sa.size(0))
        and b.size(0) == sb.size(0)
        and x.size(0) == sx.size(0)
    ):
        raise ValueError("Invalid cell dimensions")


def format_score_levels(levels: list[tuple[str, ...] | str]) -> list[tuple[str, ...]]:
    """Put all the levels in tuples."""
    formatted: list[tuple[str, ...]] = []
    for level in levels:
        if isinstance(level, str):
            formatted.append((level,))
        elif isinstance(level, tuple) and all(isinstance(x, str) for x in level):
            formatted.append(level)
        else:
            raise ValueError("`levels` should be list[tuple[str, ...] | str]")
    return formatted


def verify_score_levels(columns: list[str], levels: list[tuple[str, ...]]) -> None:
    """Levels should be unique columns of the DataFrame."""
    all_levels = list(chain.from_iterable(levels))
    unique_levels = set(all_levels)
    if len(all_levels) != len(unique_levels):
        raise ValueError("levels should not contain duplicates")
    if not unique_levels.issubset(set(columns)):
        raise ValueError("levels should be columns of the DataFrame")

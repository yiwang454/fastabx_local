"""Score the ABX task for each cell and collapse the scores into a final score."""

from collections.abc import Sequence
from pathlib import Path
import torch
import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from fastabx.distance import DistanceName, abx_on_cell, abx_on_cell_mean, distance_function
from fastabx.task import Task
from fastabx.verify import format_score_levels, verify_score_levels

MIN_CELLS_FOR_TQDM = 50

import logging, os, sys
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout, 
)
logger = logging.getLogger("score.py")

def pl_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    """Generate a Polars aggregation expression to take a weighted mean.

    https://github.com/pola-rs/polars/issues/7499#issuecomment-2569748864
    """
    values = pl.col(value_col)
    weights = pl.when(values.is_not_null()).then(weight_col)
    return weights.dot(values).truediv(weights.sum()).fill_nan(None)


class CollapseError(Exception):
    """Something wrong happened when collapsing the ``Score``."""

    def __init__(self, *, are_set: bool) -> None:
        if are_set:
            msg = "Cannot set `weighted=True` and `levels` at the same time."
        else:
            msg = "Either set `levels` or `weighted=True`."
        super().__init__(msg)


def score_details(cells: pl.DataFrame, *, levels: Sequence[tuple[str, ...] | str] | None) -> pl.DataFrame:
    """Collapse the scored cells and return the final scores and sizes for each (A, B) pairs."""
    if levels is None:
        if len(set(cells.columns) - {"index", "index_b", "score", "size"}) != 2:  # noqa: PLR2004
            raise CollapseError(are_set=False)
        levels = []
    cells = cells.select(~(cs.starts_with("index") | cs.ends_with("_x")))
    levels_in_tuples = format_score_levels(levels)
    verify_score_levels(cells.columns, levels_in_tuples)
    for level in levels_in_tuples:
        group_key = cs.exclude("score", "size", *level)
        # cells = cells.group_by(group_key, maintain_order=True).agg(pl.col("score").mean(), pl.col("size").sum())
        cells = cells.group_by(group_key, maintain_order=True).agg(
            # Filter out nulls *before* calculating the mean
            pl.col("score").filter(pl.col("score").is_not_nan()).mean(), # <--- CRITICAL CHANGE
            pl.col("size").sum()
        )
    return cells


class Score:
    """Compute the score of a :py:class:`.Task` using a given distance specified by ``distance_name``."""

    def __init__(self, task: Task, distance_name: DistanceName, frame_mean: bool = False) -> None:
        scores, sizes = [], []
        self.distance_name = distance_name
        self.frame_mean = frame_mean
        distance = distance_function(distance_name)
        if distance_name in ("cosine", "angular"):
            task.dataset.normalize_()
        for cell in tqdm(task, "Scoring each cell", disable=len(task) < MIN_CELLS_FOR_TQDM):
            # logger(f"about to call abx_on_cell")
            if not frame_mean:
                cell_score_tensor = abx_on_cell(cell, distance) #.item()
            else:
                cell_score_tensor = abx_on_cell_mean(cell, distance)
            if torch.isnan(cell_score_tensor).all() or not cell_score_tensor.ndim == 0:
                print(f"WARNING: Skipping score for cell '{cell.description}' due to invalid result: 'Empty/Invalid Tensor'")
                # continue # Skip this cell and don't append its score
                scores.append(float('nan'))
            else:
                scores.append(cell_score_tensor.item())
            sizes.append(len(cell))
        self._cells = task.cells.select(cs.exclude("description", "header")).with_columns(
            score=pl.Series(scores, dtype=pl.Float32), size=pl.Series(sizes)
        )

    @property
    def cells(self) -> pl.DataFrame:
        """Return the scored cells."""
        return self._cells

    @cells.setter
    def cells(self, _: pl.DataFrame) -> None:
        msg = "The `cells` attribute is read-only."
        raise AttributeError(msg)

    def __repr__(self) -> str:
        return f"Score({len(self.cells)} cells, {self.distance_name} distance)"

    def write_csv(self, file: str | Path) -> None:
        """Write the results of all the cells to a CSV file."""
        nested = [name for name, dtype in self.cells.schema.items() if dtype == pl.List]
        (self.cells.select(cs.exclude(nested)) if nested else self.cells).write_csv(file)

    def details(self, *, levels: Sequence[tuple[str, ...] | str] | None) -> pl.DataFrame:
        """Collapse the scored cells and return the final scores and sizes for each (A, B) pairs.

        :param levels: List of levels to collapse. The order matters a lot.
        """
        return score_details(self.cells, levels=levels)

    def collapse(self, *, levels: Sequence[tuple[str, ...] | str] | None = None, weighted: bool = False) -> float:
        """Collapse the scored cells into the final score.

        Use either `levels` or `weighted=True` to collapse the scores.

        :param levels: List of levels to collapse. The order matters a lot.
        :param weighted: Whether to collapse the scores using a mean weighted by the size of the cells.
        """
        if weighted:
            if levels is not None:
                raise CollapseError(are_set=True)
            return self.cells.select(pl_weighted_mean("score", "size")).item()
        return self.details(levels=levels)["score"].mean()

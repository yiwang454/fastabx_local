from fastabx import Task

import polars as pl
import torch
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence


@dataclass
class TaskSelectLen(Task):
    """
    A modified Dataset class that inherits from Dataset
    and uses a modified from_item method.
    """
    @classmethod
    def from_item(
        cls,
        item: str | Path,
        root: str | Path,
        frequency: int,
        *,
        feature_maker: Callable[[str | Path], torch.Tensor] = torch.load,
        extension: str = ".pt",
        file_col: str = "#file",
        onset_col: str = "onset",
        offset_col: str = "offset",
    ) -> "DatasetMod":

        labels = read_labels_mod(item, file_col, onset_col, offset_col)
        paths = find_all_files(root, extension)
        indices, data = load_data_from_item(paths, labels, frequency, feature_maker, file_col, onset_col, offset_col)
        return cls(labels=labels, accessor=InMemoryAccessor(indices, data))

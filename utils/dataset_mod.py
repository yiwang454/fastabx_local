from fastabx import Dataset, Subsampler, Task

import polars as pl
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Self, Any

def read_item_mod(item: str | Path) -> pl.DataFrame:
    """Read an item file."""

    try:
        df = pl.read_csv(item, separator=" ")
        print("[read_item_mod]DataFrame Header:", df.columns)
        return df.with_columns(df["onset"].str.to_decimal(), df["offset"].str.to_decimal())
    except pl.exceptions.ComputeError as error:
        raise InvalidItemFileError from error

def read_labels_mod(item: str | Path, file_col: str, onset_col: str, offset_col: str) -> pl.DataFrame:
    """Return the labels from the path to the item file."""
    schema_overrides = {file_col: pl.String, onset_col: pl.String, offset_col: pl.String}
    match ext := Path(item).suffix:
        case ".item":
            return read_item_mod(item)
        case ".csv":
            df = pl.read_csv(item, schema_overrides=schema_overrides)
            return df.with_columns(df[onset_col].str.to_decimal(), df[offset_col].str.to_decimal())
        case ".jsonl" | ".ndjson":
            df = pl.read_ndjson(item, schema_overrides=schema_overrides)
            return df.with_columns(df[onset_col].str.to_decimal(), df[offset_col].str.to_decimal())
        case _:
            msg = f"File extension {ext} is not supported. Supported extensions are .item, .csv, .jsonl, .ndjson."
            raise InvalidItemFileError(msg)

@dataclass(frozen=True)
class DatasetMod(Dataset):
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

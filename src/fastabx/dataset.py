"""Data utilities."""

import abc
import math
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
import torch
from polars.interchange.protocol import SupportsInterchange
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from fastabx.utils import Environment, with_librilight_bug
from fastabx.verify import verify_empty_datapoints

type FeatureMaker = Callable[[str | Path], torch.Tensor]
type ArrayLike = npt.ArrayLike


@dataclass(frozen=True)
class Batch:
    """Batch of padded data."""

    data: torch.Tensor
    sizes: torch.Tensor

    def __repr__(self) -> str:
        return f"Batch(data=Tensor(shape={self.data.shape}, dtype={self.data.dtype}), sizes={self.sizes})"


class DataAccessor(abc.ABC):
    """Abstract class for data accessors.

    A data accessor is a way to access a torch.Tensor given an index.
    """

    @abc.abstractmethod
    def __getitem__(self, i: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        pass

    @abc.abstractmethod
    def batched(self, indices: Iterator[int]) -> Batch:
        """Get the padded data and the original sizes of the data from a list of indices."""


class InMemoryAccessor(DataAccessor):
    """Data accessor where everything is in memory."""

    def __init__(self, indices: dict[int, tuple[int, int]], data: torch.Tensor) -> None:
        self.device = Environment().device
        self.indices = indices
        verify_empty_datapoints(self.indices)
        self.data = data.to(self.device)

    def __repr__(self) -> str:
        return f"InMemoryAccessor(data of shape {tuple(self.data.shape)}, with {len(self)} items)"

    def __getitem__(self, i: int) -> torch.Tensor:
        if i not in self.indices:
            raise IndexError
        start, end = self.indices[i]
        return self.data[start:end]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in self.indices:
            yield self[i]

    def batched(self, indices: Iterator[int]) -> Batch:
        """Get the padded data and the original sizes of the data from a list of indices."""
        sizes, data = [], []
        for i in indices:
            this_data = self[i]
            sizes.append(this_data.size(0))
            data.append(this_data)
        return Batch(pad_sequence(data, batch_first=True), torch.tensor(sizes, dtype=torch.int64, device=self.device))


def find_all_files(root: str | Path, extension: str) -> dict[str, Path]:
    """Recursively find all files with the given `extension` in `root`."""
    return dict(sorted((p.stem, p) for p in Path(root).rglob(f"*{extension}")))


def normalize_with_singularity(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize the given vector across the third dimension.

    Extend all vectors by eps to put the null vector at the maximal
    angular distance from any non-null vector.
    """
    norm = torch.norm(x, dim=1, keepdim=True)
    zero_vals = norm == 0
    x = torch.where(zero_vals, 1 / math.sqrt(x.size(1)), x / norm)
    border = torch.full((x.size(0), 1), eps, dtype=x.dtype, device=x.device)
    border = torch.where(zero_vals, -2 * eps, border)
    return torch.cat([x, border], dim=1)


class InvalidItemFileError(Exception):
    """The item file does not have the correct columns."""


def read_item(item: str | Path) -> pl.DataFrame:
    """Read an item file."""
    schema = {
        "#file": pl.String,
        "onset": pl.String,
        "offset": pl.String,
        "#phone": pl.String,
        "prev-phone": pl.String,
        "next-phone": pl.String,
        "speaker": pl.String,
    }
    try:
        return pl.read_csv(item, separator=" ", schema=schema).with_columns(
            pl.col("onset").str.to_decimal(), pl.col("offset").str.to_decimal()
        )
    except pl.exceptions.ComputeError as error:
        raise InvalidItemFileError from error


def item_frontiers(frequency: float) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Frontiers [start, end[ in the input features and in the concatenated ones."""
    # Remove the cast to float and -0.5 once RoundMode "half_ceil" and "half_floor" are added to polars round method
    # (or "half_away_from_zero" and "half_to_zero")
    # See: https://github.com/pola-rs/polars/issues/21800
    start = (pl.col("onset") * frequency - 0.5).cast(pl.Float64).ceil().cast(pl.Int64).alias("start")
    end = (pl.col("offset") * frequency - 0.5).cast(pl.Float64).floor().cast(pl.Int64).alias("end")
    if not with_librilight_bug():
        end += 1
    length = (end - start).alias("length")
    right = length.cum_sum().alias("right")
    left = length.cum_sum().shift(1).fill_null(0).alias("left")
    return start, end, left, right


def load_data_from_item(
    paths: dict[str, Path],
    labels: pl.DataFrame,
    frequency: int,
    feature_maker: FeatureMaker,
) -> tuple[dict[int, tuple[int, int]], torch.Tensor]:
    """Load all data in memory. Return a dictionary of indices and a tensor of data."""
    metadata = labels[["#file", "onset", "offset"]].with_row_index()
    lazy = metadata.lazy().sort("#file", maintain_order=True).with_columns(*item_frontiers(frequency))
    indices_lazy = lazy.select("left", "right", "index").sort("index").select("left", "right")
    by_file_lazy = lazy.select("#file", "start", "end").group_by("#file", maintain_order=True).agg("start", "end")
    indices, by_file = pl.collect_all([indices_lazy, by_file_lazy])

    data, device = [], Environment().device
    for fileid, start_indices, end_indices in tqdm(by_file.iter_rows(), desc="Building dataset", total=len(by_file)):
        features = feature_maker(paths[fileid]).detach().to(device)
        data += [features[start:end] for start, end in zip(start_indices, end_indices, strict=True)]
    return dict(enumerate(indices.rows())), torch.cat(data, dim=0)


class TimesArrayDimensionError(ValueError):
    """To raise if the times array is not 1D."""

    def __init__(self) -> None:
        super().__init__("Only 1D times array are supported")


def load_data_from_item_with_times(
    paths_features: dict[str, Path],
    paths_times: dict[str, Path],
    labels: pl.DataFrame,
) -> tuple[dict[int, tuple[int, int]], torch.Tensor]:
    """Load all data in memory using features and times array. This is smaller than using a predefined frequency."""
    metadata = labels[["#file", "onset", "offset"]].with_row_index()
    by_file = (
        metadata.sort("#file", maintain_order=True)
        .group_by("#file", maintain_order=True)
        .agg("index", "onset", "offset")
    )
    data, device, all_indices, right = [], Environment().device, {}, 0
    decimals = by_file["onset"].dtype.inner.scale
    for fileid, indices, onsets, offsets in tqdm(by_file.iter_rows(), desc="Building dataset", total=len(by_file)):
        features = torch.load(paths_features[fileid], map_location=device).detach()
        times = torch.load(paths_times[fileid]).round(decimals=decimals)
        if times.ndim > 1:
            raise TimesArrayDimensionError
        for index, onset, offset in zip(indices, onsets, offsets, strict=True):
            mask = torch.where(torch.logical_and(float(onset) <= times, times <= float(offset)))[0]
            data.append(features[mask])
            left = right
            right += len(mask)
            all_indices[index] = (left, right)
    return all_indices, torch.cat(data, dim=0)


@dataclass(frozen=True)
class Dataset:
    """Simple interface to a dataset.

    :param labels: ``pl.DataFrame`` containing the labels of the datapoints.
    :param accessor: ``InMemoryAccessor`` to access the data.
    """

    labels: pl.DataFrame
    accessor: InMemoryAccessor

    def __repr__(self) -> str:
        return f"labels:\n{self.labels!r}\naccessor: {self.accessor!r}"

    def normalize_(self) -> Self:
        """L2 normalization of the data."""
        self.accessor.data = normalize_with_singularity(self.accessor.data.cpu()).to(self.accessor.device)
        return self

    @classmethod
    def from_item(
        cls,
        item: str | Path,
        root: str | Path,
        frequency: int,
        *,
        feature_maker: FeatureMaker = torch.load,
        extension: str = ".pt",
    ) -> "Dataset":
        """Create a dataset from an item file."""
        labels = read_item(item)
        paths = find_all_files(root, extension)
        indices, data = load_data_from_item(paths, labels, frequency, feature_maker)
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_item_with_times(cls, item: str | Path, features: str | Path, times: str | Path) -> "Dataset":
        """Create a dataset from an item file.

        Use arrays containing the times associated to the features instead of a given frequency.
        """
        labels = read_item(item)
        paths_features = find_all_files(features, ".pt")
        paths_times = find_all_files(times, ".pt")
        indices, data = load_data_from_item_with_times(paths_features, paths_times, labels)
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_dataframe(cls, df: SupportsInterchange, feature_columns: str | Collection[str]) -> "Dataset":
        """Create a dataset from a DataFrame (polars or pandas)."""
        df = pl.from_dataframe(df.__dataframe__())
        labels = df.select(cs.exclude(feature_columns))
        indices = {i: (i, i + 1) for i in range(len(labels))}
        data = df.select(feature_columns).cast(pl.Float32).to_torch()
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_csv(cls, path: str | Path, feature_columns: str | Collection[str], *, separator: str = ",") -> "Dataset":
        """Create a dataset from a CSV file."""
        return cls.from_dataframe(pl.read_csv(path, separator=separator), feature_columns)

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[object]], feature_columns: str | Collection[str]) -> "Dataset":
        """Create a dataset from a dictionary of sequences."""
        return cls.from_dataframe(pl.from_dict(data), feature_columns)

    @classmethod
    def from_dicts(cls, data: Iterable[dict[str, Any]], feature_columns: str | Collection[str]) -> "Dataset":
        """Create a dataset from a sequence of dictionaries."""
        return cls.from_dataframe(pl.from_dicts(data), feature_columns)

    @classmethod
    def from_numpy(cls, features: ArrayLike, labels: Mapping[str, Sequence[object]]) -> "Dataset":
        """Create a dataset from the features (numpy array) and the labels (dictionary of sequences)."""
        features_df = pl.from_numpy(np.asarray(features))
        labels_df = pl.from_dict(labels)
        if len(features_df) != len(labels_df):
            raise ValueError
        return cls.from_dataframe(pl.concat((features_df, labels_df), how="horizontal"), features_df.columns)


def dummy_dataset_from_item(item: str | Path, frequency: int | None) -> Dataset:
    """To debug."""
    labels = read_item(item).with_columns(pl.lit(0).alias("dummy"))
    if frequency is not None:
        labels = labels.with_columns(*item_frontiers(frequency))
    return Dataset.from_dataframe(labels, "dummy")

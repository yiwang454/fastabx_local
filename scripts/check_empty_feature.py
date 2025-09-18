import torch
# from fastabx import zerospeech_abx
from pathlib import Path

import abc
import math
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs

from polars.interchange.protocol import SupportsInterchange
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

type ArrayLike = npt.ArrayLike

from fastabx.dataset import item_frontiers, read_item, find_all_files
layer = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def post_delete_item():
#     [170498, 264684, 397790, 505072]

def maker(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True)

def load_data_from_item_check[T](
    mapping: dict[str, T],
    labels: pl.DataFrame,
    frequency: int,
    feature_maker: Callable[[T], torch.Tensor],
) -> tuple[dict[int, tuple[int, int]], torch.Tensor]:
    """Load all data in memory. Return a dictionary of indices and a tensor of data."""
    metadata = labels[["#file", "onset", "offset"]].with_row_index()
    lazy = metadata.lazy().sort("#file", maintain_order=True).with_columns(*item_frontiers(frequency))
    indices_lazy = lazy.select("left", "right", "index").sort("index").select("left", "right")
    by_file_lazy = lazy.select("#file", "start", "end").group_by("#file", maintain_order=True).agg("start", "end")
    indices, by_file = pl.collect_all([indices_lazy, by_file_lazy])

    data, device = [], torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fileid, start_indices, end_indices in tqdm(by_file.iter_rows(), desc="Building dataset", total=len(by_file)):
        features = feature_maker(mapping[fileid]).detach().to(device)
        print(features.size())
        # print(f"empty feature {fileid}")
    return
    #  dict(enumerate(indices.rows())), torch.cat(data, dim=0)

def from_item(
    item: str | Path,
    root: str | Path,
    frequency: int,
    *,
    feature_maker: Callable[[str | Path], torch.Tensor] = torch.load,
    extension: str = ".pt",
    ) -> "Dataset":
    """Create a dataset from an item file.

    If you want to keep the Libri-Light bug to reproduce previous results,
    set the environment variable FASTABX_WITH_LIBRILIGHT_BUG=1.
    """
    labels = read_item(item)
    paths = find_all_files(root, extension)
    load_data_from_item_check(paths, labels, frequency, feature_maker)
    return
    # Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

if __name__ == "__main__":
    item, frequency = "/mnt/ceph_rbd/muavic/scripts/vctk_tenth_item.item", 50
    features = f"/mnt/ceph_rbd/data/vctk/hubert_feature/large_l{layer}_mic1"
    from_item(item, features, frequency)
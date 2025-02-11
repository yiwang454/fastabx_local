"""ZeroSpeech ABX evaluation. Reproduces ZeroSpeech 2021."""

from pathlib import Path
from typing import Literal

import torch

from fastabx.dataset import Dataset
from fastabx.score import Score
from fastabx.subsample import Subsampler
from fastabx.task import Task

__all__ = ["zerospeech_abx"]


def zerospeech_abx(  # noqa: PLR0913
    item: str | Path,
    features: str | Path,
    speaker: Literal["within", "across"],
    context: Literal["within", "any"] = "within",
    distance: Literal["cosine", "euclidean", "kl_symmetric", "identical"] = "cosine",
    frequency: int = 50,
    max_size_group: int = 10,
    max_x_across: int = 5,
    seed: int = 0,
) -> float:
    """Compute the ABX similarly to the ZeroSpeech 2021 challenge.

    On triphone or phoneme, described by an item file.
    Within or across speaker, and within context or ignoring context.
    """
    dataset = Dataset.from_item(item, features, frequency, torch.load)
    by: list[str] | None
    across: list[str] | None
    match (speaker, context):
        case ("within", "within"):
            by, across = ["prev-phone", "next-phone", "speaker"], None
        case ("within", "any"):
            by, across = ["speaker"], None
        case ("across", "within"):
            by, across = ["prev-phone", "next-phone"], ["speaker"]
        case ("across", "any"):
            by, across = None, ["speaker"]
        case _:
            raise ValueError("invalid speaker or context mode")
    subsampler = Subsampler(max_size_group, max_x_across, seed)
    task = Task(dataset, on="#phone", by=by, across=across, subsampler=subsampler)
    return Score(task, distance).collapse(levels=[("next-phone", "prev-phone"), "speaker"])

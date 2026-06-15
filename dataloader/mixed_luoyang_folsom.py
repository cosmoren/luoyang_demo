"""Mixed Luoyang + Folsom dataloader utilities.

This module adds a lightweight dataset wrapper that samples from two existing
datasets by configurable probabilities, plus a collate function compatible with
their shared sample schema.

Usage example:

    from torch.utils.data import DataLoader
    from dataloader.mixed_luoyang_folsom import (
        MixedLuoyangFolsomDataset,
        collate_mixed_luoyang_folsom,
    )

    mixed_ds = MixedLuoyangFolsomDataset(
        luoyang_dataset=luoyang_train_ds,
        folsom_dataset=folsom_train_ds,
        probs=(0.7, 0.3),   # 70% Luoyang, 30% Folsom
        epoch_len=100000,
    )
    loader = DataLoader(
        mixed_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_mixed_luoyang_folsom,
        num_workers=8,
    )
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


_CORE_TENSOR_KEYS: tuple[str, ...] = (
    "dev_idx",
    "pv",
    "pv_mask",
    "pv_timefeats",
    "forecast_timefeats",
    "kt",
    "kt_mask",
    "p_cs",
    "p_mean",
    "target_pv",
    "target_mask",
    "target_p_cs",
)

_OPTIONAL_TENSOR_KEYS: tuple[str, ...] = (
    "nwp_tensor",
    "sat_tensor",
    "sat_timefeats",
    "skimg_tensor",
    "skimg_timefeats",
    "target_weather_score",
)


class MixedLuoyangFolsomDataset(Dataset):
    """Sample from Luoyang and Folsom datasets by given probabilities.

    Notes:
    - This wrapper does not modify child datasets.
    - Returned sample keeps original tensor keys and adds:
      - ``source_dataset``: ``"luoyang"`` or ``"folsom"``
      - ``source_index``: sampled index in the chosen child dataset
    """

    def __init__(
        self,
        *,
        luoyang_dataset: Dataset,
        folsom_dataset: Dataset,
        probs: Sequence[float] = (0.5, 0.5),
        epoch_len: int | None = None,
        sample_with_replacement: bool = True,
        deterministic_by_index: bool = False,
        seed: int = 0,
    ) -> None:
        if len(probs) != 2:
            raise ValueError(f"probs must have length 2, got {len(probs)}")
        p_l, p_f = float(probs[0]), float(probs[1])
        if p_l < 0 or p_f < 0:
            raise ValueError(f"probs must be non-negative, got {probs}")
        s = p_l + p_f
        if s <= 0:
            raise ValueError("probs sum must be > 0")
        self._p_luoyang = p_l / s
        self._p_folsom = p_f / s

        self.luoyang_dataset = luoyang_dataset
        self.folsom_dataset = folsom_dataset
        self._len_luoyang = int(len(luoyang_dataset))
        self._len_folsom = int(len(folsom_dataset))
        if self._len_luoyang < 1 or self._len_folsom < 1:
            raise ValueError(
                f"both child datasets must be non-empty, got "
                f"len(luoyang)={self._len_luoyang}, len(folsom)={self._len_folsom}"
            )

        if epoch_len is None:
            self._epoch_len = self._len_luoyang + self._len_folsom
        else:
            self._epoch_len = max(1, int(epoch_len))

        self._sample_with_replacement = bool(sample_with_replacement)
        self._deterministic_by_index = bool(deterministic_by_index)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def __len__(self) -> int:
        return self._epoch_len

    def _pick_source(self, idx: int) -> str:
        if self._deterministic_by_index:
            r = np.random.default_rng(self._seed + int(idx)).random()
        else:
            r = self._rng.random()
        return "luoyang" if r < self._p_luoyang else "folsom"

    def _pick_child_index(self, source: str, idx: int) -> int:
        n = self._len_luoyang if source == "luoyang" else self._len_folsom
        if self._sample_with_replacement:
            if self._deterministic_by_index:
                return int(np.random.default_rng(self._seed * 9973 + int(idx)).integers(0, n))
            return int(self._rng.integers(0, n))
        return int(idx % n)

    @staticmethod
    def _normalize_sample(sample: dict[str, Any], source: str, source_index: int) -> dict[str, Any]:
        out = dict(sample)

        # Keep mixed batches collate-safe when a dataset omits this key.
        if "target_weather_score" not in out:
            tpv = out.get("target_pv")
            if torch.is_tensor(tpv):
                out["target_weather_score"] = torch.zeros_like(tpv)
            else:
                out["target_weather_score"] = None

        for key in _OPTIONAL_TENSOR_KEYS:
            out.setdefault(key, None)

        out["source_dataset"] = source
        out["source_index"] = int(source_index)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source = self._pick_source(idx)
        child_idx = self._pick_child_index(source, idx)
        if source == "luoyang":
            sample = self.luoyang_dataset[child_idx]
        else:
            sample = self.folsom_dataset[child_idx]
        return self._normalize_sample(sample=sample, source=source, source_index=child_idx)


def _stack_required(batch: list[dict[str, Any]], key: str) -> torch.Tensor:
    vals = [s[key] for s in batch]
    if any(not torch.is_tensor(v) for v in vals):
        kinds = [type(v).__name__ for v in vals]
        raise TypeError(f"key {key!r} must be tensor for all samples, got {kinds}")
    return torch.stack(vals)


def _stack_optional(batch: list[dict[str, Any]], key: str) -> torch.Tensor | None:
    vals = [s.get(key) for s in batch]
    non_none = [v for v in vals if v is not None]
    if not non_none:
        return None
    if any(not torch.is_tensor(v) for v in non_none):
        kinds = [type(v).__name__ for v in vals]
        raise TypeError(f"key {key!r} must be tensor or None, got {kinds}")

    template = non_none[0]
    assert torch.is_tensor(template)
    filled: list[torch.Tensor] = []
    for v in vals:
        if v is None:
            filled.append(torch.zeros_like(template))
        else:
            if not torch.is_tensor(v):
                raise TypeError(f"key {key!r} has non-tensor value {type(v).__name__}")
            if tuple(v.shape) != tuple(template.shape):
                raise ValueError(
                    f"key {key!r} shape mismatch in mixed batch: "
                    f"expected {tuple(template.shape)}, got {tuple(v.shape)}"
                )
            filled.append(v)
    return torch.stack(filled)


def collate_mixed_luoyang_folsom(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate for :class:`MixedLuoyangFolsomDataset`.

    Behavior:
    - Core training tensors are stacked directly.
    - Optional tensors accept mixed ``None/tensor``; ``None`` entries are zero-filled to
      keep batch shape valid.
    - Adds metadata:
      - ``source_dataset``: list[str]
      - ``source_index``: tensor[int64] shape [B]
    """
    if not batch:
        raise ValueError("empty batch")

    out: dict[str, Any] = {}
    for key in _CORE_TENSOR_KEYS:
        out[key] = _stack_required(batch, key)

    for key in _OPTIONAL_TENSOR_KEYS:
        out[key] = _stack_optional(batch, key)

    out["source_dataset"] = [str(s.get("source_dataset", "")) for s in batch]
    out["source_index"] = torch.tensor(
        [int(s.get("source_index", -1)) for s in batch], dtype=torch.long
    )
    return out


def build_mixed_luoyang_folsom_dataset(
    *,
    luoyang_dataset: Dataset,
    folsom_dataset: Dataset,
    probs: Sequence[float] = (0.5, 0.5),
    epoch_len: int | None = None,
    sample_with_replacement: bool = True,
    deterministic_by_index: bool = False,
    seed: int = 0,
) -> MixedLuoyangFolsomDataset:
    """Convenience factory for mixed dataset creation."""
    return MixedLuoyangFolsomDataset(
        luoyang_dataset=luoyang_dataset,
        folsom_dataset=folsom_dataset,
        probs=probs,
        epoch_len=epoch_len,
        sample_with_replacement=sample_with_replacement,
        deterministic_by_index=deterministic_by_index,
        seed=seed,
    )


__all__ = [
    "MixedLuoyangFolsomDataset",
    "collate_mixed_luoyang_folsom",
    "build_mixed_luoyang_folsom_dataset",
]

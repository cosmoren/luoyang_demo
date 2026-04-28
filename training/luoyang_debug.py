from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataloader.luoyang import INVERTER_STATE_COL, VALID_STATE, load_csv


def normalize_debug_split(split: str) -> str:
    out = str(split).strip().lower()
    if out not in {"train", "test"}:
        raise ValueError(f"debug split must be 'train' or 'test', got {out!r}")
    return out


def split_anchor_mask(dataset: Any, debug_split: str) -> np.ndarray:
    if debug_split == "train":
        return dataset._train_anchor_mask
    return dataset._test_anchor_mask


def valid_anchor_rows(dataset: Any, df: pd.DataFrame, debug_split: str) -> np.ndarray:
    split_mask = split_anchor_mask(dataset, debug_split)
    inv = pd.to_numeric(df[INVERTER_STATE_COL], errors="coerce").fillna(0).astype(int).values == VALID_STATE
    valid_mask = inv[dataset._y_idx_per_anchor].any(axis=1) & split_mask
    rows = np.nonzero(valid_mask)[0]
    if rows.size == 0:
        raise ValueError(f"no valid anchors found for split={debug_split!r}")
    return rows


def resolve_sample_path(sample_files: list[Path], requested_csv_path: str, debug_split: str) -> Path:
    if not str(requested_csv_path).strip():
        raise ValueError("debug csv path must be set when random selection is disabled")
    want = Path(requested_csv_path).expanduser()
    resolved_files = {p.resolve(): p for p in sample_files}
    key = want.resolve()
    if key in resolved_files:
        return resolved_files[key]

    names = {p.name: p for p in sample_files}
    if want.name in names:
        return names[want.name]

    raise ValueError(
        f"debug csv path {want!r} is not in {debug_split}_dataset.sample_files "
        "(use a full path or exact CSV basename from pv_dir)"
    )


def resolve_anchor_last_row_for_collect_time(
    *,
    df: pd.DataFrame,
    desired_collect_time: str,
    valid_rows: np.ndarray,
    x_idx_per_anchor: np.ndarray,
    debug_split: str,
    sample_name: str,
) -> int:
    ct = pd.to_datetime(df["collectTime"], errors="coerce")
    desired = pd.Timestamp(desired_collect_time)
    matches = np.flatnonzero((ct == desired).to_numpy())
    if matches.size != 1:
        raise ValueError(
            f"expected exactly one row with collectTime={desired!r}, found {matches.size} in {sample_name}"
        )

    j = int(matches[0])
    last_x_for_valid = x_idx_per_anchor[valid_rows, -1]
    valid_last_set = set(last_x_for_valid.tolist())
    if j in valid_last_set:
        return j

    order = np.argsort(np.abs(last_x_for_valid.astype(np.int64) - j))
    hint_lines = []
    for k in order[:10]:
        jj = int(last_x_for_valid[int(k)])
        hint_lines.append(f"  row={jj}  collectTime={ct.iloc[jj]}")
    j_best = int(last_x_for_valid[int(order[0])])
    hint = "\n".join(hint_lines)
    raise ValueError(
        f"collectTime row index j={j} is not the last PV input row of any valid {debug_split} anchor "
        f"for {sample_name} ({debug_split} split + inverter mask). "
        "This split only uses windows whose last input row is one of these anchor end rows.\n"
        "Nearest valid last-input rows (copy a collectTime into --debug-last-input-collect-time):\n"
        f"{hint}\n"
        f"Closest valid row index: {j_best} (delta {abs(j_best - j)} rows from your j={j})."
    )


def choose_random_debug_sample(dataset: Any, debug_split: str) -> tuple[Path, pd.DataFrame, int]:
    rng = np.random.default_rng()
    fi = int(rng.integers(0, len(dataset.sample_files)))
    sample_path = dataset.sample_files[fi]
    df = load_csv(sample_path)
    rows = valid_anchor_rows(dataset, df, debug_split)
    r = int(rng.choice(rows))
    return sample_path, df, r


def choose_deterministic_debug_sample(
    dataset: Any,
    *,
    debug_split: str,
    csv_path: str,
    last_input_collect_time: str,
) -> tuple[Path, pd.DataFrame, int]:
    sample_path = resolve_sample_path(dataset.sample_files, csv_path, debug_split)
    df = load_csv(sample_path)
    rows = valid_anchor_rows(dataset, df, debug_split)
    j = resolve_anchor_last_row_for_collect_time(
        df=df,
        desired_collect_time=last_input_collect_time,
        valid_rows=rows,
        x_idx_per_anchor=dataset._x_idx_per_anchor,
        debug_split=debug_split,
        sample_name=sample_path.name,
    )
    return sample_path, df, j

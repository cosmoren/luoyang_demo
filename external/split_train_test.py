#!/usr/bin/env python3
"""
Split single-year PV CSVs by calendar month on ``collectTime``.

For input directory ``ori`` (``*.csv``), creates sibling folders::

    {parent}/{ori}_train/   # rows with month in 1 .. m (inclusive)
    {parent}/{ori}_test/    # rows with month in m+1 .. 12

Each output folder has the same filenames as ``ori``; schema is unchanged (only row filter).
Each input file must contain timestamps from a single calendar year.

Example: ``-m 9`` → training through September (inclusive), test from October onward
(equivalent to ``collectTime <=`` last September row in that year vs October–December rows).

Usage::

    python external/split_train_test.py /path/to/ori -m 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _assert_single_year(series: pd.Series, csv_path: Path) -> int:
    years = series.dt.year.dropna().unique()
    if len(years) != 1:
        raise ValueError(
            f"{csv_path}: expected one calendar year in {series.name!r}, "
            f"found {sorted(years.tolist())!r}"
        )
    return int(years[0])


def split_one_csv(
    csv_path: Path,
    out_train: Path,
    out_test: Path,
    *,
    last_train_month: int,
    time_col: str = "collectTime",
) -> tuple[int, int]:
    """Write train/test CSVs; returns (n_train, n_test)."""
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"{csv_path}: missing column {time_col!r}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        n_bad = int(df[time_col].isna().sum())
        raise ValueError(f"{csv_path}: {n_bad} invalid {time_col} value(s)")

    _assert_single_year(df[time_col], csv_path)

    months = df[time_col].dt.month
    train_mask = months <= last_train_month
    test_mask = months > last_train_month

    train_df = df.loc[train_mask].sort_values(time_col)
    test_df = df.loc[test_mask].sort_values(time_col)

    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_train / csv_path.name, index=False)
    test_df.to_csv(out_test / csv_path.name, index=False)

    return len(train_df), len(test_df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split each single-year CSV: months 1..m → {dir}_train, months m+1..12 → {dir}_test."
        )
    )
    parser.add_argument(
        "csv_dir",
        type=Path,
        help="Directory of input *.csv files (e.g. ori).",
    )
    parser.add_argument(
        "-m",
        "--month",
        type=int,
        required=True,
        choices=range(1, 13),
        metavar="M",
        help="Last month in training (1=Jan … 12=Dec). Test starts at month M+1.",
    )
    parser.add_argument(
        "--time-col",
        default="collectTime",
        help="Timestamp column (default: collectTime).",
    )
    args = parser.parse_args(argv)

    src = args.csv_dir.resolve()
    if not src.is_dir():
        print(f"Not a directory: {src}", file=sys.stderr)
        return 1

    parent, name = src.parent, src.name
    dir_train = parent / f"{name}_train"
    dir_test = parent / f"{name}_test"

    paths = sorted(src.glob("*.csv"))
    if not paths:
        print(f"No *.csv in {src}", file=sys.stderr)
        return 1

    m = args.month
    print(f"Input:     {src}")
    print(f"Train out: {dir_train}  (months 1–{m}, inclusive)")
    print(f"Test out:  {dir_test}   (months {m + 1}–12)")
    if m == 12:
        print("Note: m=12 → all rows in train; test CSVs will be empty (headers only).")
    print(f"Files:     {len(paths)}")

    total_tr = total_te = 0
    for p in paths:
        try:
            n_tr, n_te = split_one_csv(
                p,
                dir_train,
                dir_test,
                last_train_month=m,
                time_col=args.time_col,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        total_tr += n_tr
        total_te += n_te
        print(f"  {p.name}: train_rows={n_tr} test_rows={n_te}")

    print(f"Done. Total train_rows={total_tr} test_rows={total_te}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

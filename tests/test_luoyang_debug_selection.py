import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from training.luoyang_debug import (
    normalize_debug_split,
    resolve_anchor_last_row_for_collect_time,
    resolve_sample_path,
)


class TestLuoyangDebugSelection(unittest.TestCase):
    def test_normalize_debug_split(self) -> None:
        self.assertEqual(normalize_debug_split("train"), "train")
        self.assertEqual(normalize_debug_split(" TEST "), "test")
        with self.assertRaisesRegex(ValueError, "debug split"):
            normalize_debug_split("dev")

    def test_resolve_sample_path_by_basename(self) -> None:
        files = [Path("/tmp/a.csv"), Path("/tmp/b.csv")]
        got = resolve_sample_path(files, "b.csv", "train")
        self.assertEqual(got, Path("/tmp/b.csv"))

    def test_resolve_anchor_last_row_hints_when_not_valid_anchor_end(self) -> None:
        df = pd.DataFrame(
            {
                "collectTime": pd.to_datetime(
                    [
                        "2025-01-01 00:00:00",
                        "2025-01-01 00:05:00",
                        "2025-01-01 00:10:00",
                        "2025-01-01 00:15:00",
                    ]
                )
            }
        )
        x_idx_per_anchor = np.array(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=np.int64,
        )
        valid_rows = np.array([0, 1], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "Nearest valid last-input rows"):
            resolve_anchor_last_row_for_collect_time(
                df=df,
                desired_collect_time="2025-01-01 00:15:00",
                valid_rows=valid_rows,
                x_idx_per_anchor=x_idx_per_anchor,
                debug_split="train",
                sample_name="x.csv",
            )

    def test_resolve_anchor_last_row_success(self) -> None:
        df = pd.DataFrame(
            {
                "collectTime": pd.to_datetime(
                    [
                        "2025-01-01 00:00:00",
                        "2025-01-01 00:05:00",
                        "2025-01-01 00:10:00",
                    ]
                )
            }
        )
        x_idx_per_anchor = np.array([[0, 1], [1, 2]], dtype=np.int64)
        valid_rows = np.array([0, 1], dtype=np.int64)
        got = resolve_anchor_last_row_for_collect_time(
            df=df,
            desired_collect_time="2025-01-01 00:10:00",
            valid_rows=valid_rows,
            x_idx_per_anchor=x_idx_per_anchor,
            debug_split="test",
            sample_name="ok.csv",
        )
        self.assertEqual(got, 2)


if __name__ == "__main__":
    unittest.main()

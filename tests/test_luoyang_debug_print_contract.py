import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from dataloader.luoyang import INVERTER_STATE_COL, data_fetched, what_to_fetch


class _FakeDebugDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._x_idx_per_anchor = np.array([[0, 1, 2]], dtype=np.int64)
        self._y_idx_per_anchor = np.array([[3, 4]], dtype=np.int64)
        self._x_tail_1d = np.array([-2, -1, 0], dtype=np.int64)
        self._y_off_1d = np.array([1, 2], dtype=np.int64)
        self._satimg_npy_shape_hwc = (2, 2, 1)
        self.pv_output_interval_min = 5
        self.pv_output_len = 2
        self.nwp_solar_df = None
        self.nwp_wind_df = None

    def _to_utc_timestamps(self, timestamps, naive_tz: str):
        out = []
        for t in timestamps:
            ts = pd.Timestamp(t)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            out.append(ts)
        return out

    def _history_sky_frame_times(self, t_x_end):
        t = pd.Timestamp(t_x_end)
        return [t - pd.Timedelta(minutes=5), t]

    def _history_satimg_frame_utc_times(self, t_x_end):
        t = pd.Timestamp(t_x_end)
        return [t - pd.Timedelta(minutes=10), t]

    def _sky_jpg_path(self, t: pd.Timestamp) -> Path:
        return self.root / "sky" / f"{t.strftime('%Y%m%d_%H%M')}.jpg"

    def _satimg_npy_path(self, t: pd.Timestamp) -> Path:
        return self.root / "sat" / f"{t.strftime('%Y%m%d_%H%M')}.npy"


class TestLuoyangDebugPrintContract(unittest.TestCase):
    def test_print_sections_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "sky").mkdir(parents=True)
            (root / "sat").mkdir(parents=True)
            sample_path = root / "sample.csv"
            sample_path.write_text("dummy", encoding="utf-8")

            ds = _FakeDebugDataset(root)
            df = pd.DataFrame(
                {
                    "collectTime": pd.to_datetime(
                        [
                            "2025-01-01 00:00:00",
                            "2025-01-01 00:05:00",
                            "2025-01-01 00:10:00",
                            "2025-01-01 00:15:00",
                            "2025-01-01 00:20:00",
                        ]
                    ),
                    "active_power": [100, 101, 102, 103, 104],
                    INVERTER_STATE_COL: [1, 1, 1, 1, 1],
                }
            )

            sat_ok = ds._satimg_npy_path(pd.Timestamp("2025-01-01 00:00:00"))
            np.save(sat_ok, np.zeros(ds._satimg_npy_shape_hwc, dtype=np.float32), allow_pickle=False)

            buf = io.StringIO()
            with redirect_stdout(buf):
                what_to_fetch(ds, df, 0)
                data_fetched(ds, sample_path, df, 0)
            out = buf.getvalue()

            self.assertIn("what to fetch:", out)
            self.assertIn("PV_input:", out)
            self.assertIn("PV_output:", out)
            self.assertIn("Sky:", out)
            self.assertIn("Satellite:", out)
            self.assertIn("NWP:", out)
            self.assertIn("data fetched:", out)
            self.assertIn("Output: file=", out)


if __name__ == "__main__":
    unittest.main()

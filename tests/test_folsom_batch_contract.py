import os
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloader.folsom import FOLSOM_BATCH_TENSOR_KEYS, build_folsom_irradiance_datasets_from_conf, collate_folsom_irradiance
from training.training_conf import FOLSOM_CONF_PATH, load_config_path


class TestFolsomBatchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        conf_path = Path(os.environ.get("FOLSOM_TEST_CONF", str(FOLSOM_CONF_PATH)))
        if not conf_path.is_file():
            raise unittest.SkipTest(f"Config not found: {conf_path}")
        try:
            cls.conf = load_config_path(conf_path)
        except Exception as e:  # pragma: no cover - defensive skip for local env issues
            raise unittest.SkipTest(f"Cannot load config {conf_path}: {e}") from e

    def test_first_batch_contract(self) -> None:
        try:
            train_ds, _ = build_folsom_irradiance_datasets_from_conf(self.conf, train_epoch_len=2)
        except Exception as e:  # pragma: no cover - defensive skip for missing local data
            raise unittest.SkipTest(f"Dataset unavailable for local test env: {e}") from e

        loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_folsom_irradiance,
            num_workers=0,
        )
        batch = next(iter(loader))

        missing = [k for k in FOLSOM_BATCH_TENSOR_KEYS if k not in batch]
        self.assertFalse(missing, f"Missing required batch keys: {missing}")

        b = int(batch["ghi"].shape[0])
        self.assertGreaterEqual(b, 1)
        self.assertEqual(batch["dni"].shape[0], b)
        self.assertEqual(batch["dhi"].shape[0], b)

        # Shape relationships after collate.
        self.assertEqual(batch["input_mask"].shape[0], b)
        self.assertEqual(batch["input_mask"].shape[1], 1)
        self.assertEqual(batch["ghi"].shape[1], batch["input_mask"].shape[2])
        self.assertEqual(batch["irr_timefeats"].shape[0], b)
        self.assertEqual(batch["irr_timefeats"].shape[1], batch["ghi"].shape[1])
        self.assertEqual(batch["irr_timefeats"].shape[2], 9)
        self.assertEqual(batch["forecast_timefeats"].shape[0], b)
        self.assertEqual(batch["forecast_timefeats"].shape[1], batch["target_ghi"].shape[1])
        self.assertEqual(batch["forecast_timefeats"].shape[2], 9)
        self.assertEqual(batch["target_ghi"].shape, batch["target_dni"].shape)
        self.assertEqual(batch["target_ghi"].shape, batch["target_dhi"].shape)
        self.assertEqual(batch["target_ghi"].shape, batch["target_mask"].shape)

        # Optional modalities are either all tensor or all None by collate contract.
        for key in ("skimg_tensor", "skimg_timefeats", "nwp_tensor"):
            self.assertIn(key, batch)
            if batch[key] is not None:
                self.assertTrue(torch.is_tensor(batch[key]), f"{key} must be Tensor or None")
                self.assertTrue(torch.isfinite(batch[key]).all(), f"{key} has NaN/Inf")

        # Core tensors should be finite.
        for key in (
            "ghi",
            "dni",
            "dhi",
            "input_mask",
            "irr_timefeats",
            "forecast_timefeats",
            "target_ghi",
            "target_dni",
            "target_dhi",
            "target_mask",
        ):
            self.assertTrue(torch.isfinite(batch[key]).all(), f"{key} has NaN/Inf")


if __name__ == "__main__":
    unittest.main()

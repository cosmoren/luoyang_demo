import os
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader.folsom import build_folsom_irradiance_datasets_from_conf, collate_folsom_irradiance
from training.train_vit_folsom import _batch_to_device_folsom, forward_folsom
from training.training_conf import FOLSOM_CONF_PATH, load_config_path


class _DummyFolsomModel(nn.Module):
    """Lightweight forward target for integration wiring checks."""

    def forward(
        self,
        ghi: torch.Tensor,
        dni: torch.Tensor,
        dhi: torch.Tensor,
        input_mask: torch.Tensor,
        irr_timefeats: torch.Tensor,
        forecast_timefeats: torch.Tensor,
        skimg_tensor: torch.Tensor | None = None,
        skimg_timefeats: torch.Tensor | None = None,
        nwp_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = int(ghi.shape[0])
        t_out = int(forecast_timefeats.shape[1])
        assert dni.shape == ghi.shape and dhi.shape == ghi.shape
        assert input_mask.shape[0] == b
        assert irr_timefeats.shape[0] == b
        return torch.zeros((b, t_out, 3), dtype=ghi.dtype, device=ghi.device)


class TestFolsomForwardWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        conf_path = Path(os.environ.get("FOLSOM_TEST_CONF", str(FOLSOM_CONF_PATH)))
        if not conf_path.is_file():
            raise unittest.SkipTest(f"Config not found: {conf_path}")
        try:
            cls.conf = load_config_path(conf_path)
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"Cannot load config {conf_path}: {e}") from e

    def test_forward_wiring_shape(self) -> None:
        try:
            train_ds, _ = build_folsom_irradiance_datasets_from_conf(self.conf, train_epoch_len=2)
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"Dataset unavailable for local test env: {e}") from e

        loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_folsom_irradiance,
            num_workers=0,
        )
        batch = next(iter(loader))
        d = _batch_to_device_folsom(batch, torch.device("cpu"))
        model = _DummyFolsomModel()
        out = forward_folsom(model, d)
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[0], d["ghi"].shape[0])
        self.assertEqual(out.shape[1], d["forecast_timefeats"].shape[1])
        self.assertEqual(out.shape[2], 3)


if __name__ == "__main__":
    unittest.main()

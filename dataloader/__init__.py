"""
Per-dataset code lives in separate modules under this package.

Example (Luoyang PV):

    from dataloader.luoyang import PVDataset

Other datasets can add ``dataloader/foo.py`` and use ``from dataloader.foo import ...``.

Folsom (GHI / DNI / DHI CSV):

    from dataloader.folsom import (
        FolsomIrradianceDataset,
        collate_folsom_irradiance,
        FOLSOM_GHI_DNI_DHI_KEYS,
        FOLSOM_BATCH_TENSOR_KEYS,
    )

Smoke test (optional ``--last-input-time`` for anchor = last input row time):

    python -m dataloader.folsom
    python -m dataloader.folsom --last-input-time "2014-01-04 07:59:00"
"""

__all__ = []

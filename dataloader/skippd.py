"""
PyTorch Dataset for processed SKIPPD data: loads a single Arrow IPC file in ``__init__``.
Compatible with files written by ``skippd_process.save_skippd_dataset_to_arrow_file``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _ensure_hf_datasets_importable() -> None:
    """
    If ``luoyang_demo`` is on ``sys.path``, it merges with HuggingFace's top-level ``datasets``
    package and breaks ``from datasets import ...``. Remove that path entry and unload cached
    ``datasets`` modules so the installed library loads correctly.
    """
    luoyang_demo = Path(__file__).resolve().parents[1]
    removed: list[str] = []
    new_path: list[str] = []
    for p in sys.path:
        if p:
            try:
                if Path(p).resolve() == luoyang_demo:
                    removed.append(p)
                    continue
            except OSError:
                pass
        new_path.append(p)
    if not removed:
        return
    for name in list(sys.modules):
        if name == "datasets" or name.startswith("datasets."):
            del sys.modules[name]
    sys.path[:] = new_path


_ensure_hf_datasets_importable()

import pyarrow.ipc as ipc
# HuggingFace class lives in ``datasets.arrow_dataset``; ``from datasets import Dataset`` can
# resolve to the wrong type when this repo's ``luoyang_demo/datasets/`` merges with the HF package.
from datasets.arrow_dataset import Dataset as HFDataset
from torch.utils.data import Dataset


def _load_skippd_arrow(arrow_path: Path) -> HFDataset:
    with arrow_path.open("rb") as source:
        reader = ipc.open_file(source)
        table = reader.read_all()
    return HFDataset.from_arrow(table)


class SkippdDataset(Dataset):
    """
    Loads a densified SKIPPD table from a ``.arrow`` / ``.ipc`` file; indexes rows by integer.

    Parameters
    ----------
    arrow_path
        Path to the Arrow file produced by preprocessing.
    """

    def __init__(self, arrow_path: str | Path) -> None:
        self.arrow_path = Path(arrow_path)
        if not self.arrow_path.is_file():
            raise FileNotFoundError(f"Arrow file not found: {self.arrow_path.resolve()}")
        self._data = _load_skippd_arrow(self.arrow_path)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._data[idx]


SkippdVitDataset = SkippdDataset


def main() -> int:
    """Smoke-test: load Arrow file and print dataset size and first row keys."""
    default_arrow = "/home/hw1/workspace/SKIPPD/skippd_continuous.arrow"
    parser = argparse.ArgumentParser(description="Test SkippdDataset loading from an Arrow file.")
    parser.add_argument(
        "arrow",
        nargs="?",
        type=Path,
        default=default_arrow,
        help=f"path to .arrow file (default: {default_arrow})",
    )
    args = parser.parse_args()
    path = args.arrow
    if not path.is_file():
        print(f"error: file not found: {path.resolve()}", file=sys.stderr)
        print("Run skippd_process.py first to build an .arrow file, or pass a path.", file=sys.stderr)
        return 1
    ds = SkippdDataset(path)
    print("arrow_path:", path.resolve())
    print("len:", len(ds))
    row0 = ds[0]
    print("row[0] keys:", sorted(row0.keys()))
    for k, v in row0.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
        else:
            s = repr(v)
            print(f"  {k}: {s[:120]}{'...' if len(s) > 120 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

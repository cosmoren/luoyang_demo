"""
SKIPPD 数据预处理：从 Hugging Face 加载 solarbench/SKIPPD，将 train 与 test 合并后按时间戳排序。
支持按 1 分钟 UTC 网格补齐缺失时段：补行 pv=0，image 为与原始同规格的全黑图。
"""

from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TypeVar


def _ensure_hf_datasets_importable() -> None:
    """Same guard as ``skippd.py`` — see docstring there."""
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

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
from datasets import concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset

T = TypeVar("T")

SKIPPD_REPO = "solarbench/SKIPPD"


def _maybe_tqdm(
    iterable: Iterable[T],
    *,
    enabled: bool,
    total: int | None = None,
    desc: str | None = None,
    unit: str | None = None,
) -> Iterable[T]:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit or "it")
    except ImportError:
        return iterable


def save_skippd_dataset_to_arrow_file(ds: Dataset, path: str | Path) -> Path:
    """
    将 ``Dataset`` 写成单个 Arrow IPC 文件（.arrow），便于含 image 列时整表落盘。

    读取请用 :func:`load_skippd_dataset_from_arrow_file`。
    """
    path = Path(path)
    if path.suffix.lower() not in (".arrow", ".ipc"):
        path = path.with_suffix(".arrow")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = ds.data.table
    with path.open("wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return path


def save_skippd_dataset_to_pickle(
    ds: Dataset,
    path: str | Path,
    *,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> Path:
    """
    Persist a HuggingFace ``Dataset`` with :mod:`pickle` (single ``.pkl`` / ``.pickle`` file).

    .. warning:: Only load pickles you trust (``pickle.load`` can execute arbitrary code).
    """
    path = Path(path)
    if path.suffix.lower() not in (".pkl", ".pickle"):
        path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(ds, f, protocol=protocol)
    return path


def load_skippd_dataset_from_pickle(path: str | Path) -> Dataset:
    """Load a dataset saved by :func:`save_skippd_dataset_to_pickle`."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_skippd_dataset_from_arrow_file(path: str | Path) -> Dataset:
    """从单个 .arrow / .ipc 文件恢复 ``Dataset``（需 ``datasets>=2.14`` 的 ``Dataset.from_arrow``）。"""
    path = Path(path)
    with path.open("rb") as source:
        reader = ipc.open_file(source)
        table = reader.read_all()
    return Dataset.from_arrow(table)


def _black_image_like(reference: Any) -> Any:
    """与 reference 同类型/尺寸的全黑图像（PIL 或 ndarray）。"""
    if reference is None:
        raise ValueError("需要至少一张有效 image 以推断全黑图的尺寸与类型。")
    try:
        from PIL import Image

        if isinstance(reference, Image.Image):
            mode = reference.mode
            size = reference.size
            if mode == "RGB":
                return Image.new("RGB", size, (0, 0, 0))
            if mode == "RGBA":
                return Image.new("RGBA", size, (0, 0, 0, 255))
            return Image.new(mode, size, 0)
    except ImportError:
        pass
    arr = np.asarray(reference)
    dtype = arr.dtype if np.issubdtype(arr.dtype, np.integer) else np.uint8
    return np.zeros(arr.shape, dtype=dtype)


def _time_to_column_value(ts: pd.Timestamp, example: Any) -> Any:
    """与原始列中 time 的存储类型尽量一致。"""
    if isinstance(example, str):
        return ts.isoformat()
    if isinstance(example, datetime):
        return ts.to_pydatetime()
    return ts.isoformat()


def densify_skippd_1min(
    merged: Dataset,
    *,
    show_progress: bool = True,
    save_to_file: str | Path | None = None,
    save_to_pickle: str | Path | None = None,
) -> Dataset:
    """
    将按时间排序后的 SKIPPD 子集扩展为从首条到末条时间之间的完整 1 分钟 UTC 序列。

    - 时间落到整分（与原始行同一分钟的多条记录保留排序中最后一条）。
    - 补行：pv 为 0.0；image 为与样本同规格的全黑图；其它列填 None。

    Parameters
    ----------
    show_progress
        为 True 时尝试用 tqdm 显示扫描源行与生成 1 分钟网格的进度（未安装 tqdm 则静默忽略）。
    save_to_file
        若指定路径，将结果写入**单个** Arrow IPC 文件（默认补 ``.arrow`` 后缀）；
        读取使用 :func:`load_skippd_dataset_from_arrow_file`。
    save_to_pickle
        若指定路径，将最终 ``Dataset`` 用 :func:`save_skippd_dataset_to_pickle` 写入磁盘。
    """
    if len(merged) == 0:
        return merged

    cols = merged.column_names
    if "time" not in cols or "pv" not in cols or "image" not in cols:
        raise ValueError(f"期望列包含 time, pv, image，实际为 {cols}")

    time_example = merged["time"][0]
    ref_image = None
    last_by_ns: dict[int, dict[str, Any]] = {}
    n_src = len(merged)
    for i in _maybe_tqdm(
        range(n_src),
        enabled=show_progress,
        total=n_src,
        desc="densify_skippd_1min: scan rows",
        unit="row",
    ):
        im = merged["image"][i]
        if im is not None and ref_image is None:
            ref_image = im
        raw_t = pd.Timestamp(pd.to_datetime(merged["time"][i], utc=True))
        if raw_t.tz is None:
            raw_t = raw_t.tz_localize("UTC")
        else:
            raw_t = raw_t.tz_convert("UTC")
        ns = int(raw_t.floor("min").value)
        last_by_ns[ns] = {c: merged[c][i] for c in cols}

    if ref_image is None:
        raise ValueError("需要至少一张有效 image 以推断全黑图的尺寸与类型。")
    black = _black_image_like(ref_image)

    t_first = pd.Timestamp(pd.to_datetime(merged["time"][0], utc=True)).tz_convert("UTC").floor("min")
    t_last = pd.Timestamp(pd.to_datetime(merged["time"][-1], utc=True)).tz_convert("UTC").floor("min")
    full_index = pd.date_range(t_first, t_last, freq="1min", tz="UTC")
    n_grid = len(full_index)

    out: dict[str, list[Any]] = {c: [] for c in cols}
    for ts in _maybe_tqdm(
        full_index,
        enabled=show_progress,
        total=n_grid,
        desc="densify_skippd_1min: 1-min grid",
        unit="min",
    ):
        ns = int(ts.value)
        if ns in last_by_ns:
            row = last_by_ns[ns]
            for c in cols:
                if c == "time":
                    out[c].append(_time_to_column_value(ts, time_example))
                else:
                    out[c].append(row[c])
        else:
            for c in cols:
                if c == "time":
                    out[c].append(_time_to_column_value(ts, time_example))
                elif c == "pv":
                    out[c].append(0.0)
                elif c == "image":
                    out[c].append(black)
                else:
                    out[c].append(None)

    ds = Dataset.from_dict(out)
    if save_to_file is not None:
        save_skippd_dataset_to_arrow_file(ds, save_to_file)
    if save_to_pickle is not None:
        save_skippd_dataset_to_pickle(ds, save_to_pickle)
    return ds


def load_skippd_merged_sorted(
    *,
    download_mode: str = "reuse_dataset_if_exists",
) -> Dataset:
    """
    加载 SKIPPD 的 train / test，沿行拼接，再按 UTC 时间戳升序稳定排序。

    Returns
    -------
    datasets.Dataset
        列与原始 SKIPPD 一致（如 time, pv, image 等），行顺序为时间递增。
    """
    ds_train = load_dataset(SKIPPD_REPO, split="train", download_mode=download_mode)
    ds_test = load_dataset(SKIPPD_REPO, split="test", download_mode=download_mode)
    merged = concatenate_datasets([ds_train, ds_test])

    time_idx = pd.DatetimeIndex(pd.to_datetime(merged["time"], utc=True))
    order = np.argsort(time_idx.asi8, kind="stable")
    return merged.select(order.tolist())


def load_skippd_1min_filled(
    *,
    download_mode: str = "reuse_dataset_if_exists",
    show_progress: bool = True,
    save_to_file: str | Path | None = None,
    save_to_pickle: str | Path | None = None,
) -> Dataset:
    """加载、合并、按时间排序后，再按 1 分钟补齐缺口。"""
    return densify_skippd_1min(
        load_skippd_merged_sorted(download_mode=download_mode),
        show_progress=show_progress,
        save_to_file=save_to_file,
        save_to_pickle=save_to_pickle,
    )


if __name__ == "__main__":
    sparse = load_skippd_merged_sorted()
    t_sparse = pd.DatetimeIndex(pd.to_datetime(sparse["time"], utc=True))
    print("sparse num_rows:", len(sparse))
    print("sparse time_range_utc:", t_sparse.min(), "->", t_sparse.max())

    save_dir = Path(__file__).resolve().parent
    save_arrow = save_dir / "skippd_1min_dense.arrow"
    save_pkl = save_dir / "skippd_1min_dense.pkl"
    dense = densify_skippd_1min(
        sparse,
        save_to_file=save_arrow,
        save_to_pickle=save_pkl,
    )
    t_dense = pd.DatetimeIndex(pd.to_datetime(dense["time"], utc=True))
    print("dense num_rows:", len(dense))
    print("dense time_range_utc:", t_dense.min(), "->", t_dense.max())
    print("saved_pickle:", save_pkl.resolve())

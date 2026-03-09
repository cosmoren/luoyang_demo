import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm

# -------------------------
# 输入目录
# -------------------------

sat_root = Path("/home/weize/ai4energy_crop")

# 输出目录
cache_root = Path("/home/weize/sat_cache")

cache_root.mkdir(parents=True, exist_ok=True)

# 使用变量
sat_vars = ["CLTT","CLTH","CLOT","CLTYPE"]


# -------------------------
# 清洗函数
# -------------------------

def clean_frame(ds):

    qa = ds["QA"].astype(np.uint16)

    algo_flag = qa & 0b111

    good = (algo_flag == 4) | (algo_flag == 5)

    good = good.isel(time=0).values.squeeze()

    channels = []

    for v in sat_vars:

        data = ds[v].values.astype(np.float32)

        data = np.squeeze(data)

        valid_min = ds[v].attrs.get("valid_min", None)
        valid_max = ds[v].attrs.get("valid_max", None)

        mask = good.copy()

        if valid_min is not None:
            mask = mask & (data >= valid_min)

        if valid_max is not None:
            mask = mask & (data <= valid_max)

        data = np.where(mask, data, np.nan)

        channels.append(data)

    img = np.stack(channels, axis=0)

    valid_mask = np.all(np.isfinite(img), axis=0).astype(np.float32)

    img = np.nan_to_num(img, nan=0.0)

    img = np.concatenate([img, valid_mask[None]], axis=0)

    return img


# -------------------------
# 遍历 nc 文件
# -------------------------

nc_files = list(sat_root.rglob("*.nc"))

print("Total nc files:", len(nc_files))

for nc in tqdm(nc_files):

    name = nc.name

    parts = name.split("_")

    date_str = parts[2]
    hm_str = parts[3]

    ts = f"{date_str}_{hm_str}"

    out_file = cache_root / f"{ts}.npy"

    if out_file.exists():
        continue

    try:

        ds = xr.open_dataset(nc)

        img = clean_frame(ds)

        ds.close()

        np.save(out_file, img)

    except Exception as e:

        print("FAILED:", nc)
        print(e)
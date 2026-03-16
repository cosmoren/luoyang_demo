import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def clean_himawari_dataset(ds):
    """
    清洗 Himawari L2 CLP 数据

    保留变量:
    CLTT, CLTH, CLOT, CLTYPE

    清洗规则:
    1. 根据 QA 只保留成功检索像元
    2. 根据 valid_min / valid_max mask 无效值
    3. CLTYPE 去除 255 fill value
    """

    ds = ds.copy()

    # -----------------------------
    # 1. 解析 QA
    # -----------------------------
    qa = ds["QA"].astype(np.uint16)

    # bits (2,1,0) = Cloud Retrieval Algorithm Flag
    algo_flag = qa & 0b111

    good = (algo_flag == 4) | (algo_flag == 5)

    # -----------------------------
    # 2. 对每个变量进行清洗
    # -----------------------------
    vars_to_clean = ["CLTT", "CLTH", "CLOT"]

    for v in vars_to_clean:

        data = ds[v]

        valid_min = data.attrs.get("valid_min", None)
        valid_max = data.attrs.get("valid_max", None)

        clean = data.where(good)

        if valid_min is not None:
            clean = clean.where(clean >= valid_min)

        if valid_max is not None:
            clean = clean.where(clean <= valid_max)

        ds[v] = clean

    # -----------------------------
    # 3. CLTYPE 单独处理
    # -----------------------------
    cltype = ds["CLTYPE"]

    valid_min = cltype.attrs.get("valid_min", 0)
    valid_max = cltype.attrs.get("valid_max", 10)

    cltype_clean = cltype.where(good)
    cltype_clean = cltype_clean.where(cltype_clean >= valid_min)
    cltype_clean = cltype_clean.where(cltype_clean <= valid_max)

    ds["CLTYPE"] = cltype_clean

    return ds

# path = "/home/bb/solar_luoyang/luoyang_demo/nc_processed/NC_H09_20260311_0250_L2CLP010_FLDK.02401_02401.nc"

# ds = xr.open_dataset(path)
# ds = clean_himawari_dataset(ds) # clean the dataset

# # 打印数据结构
# print(ds)

# # 打印全局 metadata
# print("\n==== Global Attributes ====")
# print(ds.attrs)

# # 打印变量 metadata
# print("\n==== Variable Attributes ====")
# for var in ds.data_vars:
#     print(f"\n{var}")
#     print(ds[var].attrs)

# # 打印lat, alt, lon, time的维度
# print("\n==== Dimensions ====")
# print(ds.dims)

# fig, axs = plt.subplots(2,2, figsize=(10,8))

# ds["CLTT"].isel(time=0).plot(ax=axs[0,0], cmap="turbo")
# axs[0,0].set_title("Cloud Top Temperature")

# ds["CLOT"].isel(time=0).plot(ax=axs[0,1], cmap="viridis")
# axs[0,1].set_title("Cloud Optical Thickness")

# ds["CLTYPE"].isel(time=0).plot(ax=axs[1,0], cmap="tab20")
# axs[1,0].set_title("Cloud Type")

# ds["CLTH"].isel(time=0).plot(ax=axs[1,1], cmap="plasma")
# axs[1,1].set_title("Cloud Top Height")

# plt.tight_layout()
# plt.show()
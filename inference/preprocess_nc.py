import os
import re
import numpy as np
import xarray as xr
from datetime import datetime

# ===========================
# 参数
# ===========================

INPUT_ROOTS = [
    #"/home/weize/ai4energy/202501",
    #"/home/weize/ai4energy/202502",
    #"/home/weize/ai4energy/202503",
    #"/home/weize/ai4energy/202504",
    #"/home/weize/ai4energy/202505",
    #"/home/weize/ai4energy/202506",
    #"/home/weize/ai4energy/202507",
    "/home/weize/ai4energy/202508",
    #"/home/weize/ai4energy/202509",
    #"/home/weize/ai4energy/202510",
    #"/home/weize/ai4energy/202511",
    #"/home/weize/ai4energy/202512",
]

OUTPUT_BASE = "/home/weize/ai4energy_crop"

LAT0 = 34.6814 # default is 34.6814
LON0 = 112.29862 # default is 112.29862
SIZE_KM = 500 # default is 500

VARS_TO_KEEP = ["CLTT", "CLTH", "CLOT", "CLTYPE", "QA"]

# ===========================
# 计算裁剪范围
# ===========================

def get_crop_bounds(lat0, lon0, size_km):
    dlat = size_km / 111 / 2
    dlon = size_km / (111 * np.cos(np.deg2rad(lat0))) / 2

    lat_min = lat0 - dlat
    lat_max = lat0 + dlat
    lon_min = lon0 - dlon
    lon_max = lon0 + dlon

    return lat_min, lat_max, lon_min, lon_max


LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = get_crop_bounds(LAT0, LON0, SIZE_KM)

# ===========================
# 从文件名提取时间
# ===========================

def extract_time_from_filename(filename):
    # 匹配 20250104_2310
    m = re.search(r"_(\d{8})_(\d{4})_", filename)

    if m:
        date_str = m.group(1)
        time_str = m.group(2)

        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        return np.datetime64(dt)

    return None

# ===========================
# 单文件处理
# ===========================

def process_file(input_path, output_path):

    try:

        filename = os.path.basename(input_path)

        ds = xr.open_dataset(input_path, decode_timedelta=True)

        # 裁剪区域
        roi = ds.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX)
        )

        # 只保留变量
        roi = roi[VARS_TO_KEEP]

        # 添加时间坐标
        timestamp = extract_time_from_filename(filename)

        if timestamp is not None:
            roi = roi.expand_dims(time=[timestamp])

        # 保留原始 attributes
        roi.attrs = ds.attrs

        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 压缩保存
        encoding = {var: {"zlib": True, "complevel": 4} for var in VARS_TO_KEEP}

        roi.to_netcdf(output_path, encoding=encoding)

        print("✓", input_path)

    except Exception as e:
        print("✗", input_path)
        print(e)

# ===========================
# 批量处理
# ===========================

def batch_process():

    for input_root in INPUT_ROOTS:

        month = os.path.basename(input_root)
        output_root = os.path.join(OUTPUT_BASE, month)

        for root, dirs, files in os.walk(input_root):

            for file in files:

                if file.endswith(".nc"):

                    input_path = os.path.join(root, file)

                    rel_path = os.path.relpath(input_path, input_root)

                    output_path = os.path.join(output_root, rel_path)

                    process_file(input_path, output_path)


if __name__ == "__main__":
    batch_process()
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import shutil
import tempfile

agg_path = Path("../datasets/luoyang_agg.csv")

if agg_path.exists():
    print("Found existing agg file, loading:", agg_path)
    agg = pd.read_csv(agg_path, parse_dates=["time"])
    print("agg file loaded, shape ", agg.shape)
else:
    print("agg file not found, run full preprocessing pipeline...")

    root_zip = Path("/net/storage-1/home/w84179850/canadianlab/weize/data/20260212_luoyang_powerstation.zip")
    work_dir = Path("work_all")

    work_dir.mkdir(exist_ok=True)

    # 1. 解最外层
    with zipfile.ZipFile(root_zip, 'r') as z:
        z.extractall(work_dir)

    base_dir = next(work_dir.iterdir())

    all_dfs = []

    # 2. 遍历每个月压缩包
    for m_idx, month_zip in enumerate(sorted(base_dir.glob("历史数据_*.zip"))):
        #if m_idx > 1:
        #    break
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            with zipfile.ZipFile(month_zip, 'r') as z:
                z.extractall(tmp)

            inv_dir = tmp / "逆变器"
            if not inv_dir.exists():
                continue

            for xlsx_id, xlsx in enumerate(inv_dir.glob("*.xlsx")):
                #if xlsx_id >= 2:
                #    break
                df = pd.read_excel(xlsx, skiprows=3)

                rename_map = {
                    "站点名称": "site",
                    "管理域": "domain",
                    "设备名称": "device",
                    "开始时间": "time",
                    "当日PV发电量(kWh)": "daily_pv_kwh",
                    "当日发电量(kWh)": "daily_kwh",
                    "内部温度(℃)": "inner_temp",
                    "逆变器状态": "status",
                    "输出无功功率(kvar)": "reactive_kvar",
                    "输入总功率(kW)": "input_kw",
                    "有功功率(kW)": "active_kw"
                }

                df = df.rename(columns=rename_map)

                # 只保留我们需要的列（缺列自动 NaN）
                df = df[list(rename_map.values())]
                
                all_dfs.append(df)

    print("files loaded:", len(all_dfs))

    data = pd.concat(all_dfs, ignore_index=True)

    # 3. 基础清洗
    data.replace(["N/A", "-", "—"], np.nan, inplace=True)

    data["time"] = pd.to_datetime(data["time"])

    num_cols = [
        "daily_pv_kwh","daily_kwh","inner_temp",
        "reactive_kvar","input_kw","active_kw"
    ]

    for c in num_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # 4. 排序
    data = data.sort_values(["device","time"])

    # 5. 保存
    data.to_parquet("luoyang_inverter_2025.parquet")
    data.to_csv("luoyang_inverter_2025.csv", index=False)

    print("done: data shape ", data.shape)

    df = data.copy()

    # 先保证类型干净
    df["time"] = pd.to_datetime(df["time"])

    # 防止 status 里有 NaN
    df["status"] = df["status"].astype(str)

    # 先做基础聚合（不考虑并网规则）
    agg = (
        df
        .groupby("time", as_index=False)
        .agg(
            total_active_kw = ("active_kw", lambda x: x.sum(min_count=1)),
            total_daily_pv_kwh = ("daily_pv_kwh", lambda x: x.sum(min_count=1)),
            total_daily_kwh = ("daily_kwh", lambda x: x.sum(min_count=1)),
            mean_inner_temp = ("inner_temp", "mean")
        )
    )

    # -------------------------------
    # 并网有效性判断
    # -------------------------------

    # 每个时间点，是否所有记录都以“并网”开头
    status_ok = (
        df
        .groupby("time")["status"]
        .apply(lambda s: s.astype(str)
                    .str.slice(0, 2)
                    .eq("并网")
                    .fillna(False)
                    .all())
    )

    # 对齐到聚合表
    agg["status_ok"] = agg["time"].map(status_ok).astype(int)

    # 只有不可用时间点，总功率设为 NaN
    agg.loc[agg["status_ok"] == 0, "total_active_kw"] = np.nan

    agg = agg.fillna(0.0)

    agg.to_csv("/net/storage-1/home/w84179850/canadianlab/weize/data/luoyang_agg.csv", index=False)

# dataset generator
import numpy as np
import pandas as pd


class LuoyangDataLoader:

    def __init__(
        self,
        feature_cols=["total_active_kw", "mean_inner_temp"],
        target_col="total_active_kw",
        time_col="time",
        freq_minutes=15,
        add_time_encoding=True,
        solar_forecast_path=None,
        wind_forecast_path=None,
        forecast_feature_config=None,
        satellite_root=None, # 卫星nc根目录，例如 /home/weize/ai4energy_crop
        satellite_cache=None, # 卫星cache目录 i.e. /home/weize/sat_cache
        satellite_vars=("CLTT", "CLTH", "CLOT", "CLTYPE"),
        use_satellite=False,
        satellite_history_len=9, # 过去多少帧卫星图像
        satellite_interval_minutes=10, # 卫星原始时间分辨率
        satellite_expected_hw=(90, 109), # 缺失帧时默认shape
        df=None,
        csv_path=None,
    ):
        """
        feature_cols:
            原始数值特征列，例如:
            ["total_active_kw", "mean_inner_temp"]

        如果 add_time_encoding=True,
        会自动追加:
            time_sin, time_cos

        satellite_root:
            已经 crop 好的 Himawari nc 根目录

        satellite_vars:
            从 nc 中读取哪些变量作为图像通道

        use_satellite:
            是否启用卫星图像

        satellite_history_len:
            每个样本使用过去多少帧卫星图像

        satellite_expected_hw:
            缺失帧时构造零张量的默认空间尺寸
        """

        self.csv_path = csv_path
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.time_col = time_col
        self.freq_minutes = freq_minutes
        self.add_time_encoding = add_time_encoding
        self.solar_forecast_path = solar_forecast_path
        self.wind_forecast_path = wind_forecast_path
        self.use_satellite = use_satellite
        self.satellite_root = Path(satellite_root) if satellite_root is not None else None
        self.satellite_cache = Path(satellite_cache) if satellite_cache is not None else None
        self.satellite_vars = list(satellite_vars)
        self.satellite_history_len = satellite_history_len
        self.satellite_interval_minutes = satellite_interval_minutes
        self.satellite_expected_hw = satellite_expected_hw
        # forcast feature config
        if forecast_feature_config is None:
            forecast_feature_config = {
                "solar": ["ssrd"],
                "wind": [
                    #"t2m",
                    #"msl",
                    "u10",
                    "v10",
                    "wind_speed_10m",
                    #"u100",
                    #"v100",
                    #"wind_speed_100m",
                ]
            }

        self.forecast_feature_config = forecast_feature_config
        if df is None:
            self.df = pd.read_csv(csv_path, parse_dates=[time_col])
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        else:
            self.df = df.copy()
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        self.df = self.df.sort_values(time_col).reset_index(drop=True)

        self.step_per_hour = int(60 // freq_minutes)
        self._load_forecast_data()
        self._build_features()
        #if self.use_satellite:
        self._build_satellite_index()

    # -------------------------------------------------
    # 建立卫星文件时间索引
    # -------------------------------------------------
    def _build_satellite_index(self):
        """
        扫描 satellite_root 下所有 nc 文件，
        从文件名中解析时间，建立:
            timestamp -> filepath
        """

        self.sat_time_to_file = {}

        if self.satellite_root is None:
            raise ValueError("use_satellite=True but satellite_root is None")

        for nc_path in self.satellite_root.rglob("*.nc"):
            name = nc_path.name

            # 文件名示例:
            # NC_H09_20250101_0000_L2CLP010_FLDK.02401_02401.nc
            parts = name.split("_")
            if len(parts) < 4:
                continue

            date_str = parts[2]
            hm_str = parts[3]

            try:
                t = pd.to_datetime(date_str + hm_str, format="%Y%m%d%H%M")
                self.sat_time_to_file[t] = nc_path
            except Exception:
                continue

        self.sat_times = np.array(sorted(self.sat_time_to_file.keys()))

        print(f"[Satellite] indexed {len(self.sat_time_to_file)} nc files.")

    # -------------------------------------------------
    # 给定目标时间 t，选择 <= t 的最近一张卫星图
    # -------------------------------------------------
    def _get_latest_satellite_time(self, t0):
        """
        卫星是10分钟分辨率,PV是15分钟。
        这里采用:
            latest satellite time <= t0
        例如:
            10:15 -> 10:10
            10:30 -> 10:30
        """

        if len(self.sat_times) == 0:
            return None

        idx = np.searchsorted(self.sat_times, t0, side="right") - 1
        if idx < 0:
            return None

        return self.sat_times[idx]

    # -------------------------------------------------
    # 清洗单帧卫星数据
    # -------------------------------------------------
    def _clean_satellite_frame(self, ds):

        qa = ds["QA"].astype(np.uint16)
        algo_flag = qa & 0b111

        good = (algo_flag == 4) | (algo_flag == 5)

        channels = []

        for v in self.satellite_vars:
            data = ds[v].isel(time=0).values.squeeze().astype(np.float32)

            valid_min = ds[v].attrs.get("valid_min", None)
            valid_max = ds[v].attrs.get("valid_max", None)

            mask = good

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

    # -------------------------------------------------
    # 读取一帧卫星图像
    # -------------------------------------------------
    def _load_satellite_frame(self, sat_time):
        """
        输入:
            sat_time: pandas.Timestamp

        输出:
            img: shape = (C+1, H, W)
                 最后一个通道是 valid_mask
            found: 1/0，表示是否真实找到该帧
        """
        if sat_time is None:
            h, w = self.satellite_expected_hw
            c = len(self.satellite_vars) + 1
            return np.zeros((c, h, w), dtype=np.float32), 0

        if False: # back up old code if cache is not there. Run build_satellite_cache.py to build cache first.
            file_path = self.sat_time_to_file.get(sat_time, None)

            if file_path is None:
                h, w = self.satellite_expected_hw
                c = len(self.satellite_vars) + 1
                return np.zeros((c, h, w), dtype=np.float32), 0

            try:
                ds = xr.open_dataset(file_path)
                img = self._clean_satellite_frame(ds)
                ds.close()
                return img, 1
            except Exception:
                h, w = self.satellite_expected_hw
                c = len(self.satellite_vars) + 1
                return np.zeros((c, h, w), dtype=np.float32), 0
        
        cache_file = self.satellite_cache / f"{sat_time:%Y%m%d_%H%M}.npy"
        if not cache_file.exists():
            print("cache not exist. Run build_satellite_cache.py to build cache first")
            h,w = self.satellite_expected_hw
            c = len(self.satellite_vars)+1
            return np.zeros((c,h,w),dtype=np.float32),0
        img = np.load(cache_file)
        img = np.squeeze(img)
        return img,1

    # -------------------------------------------------
    # 构造过去若干帧卫星序列
    # -------------------------------------------------
    def _build_satellite_sequence(self, t0):
        """
        对于一个 anchor time t0，构造过去 satellite_history_len 帧的卫星序列。

        注意：
        - PV 是 15min
        - 卫星原始是 10min
        - 这里仍然按 “PV 的时间步” 回看
        - 每个时刻取 <= 当前时刻的最近一张卫星图

        输出:
            sat_seq: (T_sat, C+1, H, W)
            sat_frame_mask: (T_sat,)
                1 表示该帧真实存在
                0 表示该帧缺失，使用零图填充
        """

        sat_imgs = []
        sat_frame_mask = []

        # 例如 history_len = 9
        # 对应 t0-8*15min, ..., t0
        for k in range(self.satellite_history_len):
            tk = t0 - pd.Timedelta(minutes=self.freq_minutes * (self.satellite_history_len - 1 - k))
            sat_time = self._get_latest_satellite_time(tk)
            img, found = self._load_satellite_frame(sat_time)
            img = np.squeeze(img)
            if img.ndim != 3:
                raise ValueError(f"Unexpected satellite frame shape {img.shape}")

            sat_imgs.append(img)
            sat_frame_mask.append(found)

        sat_seq = np.stack(sat_imgs, axis=0).astype(np.float32)
        sat_frame_mask = np.asarray(sat_frame_mask, dtype=np.float32)

        return sat_seq, sat_frame_mask

    def _load_forecast_data(self):
        # solar
        if self.solar_forecast_path is not None:
            df = pd.read_csv(
                self.solar_forecast_path,
                parse_dates=["start_time", "forecast_time"]
            ).sort_values(["start_time", "forecast_time"])
            self.solar_fcst_df = df
            # build cache
            self.solar_issue_times = np.array(
                sorted(df["start_time"].unique())
            )
            self.solar_issue_map = {
                issue: df[df["start_time"] == issue]
                for issue in self.solar_issue_times
            }
        else:
            self.solar_fcst_df = None
            self.solar_issue_times = None
            self.solar_issue_map = None
        # wind
        if self.wind_forecast_path is not None:
            df = pd.read_csv(
                self.wind_forecast_path,
                parse_dates=["start_time", "forecast_time"]
            ).sort_values(["start_time", "forecast_time"])
            self.wind_fcst_df = df
            # build cache
            self.wind_issue_times = np.array(
                sorted(df["start_time"].unique())
            )
            self.wind_issue_map = {
                issue: df[df["start_time"] == issue]
                for issue in self.wind_issue_times
            }
        else:
            self.wind_fcst_df = None
            self.wind_issue_times = None
            self.wind_issue_map = None

    def _select_latest_issue_cached(
        self,
        issue_times,
        issue_map,
        t0
    ):
        idx = np.searchsorted(issue_times, t0, side="right") - 1
        if idx < 0:
            return None
        issue = issue_times[idx]
        return issue_map[issue]

    def _build_forecast_features(self, t0, target_times):

        feature_blocks = []

        # solar
        if self.solar_fcst_df is not None and \
           len(self.forecast_feature_config["solar"]) > 0:

            solar_issue = self._select_latest_issue_cached(
                self.solar_issue_times,
                self.solar_issue_map,
                t0
            )

            solar_df = solar_issue.set_index("forecast_time")[
                self.forecast_feature_config["solar"]
            ].sort_index()
            full_index = solar_df.index.union(pd.to_datetime(target_times))
            solar_df = solar_df.reindex(full_index).sort_index()
            solar_df = solar_df.ffill().bfill()
            solar_block_df = solar_df.loc[pd.to_datetime(target_times)]
            solar_block = solar_block_df.values

            feature_blocks.append(solar_block)

        # wind
        if self.wind_fcst_df is not None and \
           len(self.forecast_feature_config["wind"]) > 0:

            wind_issue = self._select_latest_issue_cached(
                self.wind_issue_times,
                self.wind_issue_map,
                t0
            )

            wind_df = wind_issue.copy()

            # [ADDED] derived features example
            if "wind_speed_10m" in self.forecast_feature_config["wind"]:
                wind_df["wind_speed_10m"] = np.sqrt(
                    wind_df["u10"]**2 + wind_df["v10"]**2
                )
            if "wind_speed_100m" in self.forecast_feature_config["wind"]:
                wind_df["wind_speed_100m"] = np.sqrt(
                    wind_df["u100"]**2 + wind_df["v100"]**2
                )

            wind_df = wind_df.set_index("forecast_time")[
                self.forecast_feature_config["wind"]
            ].sort_index()
            target_times = pd.to_datetime(target_times)
            full_index = wind_df.index.union(target_times)
            wind_df = wind_df.reindex(full_index).sort_index()
            wind_df = wind_df.ffill().bfill()
            wind_block_df = wind_df.loc[target_times]
            wind_block = wind_block_df.values

            feature_blocks.append(wind_block)

        if len(feature_blocks) == 0:
            return None

        return np.concatenate(feature_blocks, axis=1)

    # -------------------------------------------------
    # 时间 sin / cos 编码
    # -------------------------------------------------
    def _add_time_encoding(self, df):
        """
        使用一天内时刻做周期编码
        """

        minutes = (
            df[self.time_col].dt.hour * 60
            + df[self.time_col].dt.minute
        ).astype(np.float32)

        period = 24 * 60

        df["time_sin"] = np.sin(2 * np.pi * minutes / period)
        df["time_cos"] = np.cos(2 * np.pi * minutes / period)

        return df

    # -------------------------------------------------
    # 构建最终特征矩阵
    # -------------------------------------------------
    def _build_features(self):

        df = self.df.copy()

        if self.add_time_encoding:
            df = self._add_time_encoding(df)
            self.used_feature_cols = (
                self.feature_cols + ["time_sin", "time_cos"]
            )
        else:
            self.used_feature_cols = list(self.feature_cols)

        self.X_all = df[self.used_feature_cols].values.astype(np.float32)
        self.y_all = df[self.target_col].values.astype(np.float32)
        self.time_all = df[self.time_col].values

        self.df = df

    # -------------------------------------------------
    # 通用单点预测数据构造
    # -------------------------------------------------
    def make_single_horizon_dataset(
        self,
        history_len,
        horizon_steps
    ):
        X_tab = []
        X_sat = []
        X_sat_mask = []
        X_fcst = []
        y = []
        curr_gt = []
        t = []

        max_i = len(self.df) - horizon_steps
        count= 0
        for i in range(history_len - 1, max_i):
            count+=1
            if count % 1000 == 0:
                print("count ", count, " total range ", history_len - 1, " ", max_i)
            t0 = pd.to_datetime(self.time_all[i])
            # tabular history
            x_hist = self.X_all[i - history_len + 1 : i + 1] # (T_tab, F_tab)
            # forecast sequence
            target_time = self.time_all[i + horizon_steps]
            fcst = self._build_forecast_features(
                t0,
                [target_time]
            )
            if fcst is None:
                fcst = np.zeros((1, 0), dtype=np.float32)
            # satellite
            if self.use_satellite:
                sat_seq, sat_mask = self._build_satellite_sequence(t0)
                X_sat.append(sat_seq)
                X_sat_mask.append(sat_mask)
            # target
            y_tar = self.y_all[i + horizon_steps]
            X_tab.append(x_hist)
            X_fcst.append(fcst)
            y.append(y_tar)
            curr_gt.append(self.y_all[i])
            t.append(t0)

        X_tab = np.asarray(X_tab, dtype=np.float32)
        X_fcst = np.asarray(X_fcst, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        curr_gt = np.asarray(curr_gt, dtype=np.float32)
        t = np.asarray(t)

        if self.use_satellite:
            X_sat = np.asarray(X_sat, dtype=np.float32)
            X_sat_mask = np.asarray(X_sat_mask, dtype=np.float32)
            return (
                X_tab,
                X_sat,
                X_sat_mask,
                X_fcst,
                y,
                curr_gt,
                t
            )
        else:
            return (
                X_tab,
                X_fcst,
                y,
                curr_gt,
                t
            )

    # -------------------------------------------------
    # ultra short
    # -------------------------------------------------
    def make_ultra_short_dataset(self, history_len):
        return self.make_single_horizon_dataset(
            history_len=history_len,
            horizon_steps=1
        )

    # -------------------------------------------------
    # short
    # -------------------------------------------------
    def make_short_dataset(
        self,
        history_len,
        horizon_hours=4
    ):

        horizon_steps = horizon_hours * self.step_per_hour

        return self.make_single_horizon_dataset(
            history_len=history_len,
            horizon_steps=horizon_steps
        )

    # -------------------------------------------------
    # long
    # -------------------------------------------------
    def make_long_dataset(
        self,
        history_len,
        anchor_hour=9
    ):
        X_tab = []
        X_sat = []
        X_sat_mask = []
        X_fcst = []
        Y = []
        CURR_GT = []
        T = []

        start_offset_steps = 15 * self.step_per_hour
        pred_len = 96

        df = self.df
        count= 0
        for i in range(history_len - 1, len(df)):
            count+=1
            if count % 1000 == 0:
                print("count ", count, " total range ", history_len - 1, " ", len(df))
            current_time = pd.to_datetime(df.iloc[i][self.time_col])
            if not (
                current_time.hour == anchor_hour
                and current_time.minute == 0
            ):
                continue

            start_y = i + start_offset_steps
            end_y = start_y + pred_len

            if end_y > len(df):
                break

            # tabular history
            x_hist = self.X_all[i - history_len + 1 : i + 1]
            # forecast sequence
            target_times = self.time_all[start_y:end_y]

            fcst = self._build_forecast_features(
                t0=current_time,
                target_times=target_times
            )

            if fcst is None:
                fcst = np.zeros((pred_len, 0), dtype=np.float32)
            # satellite
            if self.use_satellite:
                sat_seq, sat_mask = self._build_satellite_sequence(current_time)
                X_sat.append(sat_seq)
                X_sat_mask.append(sat_mask)
            # target
            y_seq = self.y_all[start_y:end_y]
            X_tab.append(x_hist)
            X_fcst.append(fcst)
            Y.append(y_seq)
            CURR_GT.append(self.y_all[i])
            T.append(current_time)

        X_tab = np.asarray(X_tab, dtype=np.float32)
        X_fcst = np.asarray(X_fcst, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        CURR_GT = np.asarray(CURR_GT, dtype=np.float32)
        T = np.asarray(T)

        if self.use_satellite:
            X_sat = np.asarray(X_sat, dtype=np.float32)
            X_sat_mask = np.asarray(X_sat_mask, dtype=np.float32)
            return (
                X_tab,
                X_sat,
                X_sat_mask,
                X_fcst,
                Y,
                CURR_GT,
                T
            )
        else:
            return (
                X_tab,
                X_fcst,
                Y,
                CURR_GT,
                T
            )

    def make_sequence_dataset(
        self,
        history_len,
        horizon_steps
    ):
        X_tab = []
        X_sat = []
        X_sat_mask = []
        X_fcst = []
        Y = []
        CURR_GT = []
        T = []

        max_i = len(self.df) - horizon_steps
        count = 0
        for i in range(history_len - 1, max_i):
            count+=1
            if count % 1000 == 0:
                print("count ", count, " total range ", history_len - 1, " ", max_i)
            t0 = pd.to_datetime(self.time_all[i])
            # tabular history
            x_hist = self.X_all[i - history_len + 1 : i + 1]
            # forecast sequence
            target_times = self.time_all[
                i + 1 : i + 1 + horizon_steps
            ]
            fcst = self._build_forecast_features(
                t0,
                target_times
            )
            if fcst is None:
                fcst = np.zeros((horizon_steps, 0), dtype=np.float32)
            # satellite
            if self.use_satellite:
                sat_seq, sat_mask = self._build_satellite_sequence(t0)
                X_sat.append(sat_seq)
                X_sat_mask.append(sat_mask)
            # target
            y_seq = self.y_all[i + 1 : i + 1 + horizon_steps]
            X_tab.append(x_hist)
            X_fcst.append(fcst)
            CURR_GT.append(self.y_all[i])
            Y.append(y_seq)
            T.append(t0)

        X_tab = np.asarray(X_tab, dtype=np.float32)
        X_fcst = np.asarray(X_fcst, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        CURR_GT = np.asarray(CURR_GT, dtype=np.float32)
        T = np.asarray(T)

        if self.use_satellite:
            X_sat = np.asarray(X_sat, dtype=np.float32)
            X_sat_mask = np.asarray(X_sat_mask, dtype=np.float32)
            return (
                X_tab,
                X_sat,
                X_sat_mask,
                X_fcst,
                Y,
                CURR_GT,
                T
            )
        else:
            return (
                X_tab,
                X_fcst,
                Y,
                CURR_GT,
                T
            )

# 帮助函数：把 X_tab 和 X_fcst flatten 后拼接成 XGBoost 可用的二维特征
def build_xgb_features(X_tab, X_fcst):
    """
    X_tab:  (N, T_tab, F_tab)
    X_fcst: (N, T_fcst, F_fcst)

    返回:
        X_flat: (N, D)
    """
    N = X_tab.shape[0]

    # [NEW] tabular flatten
    X_tab_flat = X_tab.reshape(N, -1)

    # [NEW] forecast flatten
    X_fcst_flat = X_fcst.reshape(N, -1)

    # [NEW] 拼接成最终树模型输入
    X_flat = np.concatenate([X_tab_flat, X_fcst_flat], axis=1)

    return X_flat

# 帮助函数： extract_satellite_features
def extract_satellite_features(X_sat):
    """
    从卫星序列提取简单统计特征

    X_sat shape:
        (N, T_sat, C, H, W)

    返回:
        (N, 3)
    """

    N = X_sat.shape[0]

    feats = []

    for i in range(N):

        sat = X_sat[i]  # (T,C,H,W)

        # flatten temporal
        sat = sat.reshape(-1, sat.shape[1], sat.shape[2], sat.shape[3])
        # (T,C,H,W)

        CLTT = sat[:,0,:,:]
        CLOT = sat[:,2,:,:]

        mean_cltt = np.mean(CLTT)

        mean_clot = np.mean(CLOT)

        cloud_cov = np.mean(CLOT > 0)

        feats.append([
            mean_cltt,
            mean_clot,
            cloud_cov
        ])

    return np.asarray(feats, dtype=np.float32)

# ==========================================================
# unit test（检查结构, shape and reasonable values）
# ==========================================================
def _unit_test():

    print("Running unit tests for LuoyangDataLoader...\n")

    loader = LuoyangDataLoader(
        csv_path=agg_path,
        feature_cols=["total_active_kw", "mean_inner_temp"],
        add_time_encoding=True,
        use_satellite=False,
        satellite_root="/home/weize/ai4energy_crop",
        satellite_cache="/home/weize/sat_cache",
    )

    # =====================================================
    # ultra short
    # =====================================================
    history_len = 3
    X_tab, X_fcst, y, curr_gt, t = loader.make_ultra_short_dataset(
        history_len=history_len
    )

    N = len(y)

    assert X_tab.shape == (N, history_len, 4)
    assert y.shape == (N,)
    assert curr_gt.shape == (N,)
    assert X_fcst.ndim == 3
    assert len(t) == N

    # [新增] NaN 检查
    assert not np.isnan(X_tab).any(), "NaN detected in X_tab"
    assert not np.isnan(X_fcst).any(), "NaN detected in X_fcst"
    assert not np.isnan(y).any(), "NaN detected in y"

    # [新增] Inf 检查
    assert not np.isinf(X_tab).any(), "Inf detected in X_tab"
    assert not np.isinf(X_fcst).any(), "Inf detected in X_fcst"

    # 时间递增检查
    assert np.all(np.diff(t).astype("timedelta64[m]") >= np.timedelta64(0, "m"))

    print("✓ ultra short dataset OK")

    # =====================================================
    # short
    # =====================================================
    X_tab, X_fcst, y, curr_gt, t = loader.make_short_dataset(
        history_len=16,
        horizon_hours=4
    )

    N = len(y)

    assert X_tab.shape[1] == 16
    assert X_tab.shape[2] == 4
    assert y.shape == (N,)
    assert curr_gt.shape == (N,)
    assert X_fcst.ndim == 3

    # [新增] NaN 检查
    assert not np.isnan(X_tab).any(), "NaN detected in X_tab"
    assert not np.isnan(X_fcst).any(), "NaN detected in X_fcst"
    assert not np.isnan(y).any(), "NaN detected in y"

    # [新增] Inf 检查
    assert not np.isinf(X_tab).any(), "Inf detected in X_tab"
    assert not np.isinf(X_fcst).any(), "Inf detected in X_fcst"

    print("✓ short dataset OK")

    # =====================================================
    # long
    # =====================================================
    X_tab, X_fcst, Y, curr_gt, t = loader.make_long_dataset(
        history_len=32,
        anchor_hour=9
    )

    if len(X_tab) > 0:

        N = len(X_tab)

        assert X_tab.shape == (N, 32, 4)
        assert Y.shape == (N, 96)

        assert X_fcst.ndim == 3
        assert curr_gt.shape == (N,)

        # [新增] NaN 检查
        assert not np.isnan(X_tab).any(), "NaN detected in X_tab"
        assert not np.isnan(X_fcst).any(), "NaN detected in X_fcst"
        assert not np.isnan(Y).any(), "NaN detected in Y"

        # [新增] Inf 检查
        assert not np.isinf(X_tab).any(), "Inf detected in X_tab"
        assert not np.isinf(X_fcst).any(), "Inf detected in X_fcst"

        # anchor time 必须是 09:00
        hours = pd.to_datetime(t).hour
        assert np.all(hours == 9)

    print("✓ long dataset OK")

    # =====================================================
    # sequence dataset
    # =====================================================
    X_tab, X_fcst, Y, curr_gt, t = loader.make_sequence_dataset(
        history_len=32,
        horizon_steps=16
    )

    N = len(X_tab)

    assert X_tab.shape == (N, 32, 4)
    assert Y.shape == (N, 16)
    assert X_fcst.shape[1] == 16

    # [新增] NaN 检查
    assert not np.isnan(X_tab).any(), "NaN detected in X_tab"
    assert not np.isnan(X_fcst).any(), "NaN detected in X_fcst"
    assert not np.isnan(Y).any(), "NaN detected in Y"

    # [新增] Inf 检查
    assert not np.isinf(X_tab).any(), "Inf detected in X_tab"
    assert not np.isinf(X_fcst).any(), "Inf detected in X_fcst"

    print("✓ sequence dataset OK")

    # =====================================================
    # satellite enabled test
    # =====================================================
    loader_sat = LuoyangDataLoader(
        csv_path=agg_path,
        feature_cols=["total_active_kw", "mean_inner_temp"],
        add_time_encoding=True,
        use_satellite=True,
        satellite_root="/home/weize/ai4energy_crop",
        satellite_cache="/home/weize/sat_cache",
    )

    print("Dataloader done.")

    X_tab, X_sat, X_sat_mask, X_fcst, y, curr_gt, t = \
        loader_sat.make_ultra_short_dataset(history_len=8)

    print("make_ultra_short_dataset done.")

    N = len(y)

    assert X_sat.ndim == 5
    assert X_sat_mask.ndim == 2

    assert X_sat.shape[0] == N
    assert X_sat_mask.shape[0] == N

    # mask 只能是 0 或 1
    assert np.all((X_sat_mask == 0) | (X_sat_mask == 1))

    # [新增] NaN 检查
    assert not np.isnan(X_sat).any(), "NaN detected in X_sat"
    assert not np.isnan(X_fcst).any(), "NaN detected in X_fcst"
    assert not np.isnan(y).any(), "NaN detected in y"

    # [新增] Inf 检查
    assert not np.isinf(X_sat).any(), "Inf detected in X_sat"

    print("✓ satellite dataset OK")

    print("\nAll unit tests passed successfully.")

if __name__ == "__main__":
    _unit_test()
    breakpoint()
    capacity = 0.5 * 117 * 1e3
    loader = LuoyangDataLoader(
        csv_path=agg_path,
        feature_cols=[
            "total_active_kw",
            "mean_inner_temp",
            "status_ok"
        ],
        add_time_encoding=True,
        solar_forecast_path="/home/weize/ai4energy/luoyang_demo/datasets/112.285_34.700_UTC0_model_solar_v5.csv",
        wind_forecast_path="/home/weize/ai4energy/luoyang_demo/datasets/112.285_34.700_UTC0_model_wind_v5.csv",
        forecast_feature_config={
            "solar": [
                "ssrd"
            ],
            "wind": [
                #"t2m",
                #"msl",
                "u10",
                "v10",
                "wind_speed_10m",
                #"u100",
                #"v100",
                #"wind_speed_100m",
            ]
        },
        use_satellite=True,
        satellite_root="/home/weize/ai4energy_crop",
        satellite_cache="/home/weize/sat_cache",
    )
    # train and eval
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    # -------------------------------------------------------
    # 构造数据
    # -------------------------------------------------------

    mode = "short" #ultra-short", "short", "long", "windowed-long"
    if mode == "ultra-short":
        X_tab, X_sat, X_sat_mask, X_fcst, y, curr_gt, t = loader.make_ultra_short_dataset(history_len=32)
    elif mode == "short":
        X_tab, X_sat, X_sat_mask, X_fcst, y, curr_gt, t = loader.make_short_dataset(history_len=64, horizon_hours=4)
    elif mode == "long":
        X_tab, X_sat, X_sat_mask, X_fcst, y, curr_gt, t = loader.make_long_dataset(history_len=96, anchor_hour=9)
    elif mode == "windowed-long":
        X_tab, X_sat, X_sat_mask, X_fcst, y, curr_gt, t = loader.make_sequence_dataset(history_len=96, horizon_steps=192)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # XGBoost demo
    model_path = (
        agg_path.parent
        / f"{agg_path.stem}_{mode}_xgb.json"
    )
    
    # ---------------------------------------------------------
    # 判断是否是 sequence 模式, long or windowed-long
    # ---------------------------------------------------------

    is_sequence = (y.ndim == 2)

    print("is_sequence  mode:", is_sequence)

    # ---------------------------------------------------------
    # 用新 helper 构造 XGBoost 输入
    # ---------------------------------------------------------
    if loader.use_satellite == False:
        X_flat = build_xgb_features(X_tab, X_fcst)
    else:
        X_tab_flat = X_tab.reshape(X_tab.shape[0], -1)
        X_fcst_flat = X_fcst.reshape(X_fcst.shape[0], -1)

        # NEW
        X_sat_feat = extract_satellite_features(X_sat)

        # concatenate
        X_flat = np.concatenate(
            [X_tab_flat, X_fcst_flat, X_sat_feat],
            axis=1
        )
    # ---------------------------------------------------------
    # 构造监督样本
    # ---------------------------------------------------------

    if not is_sequence:
        # -------------------------------
        # ultra short / short
        # -------------------------------
        y_flat = y
        t_flat = pd.to_datetime(t)

    else:
        # -------------------------------
        # is_sequence
        # 每天一个样本 -> n_seq个点
        # 展开成 N*seq 个样本
        # -------------------------------

        y_list = []
        t_list = []
        X_list = []

        for i in range(X_flat.shape[0]):
            # t[i] 是 anchor time（9点）
            base_time = pd.to_datetime(t[i])

            # 第一个预测点时间：+15小时 or 15 minutes
            if mode == "long":
                start_time = base_time + pd.Timedelta(hours=15)
            if mode == "windowed-long":
                start_time = base_time + pd.Timedelta(minutes=loader.freq_minutes)

            times = pd.date_range(
                start_time,
                periods=y.shape[1],
                freq=f"{loader.freq_minutes}min"
            )

            for k in range(y.shape[1]):
                X_list.append(X_flat[i])
                y_list.append(y[i, k])
                t_list.append(times[k])

        X_flat = np.asarray(X_list, dtype=np.float32)
        y_flat = np.asarray(y_list, dtype=np.float32)
        t_flat = pd.to_datetime(t_list)

    print("Final dataset shape:")
    print("X:", X_flat.shape)
    print("y:", y_flat.shape)

    # ---------------------------------------------------------
    # 按预测时间排序
    # ---------------------------------------------------------

    order = np.argsort(t_flat.values)

    X_flat = X_flat[order]
    y_flat = y_flat[order]
    t_flat = t_flat.values[order]

    # ---------------------------------------------------------
    # 70 / 30 时间切分
    # ---------------------------------------------------------

    split = int(len(X_flat) * 0.7)

    X_train = X_flat[:split]
    y_train = y_flat[:split]

    X_test = X_flat[split:]
    y_test = y_flat[split:]

    print("Train size:", X_train.shape)
    print("Test  size:", X_test.shape)

    # ---------------------------------------------------------
    # 训练 XGBoost
    # ---------------------------------------------------------

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    model.save_model(model_path)
    # ---------------------------------------------------------
    # 预测 & 评测
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("MAE :", mae, " acc ", 1.0 - mae / capacity)
    print("RMSE:", rmse, " acc ", 1.0 - rmse / capacity)

    '''
    # ---------------------------------------------------------
    # long 模式下的额外评测（按天还原）
    # ---------------------------------------------------------

    if is_sequence:

        df_eval = pd.DataFrame({
            "time": t_flat[split:],
            "y_true": y_test,
            "y_pred": y_pred,
        })

        df_eval["date"] = df_eval["time"].dt.floor("D")

        daily_metrics = (
        df_eval
        .groupby("date")
        .apply(
            lambda x: pd.Series({
                "daily_rmse": np.sqrt(
                    mean_squared_error(x["y_true"], x["y_pred"])
                ),
                "daily_mae": mean_absolute_error(
                    x["y_true"], x["y_pred"]
                )
            })
        )
        .reset_index()
    )

    print(daily_metrics.head())
    '''
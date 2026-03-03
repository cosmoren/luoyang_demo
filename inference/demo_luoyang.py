import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
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
        csv_path,
        feature_cols=["total_active_kw", "mean_inner_temp"],
        target_col="total_active_kw",
        time_col="time",
        freq_minutes=15,
        add_time_encoding=True,
        solar_forecast_path=None,
        wind_forecast_path=None,
        forecast_feature_config=None,
    ):
        """
        feature_cols:
            原始数值特征列，例如:
            ["total_active_kw", "mean_inner_temp"]

        如果 add_time_encoding=True，
        会自动追加:
            time_sin, time_cos
        """

        self.csv_path = csv_path
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.time_col = time_col
        self.freq_minutes = freq_minutes
        self.add_time_encoding = add_time_encoding
        self.solar_forecast_path = solar_forecast_path
        self.wind_forecast_path = wind_forecast_path
        # forcast feature config
        if forecast_feature_config is None:
            forecast_feature_config = {
                "solar": ["ssrd"],
                "wind": [
                    "t2m",
                    #"msl",
                    #"u10",
                    #"v10",
                    #"u100",
                    #"v100",
                    #"wind_speed_10m",
                ]
            }

        self.forecast_feature_config = forecast_feature_config
        self.df = pd.read_csv(csv_path, parse_dates=[time_col])
        self.df = self.df.sort_values(time_col).reset_index(drop=True)

        self.step_per_hour = int(60 // freq_minutes)
        self._load_forecast_data()
        self._build_features()

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

            solar_block = solar_issue.set_index("forecast_time")[
                self.forecast_feature_config["solar"]
            ].reindex(target_times).values

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

            wind_block = wind_df.set_index("forecast_time")[
                self.forecast_feature_config["wind"]
            ].reindex(target_times).values

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

        X = []
        y = []
        t = []

        max_i = len(self.df) - horizon_steps

        for i in range(history_len - 1, max_i):
            t0 = self.time_all[i]
            x_hist = self.X_all[i - history_len + 1: i + 1]
            target_time = self.time_all[i + horizon_steps]
            fcst = self._build_forecast_features(
                t0,
                [target_time]
            )
            if fcst is not None:
                x_hist = np.concatenate([
                    x_hist.flatten(),
                    fcst.flatten()
                ])
            else:
                x_hist = x_hist.flatten()
            y_tar = self.y_all[i + horizon_steps]

            X.append(x_hist)
            y.append(y_tar)
            t.append(t0)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(t)
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
        """
        每天 anchor_hour 作为当前时刻
        预测:
            t + 15 小时开始
            连续 96 个 15 分钟点
        """

        X = []
        Y = []
        T = []

        start_offset_steps = 15 * self.step_per_hour
        pred_len = 96

        df = self.df

        for i in range(history_len - 1, len(df)):

            current_time = df.iloc[i][self.time_col]

            if not (
                current_time.hour == anchor_hour
                and current_time.minute == 0
            ):
                continue

            start_y = i + start_offset_steps
            end_y = start_y + pred_len

            if end_y > len(df):
                break

            x_hist = self.X_all[i - history_len + 1: i + 1]
            y_seq = self.y_all[start_y:end_y]

            target_times = self.time_all[start_y:end_y]
            fcst = self._build_forecast_features(
                t0=current_time,
                target_times=target_times
            )
            if fcst is not None:
                x = np.concatenate([
                    x_hist.flatten(),
                    fcst.flatten()
                ])
            else:
                x = x_hist.flatten()

            X.append(x)
            Y.append(y_seq)
            T.append(current_time)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
            np.asarray(T)
        )

    def make_sequence_dataset(
        self,
        history_len,
        horizon_steps
    ):
        """
        通用序列预测
        Input:
            history_len: 历史长度
            horizon_steps: 预测步数（192 = 48h）
        Output:
            X shape = (N, history_len, F)
            Y shape = (N, horizon_steps)
        """
        X = []
        Y = []
        T = []

        max_i = len(self.df) - horizon_steps

        for i in range(history_len - 1, max_i):
            t0 = self.time_all[i]
            x_hist = self.X_all[i - history_len + 1 : i + 1]
            target_times = self.time_all[
                i+1 : i+1+horizon_steps
            ]
            fcst = self._build_forecast_features(
                t0,
                target_times
            )
            if fcst is not None:
                x = np.concatenate([
                    x_hist.flatten(),
                    fcst.flatten()
                ])
            else:
                x = x_hist.flatten()
            
            y_seq = self.y_all[i + 1 : i + 1 + horizon_steps]
            X.append(x_hist)
            Y.append(y_seq)
            T.append(t0)

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
            np.asarray(T)
        )


# ==========================================================
# unit test（只检查结构与 shape）
# ==========================================================

def _unit_test():

    print("Running unit test...")

    loader = LuoyangDataLoader(
        csv_path=agg_path,
        feature_cols=["total_active_kw", "mean_inner_temp"],
        add_time_encoding=True
    )

    # ---------- ultra short ----------
    X, y, t = loader.make_ultra_short_dataset(history_len=8)

    assert X.ndim == 3
    assert y.ndim == 1
    assert X.shape[1] == 8
    assert X.shape[2] == 4   # 2原始特征 + sin + cos

    # ---------- short ----------
    Xs, ys, ts = loader.make_short_dataset(
        history_len=16,
        horizon_hours=4
    )

    assert Xs.ndim == 3
    assert ys.ndim == 1
    assert Xs.shape[1] == 16
    assert Xs.shape[2] == 4

    # ---------- long ----------
    Xl, Yl, tl = loader.make_long_dataset(
        history_len=32,
        anchor_hour=9
    )

    if len(Xl) > 0:
        assert Xl.ndim == 3
        assert Yl.ndim == 2
        assert Xl.shape[1] == 32
        assert Xl.shape[2] == 4
        assert Yl.shape[1] == 96

    print("Unit test passed.")
    print("ultra short X shape:", X.shape)
    print("short       X shape:", Xs.shape)
    print("long        X shape:", Xl.shape)

if __name__ == "__main__":
    #_unit_test()
    capacity = 0.5 * 117 * 1e3
    loader = LuoyangDataLoader(
        csv_path=agg_path,
        feature_cols=[
            "total_active_kw",
            "mean_inner_temp",
            "status_ok"
        ],
        add_time_encoding=True,
        solar_forecast_path="/home/kyber/Desktop/pv_forcast/pv_forcast/datasets/112.285_34.700_UTC0_model_solar_v5.csv",
        wind_forecast_path="/home/kyber/Desktop/pv_forcast/pv_forcast/datasets/112.285_34.700_UTC0_model_wind_v5.csv",
        forecast_feature_config={
            "solar": [
                "ssrd"
            ],
            "wind": [
                "t2m",
                #"msl",
                #"u10",
                #"v10",
                #"u100",
                #"v100",
                #"wind_speed_10m"
            ]
        },
    )
    # train and eval
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    # -------------------------------------------------------
    # 构造数据
    # -------------------------------------------------------

    mode = "ultra-short" # "ultra-short", "short", "long", "windowed-long"
    if mode == "ultra-short":
        X, y, t = loader.make_ultra_short_dataset(history_len=32)
    elif mode == "short":
        X, y, t = loader.make_short_dataset(history_len=64, horizon_hours=4)
    elif mode == "long":
        X, y, t = loader.make_long_dataset(history_len=96, anchor_hour=9)
    elif mode == "windowed-long":
        X, y, t = loader.make_sequence_dataset(history_len=96, horizon_steps=192)
    else:
        raise ValueError(f"Unknown mode: {mode}")

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
    # 展平 X
    # ---------------------------------------------------------
    #N, T, F = X.shape
    N = X.shape[0]
    X_flat = X.reshape(N, -1)

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

        for i in range(N):
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
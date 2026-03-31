"""
Online inference GUI: PV forecast panels (short/mid/long term), left panel
(CLOT, sky images, ssrd/t2m), and pulldown selection for 洛阳电站 / 雅砻江电站.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sys
import yaml

import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

UTC8 = timezone(timedelta(hours=8))

from infer_online_alg import infer_online_alg
from inspect_nc import clean_himawari_dataset
import xarray as xr
import re
from preprocess_nc import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, VARS_TO_KEEP
from config_utils import get_infer_online_data_paths

_XINSHENG_EXCEL = _PROJECT_ROOT / "datasets" / "XINSHENG_power_total_devices_type1.xlsx"
_CONF_PATH = _PROJECT_ROOT / "config" / "conf.yaml"


def _load_infer_online_paths() -> dict[str, Path]:
    with open(_CONF_PATH) as f:
        conf = yaml.safe_load(f)
    return get_infer_online_data_paths(conf, _PROJECT_ROOT)


def _devdn_display_name(raw: str) -> str:
    """Remove 'NE=' from devDn and retain only digits for display."""
    if pd.isna(raw) or not isinstance(raw, str):
        return ""
    s = str(raw).strip()
    if s.upper().startswith("NE="):
        s = s[3:]
    return "".join(c for c in s if c.isdigit()) or s


def _load_plant_devdn_from_excel(excel_path: Path):
    """Return (list of plant names, dict plant -> list of devDn display names)."""
    if not excel_path.is_file():
        return [], {}
    df = pd.read_excel(excel_path, sheet_name=0)
    plant_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("plantname", "plant name", "站点名称"):
            plant_col = c
            break
    if plant_col is None and "plantName" in df.columns:
        plant_col = "plantName"
    dev_col = "devDn" if "devDn" in df.columns else None
    for c in df.columns:
        if str(c).strip().lower() in ("devdn", "dev_dn"):
            dev_col = c
            break
    if plant_col is None or dev_col is None:
        return [], {}
    df = df[[plant_col, dev_col]].dropna(how="all")
    df[dev_col] = df[dev_col].astype(str).map(_devdn_display_name)
    df = df[df[dev_col].str.len() > 0]
    plants = df[plant_col].dropna().astype(str).unique().tolist()
    plant_to_devdns = {}
    for p in plants:
        devs = df.loc[df[plant_col] == p, dev_col].dropna().unique().tolist()
        plant_to_devdns[p] = devs
    return plants, plant_to_devdns


try:
    from PIL import Image, ImageTk

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_p_infer = _load_infer_online_paths()
NC_PROCESSED_DIR = _p_infer["nc_processed"]
SKY_IMAGE_DIR = _p_infer["sky_image"]
SKY_IMAGE_PRED_DIR = _p_infer["sky_image_pred"]
NWP_SOLAR_DIR = _p_infer["nwp_solar"]
NWP_WIND_DIR = _p_infer["nwp_wind"]
_SKY_IMAGE_TIME_FMTS = [
    "%Y-%m-%d_%H-%M",
    "%Y-%m-%d_%H-%M-%S",
    "%Y%m%d_%H%M%S",
    "%Y%m%d_%H%M",
]


def _parse_sky_image_stem(stem: str) -> datetime | None:
    """Parse sky image filename stem; supports YYYYMMDDHHMMSS_11 / _12 and legacy patterns."""
    m = re.match(r"^(\d{14})_(11|12)$", stem)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            return None
    for fmt in _SKY_IMAGE_TIME_FMTS:
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    return None


def _load_most_recent_clot(nc_dir: Path):
    """Load latest *valid* NC from nc_dir, crop to ROI, return (path, CLOT) or None.

    If the most recent file is incomplete/corrupted (e.g. still downloading),
    it will fail to open or process and we fall back to the previous file.
    """
    if not nc_dir.is_dir():
        return None
    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        return None
    # Try newest to oldest; first file that opens/plots successfully is used.
    for path in reversed(nc_files):
        try:
            if path.stat().st_size <= 0:
                continue
            ds = xr.open_dataset(path, decode_timedelta=True)
            roi = ds.sel(
                latitude=slice(LAT_MAX, LAT_MIN),
                longitude=slice(LON_MIN, LON_MAX),
            )
            roi = roi[VARS_TO_KEEP]
            try:
                roi = clean_himawari_dataset(roi)
            except Exception:
                pass
            clot = roi["CLOT"]
            if "time" in clot.dims:
                clot = clot.isel(time=0)
            ds.close()
            return (path, clot)
        except Exception:
            # If this file is incomplete or corrupt, try the previous one.
            continue
    return None


def _load_latest_nwp_series(dir_path: Path, value_col: str):
    """Load latest NWP CSV in dir_path and return Series indexed by forecast_time (UTC+8 naive)."""
    if not dir_path.is_dir():
        return None
    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        return None
    path = csv_files[-1]
    try:
        df = pd.read_csv(path)
        # Heuristic: find time column, default to 'forecast_time'
        time_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("forecast_time", "time", "forecasttime"):
                time_col = c
                break
        if time_col is None or value_col not in df.columns:
            return None
        times = pd.to_datetime(df[time_col])
        # Treat as UTC+8 naive timestamps
        s = pd.Series(df[value_col].values, index=times)
        s = s.sort_index()
        return s
    except Exception:
        return None


def _nc_path_to_time_label(path: Path) -> str:
    """Return NC timestamp label in UTC+8 as 'YYYY-MM-DD HH:MM:SS (UTC+8)'.

    Filenames are assumed to encode UTC as ..._YYYYMMDD_HHMM_...
    """
    stem = path.stem
    m = re.search(r"_(\d{8})_(\d{4})_", stem)
    if not m:
        return ""
    d, t = m.group(1), m.group(2)
    try:
        dt_utc = datetime(
            year=int(d[:4]),
            month=int(d[4:6]),
            day=int(d[6:8]),
            hour=int(t[:2]),
            minute=int(t[2:4]),
            second=0,
            tzinfo=timezone.utc,
        )
        dt_local = dt_utc.astimezone(UTC8)
        return dt_local.strftime("%Y-%m-%d %H:%M:%S") + " (UTC+8)"
    except Exception:
        return ""


def _find_sky_image_for_time(dir_path: Path, t: datetime) -> Path | None:
    """Find sky image with timestamp closest to t; try exact 15-min filename first.

    New names use YYYYMMDDHHMMSS_11 or _12.png (same time); either suffix is fine—pick at random.
    """
    if not dir_path.is_dir() or not _PIL_AVAILABLE:
        return None
    t_rounded = t.replace(second=0, microsecond=0)
    minute = (t_rounded.minute // 15) * 15
    t_rounded = t_rounded.replace(minute=minute)
    exact_names = [t_rounded.strftime("%Y-%m-%d_%H-%M-%S") + ".png"]
    ts_compact = t_rounded.strftime("%Y%m%d%H%M%S")
    suffix_order = ["11", "12"]
    random.shuffle(suffix_order)
    exact_names.extend(f"{ts_compact}_{suf}.png" for suf in suffix_order)
    for name in exact_names:
        exact_path = dir_path / name
        if exact_path.is_file():
            return exact_path
    candidates: list[Path] = []
    best_delta: float | None = None
    t_ts = t.timestamp()
    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        parsed = _parse_sky_image_stem(f.stem)
        if parsed is None:
            continue
        delta = abs(parsed.timestamp() - t_ts)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            candidates = [f]
        elif delta == best_delta:
            candidates.append(f)
    if not candidates:
        return None
    return random.choice(candidates)


def _load_sky_image(path: Path, max_size: tuple[int, int] = (128, 128)):
    if not path.exists() or not _PIL_AVAILABLE:
        return None
    try:
        img = Image.open(path).convert("RGB")
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img = img.resize(max_size, resample)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def create_infer_online_gui():
    """Create GUI: PV forecast panels, left panel (CLOT, sky, ssrd/t2m), pulldowns."""
    root = tk.Tk()
    root.title("Inference Online")
    root.geometry("1100x650")
    try:
        root.attributes("-zoomed", True)
    except Exception:
        try:
            root.state("zoomed")
        except Exception:
            pass

    t0_utc = datetime.now(timezone.utc)
    top_frame = tk.Frame(root, padx=10, pady=5)
    top_frame.pack(fill=tk.X)
    tk.Label(top_frame, text="Current time (UTC):", font=("", 10)).pack(side=tk.LEFT)
    utc_var = tk.StringVar(value=t0_utc.strftime("%Y-%m-%d %H:%M:%S"))
    tk.Entry(top_frame, width=24, font=("", 11), state="readonly", textvariable=utc_var).pack(
        side=tk.LEFT, padx=(5, 15)
    )
    tk.Label(top_frame, text="Current time (UTC+8):", font=("", 10)).pack(side=tk.LEFT, padx=(0, 5))
    now_local_utc8 = datetime.now(UTC8)
    local_var = tk.StringVar(
        value=f"{now_local_utc8.strftime('%Y-%m-%d %H:%M:%S')} UTC+8"
    )
    tk.Entry(top_frame, width=28, font=("", 11), state="readonly", textvariable=local_var).pack(
        side=tk.LEFT, padx=(0, 0)
    )

    checkpoint_path = _PROJECT_ROOT / "checkpoints" / "pv_forecast_epoch_10.pt"
    alg = infer_online_alg(checkpoint_path=str(checkpoint_path))
    latest_pv_pred: list = [None]

    def run_periodic_inference():
        try:
            result = alg.inference()
            latest_pv_pred[0] = result
            update_forecast_plots()
        except Exception:
            pass
        root.after(5 * 60 * 1000, run_periodic_inference)

    root.after(10 * 1000, run_periodic_inference)

    content = tk.Frame(root, padx=10, pady=10)
    content.pack(expand=True, fill=tk.BOTH)
    content.grid_columnconfigure(0, weight=0)
    content.grid_columnconfigure(1, weight=1)
    content.grid_rowconfigure(0, weight=1)

    left_panel = tk.Frame(content)
    left_panel.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

    right_panel = tk.Frame(content)
    right_panel.grid(row=0, column=1, sticky="nsew")
    right_panel.grid_columnconfigure(0, weight=1)
    right_panel.grid_rowconfigure(0, weight=1)

    CLOT_BASE_INCHES = 3.0
    fig_dpi = 100
    fig_clot = Figure(figsize=(2, 2), dpi=fig_dpi)

    clot_frame = tk.Frame(left_panel, relief=tk.GROOVE, borderwidth=1, padx=4, pady=4)
    clot_frame.pack(anchor="nw", fill=tk.X, pady=(0, 8))
    tk.Label(clot_frame, text="Cloud Thickness", font=("", 10)).pack()
    canvas_clot = FigureCanvasTkAgg(fig_clot, master=clot_frame)
    canvas_clot.get_tk_widget().pack()
    clot_time_label = tk.Label(clot_frame, text="", font=("", 9))
    clot_time_label.pack(pady=(2, 0))

    def update_clot_panel():
        fig_clot.clf()
        item = _load_most_recent_clot(NC_PROCESSED_DIR)
        ax = fig_clot.add_subplot(1, 1, 1)
        if item is not None:
            path, clot = item
            clot_time_label.config(text=_nc_path_to_time_label(path))
            ny, nx = int(clot.shape[0]), int(clot.shape[1])
            aspect = (nx / ny) if ny else 1.0
            if nx >= ny:
                base_w, base_h = CLOT_BASE_INCHES, CLOT_BASE_INCHES / aspect
            else:
                base_h, base_w = CLOT_BASE_INCHES, CLOT_BASE_INCHES * aspect
            fig_clot.set_size_inches(base_w, base_h)
            clot.plot(ax=ax, cmap="viridis", add_colorbar=False)
        else:
            clot_time_label.config(text="")
            fig_clot.set_size_inches(2, 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        fig_clot.tight_layout(pad=0.1)
        canvas_clot.draw()

    font_time = ("", 10)
    sky_photo_ref: list = []
    blank_photo = None
    if _PIL_AVAILABLE:
        try:
            from PIL import Image as _ImgMod, ImageTk as _ImgTkMod  # type: ignore
            _blank_img = _ImgMod.new("RGB", (128, 128), (255, 255, 255))
            blank_photo = _ImgTkMod.PhotoImage(_blank_img)
        except Exception:
            blank_photo = None

    sky_group = tk.LabelFrame(left_panel, text="Sky images", padx=4, pady=4)
    sky_group.pack(anchor="nw", fill=tk.X)

    sky_row = tk.Frame(sky_group)
    sky_row.pack()

    sky_left_frame = tk.Frame(sky_row)
    sky_left_frame.pack(side=tk.LEFT, padx=(0, 8))
    sky_time_label_left = tk.Label(sky_left_frame, text="--:--:--", font=font_time)
    sky_time_label_left.pack()
    sky_image_label_left = tk.Label(sky_left_frame, text="No Image", font=font_time, fg="gray")
    sky_image_label_left.pack()
    tk.Label(sky_left_frame, text="Current", font=font_time).pack()

    sky_right_frame = tk.Frame(sky_row)
    sky_right_frame.pack(side=tk.LEFT)
    tk.Label(sky_right_frame, text="+15 min", font=font_time).pack()
    sky_image_label_right = tk.Label(sky_right_frame, text="No Image", font=font_time, fg="gray")
    sky_image_label_right.pack()
    tk.Label(sky_right_frame, text="Predicted", font=font_time).pack()

    met_frame = tk.LabelFrame(left_panel, text="Forecast data", padx=4, pady=4)
    met_frame.pack(anchor="nw", fill=tk.X, pady=(8, 0))
    ssrd_frame = tk.Frame(met_frame)
    ssrd_frame.pack(anchor="nw", pady=(0, 6))
    tk.Label(ssrd_frame, text="SSRD", font=("", 10)).grid(row=0, column=0, columnspan=2, sticky="w")
    ssrd_time1_label = tk.Label(ssrd_frame, text="--:--:--", font=("", 9))
    ssrd_time1_label.grid(row=1, column=0, padx=(0, 8))
    ssrd_time2_label = tk.Label(ssrd_frame, text="--:--:--", font=("", 9))
    ssrd_time2_label.grid(row=1, column=1)
    ssrd_val1_label = tk.Label(ssrd_frame, text="0", font=("", 9))
    ssrd_val1_label.grid(row=2, column=0, padx=(0, 8))
    ssrd_val2_label = tk.Label(ssrd_frame, text="0", font=("", 9))
    ssrd_val2_label.grid(row=2, column=1)
    wind_frame = tk.Frame(met_frame)
    wind_frame.pack(anchor="nw")
    tk.Label(wind_frame, text="Wind speed", font=("", 10)).grid(row=0, column=0, columnspan=2, sticky="w")
    wind_time1_label = tk.Label(wind_frame, text="--:--:--", font=("", 9))
    wind_time1_label.grid(row=1, column=0, padx=(0, 8))
    wind_time2_label = tk.Label(wind_frame, text="--:--:--", font=("", 9))
    wind_time2_label.grid(row=1, column=1)
    wind_val1_label = tk.Label(wind_frame, text="0", font=("", 9))
    wind_val1_label.grid(row=2, column=0, padx=(0, 8))
    wind_val2_label = tk.Label(wind_frame, text="0", font=("", 9))
    wind_val2_label.grid(row=2, column=1)

    LEVEL1_OPTIONS = ["洛阳电站", "雅砻江电站"]
    _luoyang_plants, _luoyang_plant_to_devdns = _load_plant_devdn_from_excel(_XINSHENG_EXCEL)
    LEVEL2_OPTIONS = {
        "洛阳电站": [""] + _luoyang_plants,
        "雅砻江电站": ["", "Y1", "Y2", "Y3"],
    }
    LEVEL3_OPTIONS = {"": [""]}
    for _p in _luoyang_plants:
        LEVEL3_OPTIONS[_p] = [""] + _luoyang_plant_to_devdns.get(_p, [])
    LEVEL3_OPTIONS.update({
        "Y1": ["", "Y1_1", "Y1_2", "Y1_3"],
        "Y2": ["", "Y2_1", "Y2_2", "Y2_3"],
        "Y3": ["", "Y3_1", "Y3_2", "Y3_3"],
    })
    dropdown_frame = tk.Frame(right_panel)
    dropdown_frame.grid(row=0, column=0, columnspan=1, sticky="ew", padx=2, pady=(0, 4))
    right_panel.grid_rowconfigure(0, minsize=36)
    right_panel.grid_rowconfigure(1, weight=1)
    right_panel.grid_columnconfigure(0, weight=1)
    right_panel.grid_columnconfigure(1, weight=1)
    for _c in range(3):
        dropdown_frame.grid_columnconfigure(_c, weight=1)

    fig = Figure(figsize=(7.5, 6), dpi=100)
    ax_top = fig.add_subplot(3, 1, 1)
    ax_mid = fig.add_subplot(3, 1, 2)
    ax_bot = fig.add_subplot(3, 1, 3)
    ax_top.set_ylabel("PV power (kW)")
    ax_top.set_title("Short-term (4 h forecast)")
    ax_top.grid(True, alpha=0.3)
    ax_mid.set_ylabel("PV power (kW)")
    ax_mid.set_title("Mid-term (24 h forecast)")
    ax_mid.grid(True, alpha=0.3)
    ax_bot.set_ylabel("PV power (kW)")
    ax_bot.set_title("Long-term (48 h forecast)")
    ax_bot.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.autofmt_xdate()
    canvas = FigureCanvasTkAgg(fig, master=right_panel)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky="nsew")

    def _get_pv_series_for_selection(pv_pred_dict, v1, v2, v3, luoyang_plant_to_devdns):
        """Return (forecast_timestamps_utc, values_1d) for pulldown selection or (None, None)."""
        if pv_pred_dict is None or not pv_pred_dict:
            return None, None
        if v1 != "洛阳电站":
            return None, None
        first_key = next(iter(pv_pred_dict))
        times = pv_pred_dict[first_key]["forecast_timestamps_utc"]
        if not times:
            return None, None
        if not v2 or not v3:
            if not v2:
                keys = list(pv_pred_dict.keys())
            else:
                devdns_plant = set(luoyang_plant_to_devdns.get(v2, []))
                keys = [
                    k for k in pv_pred_dict
                    if _devdn_display_name(k) in devdns_plant
                ]
            if not keys:
                return times, np.zeros(len(times), dtype=np.float64)
            preds = [
                np.squeeze(pv_pred_dict[k]["pv_pred"]).astype(np.float64)
                for k in keys
            ]
            values = np.sum(preds, axis=0)
            return times, values
        alg_key = "NE=" + str(v3).strip()
        if alg_key not in pv_pred_dict:
            return None, None
        values = np.squeeze(pv_pred_dict[alg_key]["pv_pred"]).astype(np.float64)
        return times, values

    def update_forecast_plots():
        v1, v2, v3 = var1.get(), var2.get(), var3.get()
        times, values = _get_pv_series_for_selection(
            latest_pv_pred[0], v1, v2, v3, _luoyang_plant_to_devdns
        )
        for ax, n_pts, hour_interval in [
            (ax_top, 16, 1),
            (ax_mid, 96, 2),
            (ax_bot, 192, 4),
        ]:
            ax.clear()
            ax.set_ylabel("PV power (kW)")
            ax.grid(True, alpha=0.3)
            if times is not None and values is not None and len(times) >= n_pts:
                t_plot = pd.to_datetime(times[:n_pts]) + pd.Timedelta(hours=8)
                v_plot = values[:n_pts]
                ax.plot(t_plot, v_plot, color="C0", lw=1.5)
                ax.scatter(t_plot, v_plot, color="C0", s=10, zorder=3)
                ax.set_xlim(t_plot.min(), t_plot.max())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
            else:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            ax.set_xlabel("Time (UTC+8)")
        ax_top.set_title("Short-term (4 h forecast)")
        ax_mid.set_title("Mid-term (24 h forecast)")
        ax_bot.set_title("Long-term (48 h forecast)")
        fig.tight_layout()
        canvas.draw_idle()

    def on_selection_changed():
        update_forecast_plots()

    var1 = tk.StringVar(value=LEVEL1_OPTIONS[0])
    _init_l2 = LEVEL2_OPTIONS["洛阳电站"][0]
    _init_l3 = LEVEL3_OPTIONS.get(_init_l2, [""])[0]
    var2 = tk.StringVar(value=_init_l2)
    var3 = tk.StringVar(value=_init_l3)
    menu2_ref = [None]
    menu3_ref = [None]

    def _rebuild_menu2(*args):
        val1 = var1.get()
        opts2 = LEVEL2_OPTIONS.get(val1, [])
        if not opts2:
            return
        var2.set(opts2[0])
        menu = tk.OptionMenu(dropdown_frame, var2, *opts2, command=lambda _: _rebuild_menu3())
        menu.config(width=6)
        if menu2_ref[0] is not None:
            menu2_ref[0].grid_forget()
        menu.grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        menu2_ref[0] = menu
        _rebuild_menu3()

    def _rebuild_menu3(*args):
        val2 = var2.get()
        opts3 = LEVEL3_OPTIONS.get(val2, [])
        if not opts3:
            return
        var3.set(opts3[0])
        menu = tk.OptionMenu(dropdown_frame, var3, *opts3, command=lambda _: on_selection_changed())
        menu.config(width=8)
        if menu3_ref[0] is not None:
            menu3_ref[0].grid_forget()
        menu.grid(row=0, column=2, padx=4, pady=2, sticky="ew")
        menu3_ref[0] = menu
        on_selection_changed()

    menu1 = tk.OptionMenu(dropdown_frame, var1, *LEVEL1_OPTIONS, command=lambda _: _rebuild_menu2())
    menu1.config(width=10)
    menu1.grid(row=0, column=0, padx=4, pady=2, sticky="ew")
    _rebuild_menu2()

    def update_left_panels(now_local: datetime):
        sky_photo_ref.clear()
        # Floor display time to whole minute (seconds=0) for label
        t_left = now_local.replace(second=0, microsecond=0)
        t_right = now_local + timedelta(minutes=15)
        sky_time_label_left.config(text=t_left.strftime("%H:%M:%S"))
        for path, lbl in [
            (_find_sky_image_for_time(SKY_IMAGE_DIR, t_left), sky_image_label_left),
            (_find_sky_image_for_time(SKY_IMAGE_PRED_DIR, t_right), sky_image_label_right),
        ]:
            if path is not None:
                ph = _load_sky_image(path, max_size=(128, 128))
                if ph is not None:
                    sky_photo_ref.append(ph)
                    lbl.config(image=ph, text="")
                else:
                    if blank_photo is not None:
                        sky_photo_ref.append(blank_photo)
                        lbl.config(image=blank_photo, text="")
                    else:
                        lbl.config(image="", text="No Image")
            else:
                if blank_photo is not None:
                    sky_photo_ref.append(blank_photo)
                    lbl.config(image=blank_photo, text="")
                else:
                    lbl.config(image="", text="No Image")

        # NWP forecast (UTC+8) for SSRD and wind speed: use latest CSVs,
        # pick the two most recent forecast times <= now_local.
        def _update_nwp_labels(series, time1_label, time2_label, val1_label, val2_label, fmt="{:.1f}"):
            time1_label.config(text="--:--:--")
            time2_label.config(text="--:--:--")
            val1_label.config(text="0")
            val2_label.config(text="0")
            if series is None or series.empty:
                return
            s = series[series.index <= now_local]
            if s.empty:
                return
            times = s.index.to_pydatetime()
            vals = s.values
            if len(times) == 1:
                t1 = times[0]
                v1 = vals[0]
                time1_label.config(text=t1.strftime("%H:%M:%S"))
                val1_label.config(text=fmt.format(v1))
            else:
                t1, t2 = times[-2], times[-1]
                v1, v2 = vals[-2], vals[-1]
                time1_label.config(text=t1.strftime("%H:%M:%S"))
                time2_label.config(text=t2.strftime("%H:%M:%S"))
                val1_label.config(text=fmt.format(v1))
                val2_label.config(text=fmt.format(v2))

        ssrd_series = _load_latest_nwp_series(NWP_SOLAR_DIR, "ssrd")
        wind_series = _load_latest_nwp_series(NWP_WIND_DIR, "t2m")
        _update_nwp_labels(ssrd_series, ssrd_time1_label, ssrd_time2_label, ssrd_val1_label, ssrd_val2_label)
        _update_nwp_labels(wind_series, wind_time1_label, wind_time2_label, wind_val1_label, wind_val2_label)

    def update_visuals():
        now_local = datetime.now(UTC8)
        update_clot_panel()
        update_left_panels(now_local.replace(tzinfo=None))
        root.after(60 * 1000, update_visuals)

    def update_time():
        now_utc = datetime.now(timezone.utc)
        now_local = datetime.now(UTC8)
        utc_var.set(now_utc.strftime("%Y-%m-%d %H:%M:%S"))
        local_var.set(f"{now_local.strftime('%Y-%m-%d %H:%M:%S')} UTC+8")
        # Keep sky time label aligned to whole minute of current time,
        # even though the image content itself only refreshes once per minute.
        t_left = now_local.replace(second=0, microsecond=0)
        sky_time_label_left.config(text=t_left.strftime("%H:%M:%S"))
        root.after(1000, update_time)

    root.after(0, update_visuals)
    root.after(1000, update_time)

    root.mainloop()


if __name__ == "__main__":
    create_infer_online_gui()

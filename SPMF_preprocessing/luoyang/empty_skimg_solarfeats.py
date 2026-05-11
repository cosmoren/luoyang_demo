from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pvlib import solarposition


def utc_to_local_solar_time_pvlib(utc_times: pd.DatetimeIndex, longitude: float) -> pd.DatetimeIndex:
    """Convert UTC timestamps to local apparent solar time with pvlib equation-of-time."""
    if utc_times.tz is not None:
        utc_naive = utc_times.tz_convert("UTC").tz_localize(None)
    else:
        utc_naive = utc_times

    lmst_offset_hours = longitude / 15.0
    day_of_year = utc_naive.dayofyear
    eot_minutes = solarposition.equation_of_time_spencer71(day_of_year)
    return utc_naive + pd.Timedelta(hours=lmst_offset_hours) + pd.to_timedelta(eot_minutes, unit="m")


t0 = datetime(2024, 12, 30, 0, 0, 0, tzinfo=timezone.utc)
tf = datetime(2026, 1, 2, 0, 30, 0, tzinfo=timezone.utc)
freq = "1min"
lat = 34.68
lon = 112.45
out_store = Path("/work/datasets/luoyang_SPMF/skimg_zarr")

ts_index_utc = pd.date_range(start=t0, end=tf, freq=freq)
chunk_size = 30
out_store.parent.mkdir(parents=True, exist_ok=True)

for start in range(0, len(ts_index_utc), chunk_size):
    print(start, "/", len(ts_index_utc))
    stop = min(start + chunk_size, len(ts_index_utc))
    ts_chunk = ts_index_utc[start:stop]
    ts_datetime_utc = ts_chunk.to_numpy(dtype="datetime64[ns]")

    solpos = solarposition.get_solarposition(ts_chunk, latitude=lat, longitude=lon)
    zenith_arr = solpos["zenith"].to_numpy(dtype=np.float32)
    azimuth_arr = solpos["azimuth"].to_numpy(dtype=np.float32)

    local_solar_time = utc_to_local_solar_time_pvlib(ts_chunk, lon)
    # Store local solar time as unix ns int64 to avoid CF datetime decode issues on append.
    local_solar_time_arr = local_solar_time.asi8.astype(np.int64)
    day_of_year_arr = local_solar_time.dayofyear.to_numpy(dtype=np.int32)
    hour_of_day_arr = (
        local_solar_time.hour
        + local_solar_time.minute / 60.0
        + local_solar_time.second / 3600.0
    ).to_numpy(dtype=np.float32)

    image_tensor = np.zeros((len(ts_chunk), 4, 224, 224), dtype=np.float32)

    ds_chunk = xr.Dataset(
        data_vars={
            "images": (("time_utc", "channel", "H", "W"), image_tensor),
            "zenith": (("time_utc",), zenith_arr),
            "azimuth": (("time_utc",), azimuth_arr),
            "day_of_year": (("time_utc",), day_of_year_arr),
            "hour_of_day": (("time_utc",), hour_of_day_arr),
            "local_solar_time": (("time_utc",), local_solar_time_arr),
        },
        coords={
            "time_utc": ts_datetime_utc,
            "channel": ["R", "G", "B", "mask"],
        },
    )
    ds_chunk["time_utc"].attrs["timezone"] = "UTC+0"
    ds_chunk["local_solar_time"].attrs["unit"] = "unix_ns"
    ds_chunk.attrs["latitude"] = lat
    ds_chunk.attrs["longitude"] = lon

    if start == 0:
        ds_chunk.to_zarr(str(out_store), mode="w")
    else:
        ds_chunk.to_zarr(str(out_store), mode="a", append_dim="time_utc")
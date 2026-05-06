import xarray as xr
import numpy as np
from scipy.ndimage import zoom
from datetime import datetime, timezone
from pathlib import Path


def _open_dataset_robust(path: str) -> xr.Dataset:
    """
    Open NetCDF once using a fixed backend for speed.
    """
    try:
        return xr.open_dataset(path, decode_timedelta=True, engine="h5netcdf")
    except Exception as e:
        raise OSError(f"NetCDF open failed for {path} (engine=h5netcdf): {type(e).__name__}: {e}") from e

def nc_crop_mask(path: str, lat0: float, lon0: float, size_km: float = 500, resample_size: int = 100):
    # Bounds from center (lat0, lon0) and size_km (approx degrees: 1 deg lat ~ 111 km)
    dlat = size_km / 111.0 / 2
    dlon = size_km / (111.0 * np.cos(np.deg2rad(lat0))) / 2
    lat_min, lat_max = lat0 - dlat, lat0 + dlat
    lon_min, lon_max = lon0 - dlon, lon0 + dlon

    ds = _open_dataset_robust(path)
    try:
        ds_id = ds.attrs.get("id")
        if not ds_id:
            ds_id = Path(path).name
        yyyymmdd = ds_id.split("_")[2]

        roi = ds.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )

        timedelta = roi["Hour"]

        timedelta_hours = timedelta / np.timedelta64(1, "h")
        mean_timedelta_hours = float(np.nanmean(timedelta_hours.values))
        total_seconds = int(round(mean_timedelta_hours * 3600))
        mean_hour = total_seconds // 3600
        mean_min = (total_seconds % 3600) // 60
        mean_sec = total_seconds % 60
        hhmmss = f"{mean_hour:02d}{mean_min:02d}{mean_sec:02d}"
        ts_string = yyyymmdd + hhmmss
        ts_datetime = datetime.strptime(ts_string, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

        clot = np.asarray(roi["CLOT"])
        clot_mask = (clot >= 0)
        np.clip(clot, 0, 150, out=clot)
        clot = clot/150.0
        clot = np.squeeze(clot).astype(np.float32)
        clot_mask = np.squeeze(clot_mask).astype(np.float32)
        merged = np.stack([clot, clot_mask], axis=-1)
    finally:
        ds.close()

    h, w = merged.shape[0], merged.shape[1]
    zoom_factors = (resample_size / h, resample_size / w, 1)
    merged = zoom(merged, zoom_factors, order=0)
    return merged, ts_datetime, ts_string
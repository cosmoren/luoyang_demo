import xarray as xr
import numpy as np
from scipy.ndimage import zoom

def nc_crop_mask(path: str, lat0: float, lon0: float, size_km: float = 500, resample_size: int = 100):
    # Bounds from center (lat0, lon0) and size_km (approx degrees: 1 deg lat ~ 111 km)
    dlat = size_km / 111.0 / 2
    dlon = size_km / (111.0 * np.cos(np.deg2rad(lat0))) / 2
    lat_min, lat_max = lat0 - dlat, lat0 + dlat
    lon_min, lon_max = lon0 - dlon, lon0 + dlon

    ds = xr.open_dataset(path, decode_timedelta=True)
    roi = ds.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )
    clot = np.asarray(roi["CLOT"])
    clot_mask = (clot >= 0)
    np.clip(clot, 0, 150, out=clot)
    clot = clot/150.0
    clot = np.squeeze(clot).astype(np.float32)
    clot_mask = np.squeeze(clot_mask).astype(np.float32)
    ch3 = np.ones_like(clot)
    merged = np.stack([clot, clot_mask, ch3], axis=-1)
    h, w = merged.shape[0], merged.shape[1]
    zoom_factors = (resample_size / h, resample_size / w, 1)
    merged = zoom(merged, zoom_factors, order=0)
    return merged
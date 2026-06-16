"""Per-frame sun_mask geometry for calibrated all-sky fisheye cameras.

Pure-numpy utilities that consume the (cx, cy, f, alpha0) parameters fitted by
:mod:`SPMF_preprocessing.fisheye_calib.fisheye_model` and produce a binary sun
mask in the runtime sky-tensor pixel grid.

The fit is performed in the *horizontally flipped* image space (Folsom's raw
JPGs have image-right = West, but the calibration tooling first applies an
``u_flipped = (W-1) - u_raw`` flip so that image-right = East before fitting).
Code consuming these params therefore has to undo the same flip when emitting
raw-image pixel coordinates -- both for the sun-pixel projection and for the
per-pixel ray-direction map built underneath :func:`compute_sun_mask`.

Conventions
-----------
* ``az_deg`` -- meteorological azimuth, degrees: 0 = North, 90 = East,
  increasing clockwise.
* ``zen_deg`` -- zenith angle from up, degrees. ``zen_deg >= 90`` means the
  sun is below the horizon and the mask is forced to zero.
* ``alpha0`` -- camera yaw offset (radians) relative to North, in the SAME
  sign convention used by :func:`fisheye_model.predict_simple`.
* ``f`` -- equidistant focal length in pixels-per-radian at the fit's native
  image size (typically the camera's native JPG width, e.g. 1536 for Folsom).
* ``image_size`` -- runtime square sky-tensor size (e.g. 224); the fit
  parameters are scaled by ``image_size / fit["native_size"]``.

Masking method
--------------
:func:`compute_sun_mask` thresholds the **angular** (great-circle) distance
between the sun direction and each pixel's calibrated ray direction. This
matches the playground calibration reference strip exactly (and the previous
dataloader behavior up to the calibration upgrade). The Euclidean
sun-pixel-distance approximation diverges from it at high zenith because
equidistant pixel distance only equals ``f * angular_distance`` along the
radial axis; it underestimates the angular disk along the azimuthal axis when
the sun is far from the optical axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent


def load_fisheye_fit(camera_name: str, native_size: int = 1536) -> dict:
    """Load fitted ``(cx, cy, f, alpha0)`` from the per-camera fit CSV.

    Reads ``SPMF_preprocessing/fisheye_calib/{camera_name}_fisheye_fit.csv``
    and returns a flat dict with the four scalar params plus the ``native_size``
    they were fitted at. The CSV may contain additional diagnostic columns
    (residuals, sample counts, ...); those are ignored.
    """
    csv_path = _THIS_DIR / f"{camera_name}_fisheye_fit.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"fisheye_sunmask.load_fisheye_fit: missing fit CSV {csv_path!s}"
        )
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"fisheye_sunmask.load_fisheye_fit: empty CSV {csv_path!s}")
    row = df.iloc[0]
    required = ("cx", "cy", "f", "alpha0")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"fisheye_sunmask.load_fisheye_fit: {csv_path.name} missing required "
            f"columns {missing} (have {list(df.columns)})"
        )
    return {
        "cx": float(row["cx"]),
        "cy": float(row["cy"]),
        "f": float(row["f"]),
        "alpha0": float(row["alpha0"]),
        "native_size": int(native_size),
    }


def _scale_fit_to_image(fit: dict, image_size: int) -> tuple[float, float, float, float]:
    """Scale ``(cx, cy, f)`` from ``fit['native_size']`` to ``image_size``; alpha0 unchanged."""
    native = int(fit["native_size"])
    if native <= 0:
        raise ValueError(f"fisheye_sunmask: native_size must be > 0 (got {native})")
    if image_size <= 0:
        raise ValueError(f"fisheye_sunmask: image_size must be > 0 (got {image_size})")
    scale = float(image_size) / float(native)
    return (
        float(fit["cx"]) * scale,
        float(fit["cy"]) * scale,
        float(fit["f"]) * scale,
        float(fit["alpha0"]),
    )


def project_sun_to_pixel(
    az_deg: np.ndarray,
    zen_deg: np.ndarray,
    image_size: int,
    fit: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Project per-frame ``(az_deg, zen_deg)`` onto raw-image pixel coords.

    Inverts the same horizontal flip the calibration applied so the returned
    ``(u_raw, v_raw)`` lives in the raw image's coordinate system (image-right
    = West for Folsom).

    Returns
    -------
    u_raw, v_raw : np.ndarray ``[T]`` float64
        Pixel coords; NaN where the sun is below the horizon.
    above_horizon : np.ndarray ``[T]`` bool
        True where the sun is above the horizon and inputs are finite.
    (cx_s, cy_s, f_s) : tuple of floats
        The fit scaled to ``image_size``, exposed so callers can do bounds
        checks without re-scaling.
    """
    az = np.asarray(az_deg, dtype=np.float64)
    zen = np.asarray(zen_deg, dtype=np.float64)
    if az.shape != zen.shape:
        raise ValueError(
            f"fisheye_sunmask.project_sun_to_pixel: az/zen shape mismatch "
            f"{az.shape} vs {zen.shape}"
        )
    cx_s, cy_s, f_s, alpha0 = _scale_fit_to_image(fit, image_size)

    above = np.isfinite(az) & np.isfinite(zen) & (zen < 90.0)
    az_r = np.deg2rad(az)
    zen_r = np.deg2rad(zen)

    r_pix = f_s * zen_r
    u_flipped = cx_s + r_pix * np.sin(az_r - alpha0)
    v_raw = cy_s - r_pix * np.cos(az_r - alpha0)
    u_raw = (float(image_size) - 1.0) - u_flipped

    u_raw = np.where(above, u_raw, np.nan)
    v_raw = np.where(above, v_raw, np.nan)
    return u_raw, v_raw, above, (cx_s, cy_s, f_s)


def _build_pixel_directions(
    image_size: int, cx_s: float, cy_s: float, f_s: float, alpha0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel calibrated ray directions for a square raw-image grid.

    Returns ``(rx, ry, rz, valid)``, each shape ``[image_size, image_size]``
    ``float64`` (``valid`` is ``bool``). ``(rx, ry, rz)`` are unit-length
    vectors in the ``(East, North, Up)`` frame; ``valid`` flags pixels whose
    implied zenith is ``<= 90 deg`` (i.e. inside the fisheye image circle).

    Mirrors the calibration reference (``playground/2026-06-16_folsom-lens-calib
    /render_strip.py::build_pixel_directions``) so the runtime mask is
    geometrically identical to the calibration tooling.
    """
    s = int(image_size)
    ys, xs = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")

    u_flip = (s - 1) - xs.astype(np.float64)
    dx = u_flip - float(cx_s)
    dy = ys.astype(np.float64) - float(cy_s)

    r = np.sqrt(dx * dx + dy * dy)
    zen = r / float(f_s)
    az = np.arctan2(dx, -dy) + float(alpha0)

    sin_z = np.sin(zen)
    rx = sin_z * np.sin(az)
    ry = sin_z * np.cos(az)
    rz = np.cos(zen)

    valid = zen <= (np.pi / 2.0)
    return rx, ry, rz, valid


def compute_sun_mask(
    az_deg: np.ndarray,
    zen_deg: np.ndarray,
    image_size: int,
    radius_deg: float,
    fit: dict,
) -> np.ndarray:
    """Per-frame ``[T, image_size, image_size]`` ``float32`` binary sun mask.

    For each frame, the sun's unit direction (from ``az_deg, zen_deg``) is
    compared to every pixel's calibrated ray direction; pixels whose
    great-circle angle to the sun is within ``radius_deg`` are marked ``1``.
    Pixels outside the fisheye image circle and frames with the sun below the
    horizon are forced to ``0``.

    The lens math (``cx, cy, f, alpha0`` + the calibration's horizontal flip)
    comes from ``fit`` (see :func:`load_fisheye_fit`); the runtime grid is
    ``image_size`` square and the fit is scaled from ``fit['native_size']``.
    """
    az = np.asarray(az_deg, dtype=np.float64)
    zen = np.asarray(zen_deg, dtype=np.float64)
    if az.ndim != 1 or zen.ndim != 1:
        raise ValueError(
            f"fisheye_sunmask.compute_sun_mask: az_deg/zen_deg must be 1-D, "
            f"got shapes {az.shape}, {zen.shape}"
        )
    if az.shape[0] != zen.shape[0]:
        raise ValueError(
            f"fisheye_sunmask.compute_sun_mask: az/zen length mismatch "
            f"{az.shape[0]} vs {zen.shape[0]}"
        )
    if image_size <= 0:
        raise ValueError(
            f"fisheye_sunmask.compute_sun_mask: image_size must be > 0 "
            f"(got {image_size})"
        )
    if not (radius_deg > 0.0):
        raise ValueError(
            f"fisheye_sunmask.compute_sun_mask: radius_deg must be > 0 "
            f"(got {radius_deg})"
        )

    t = int(az.shape[0])
    s = int(image_size)
    cx_s, cy_s, f_s, alpha0 = _scale_fit_to_image(fit, image_size)

    rx, ry, rz, valid = _build_pixel_directions(s, cx_s, cy_s, f_s, alpha0)

    above = np.isfinite(az) & np.isfinite(zen) & (zen < 90.0)
    az_r = np.deg2rad(az)
    zen_r = np.deg2rad(zen)
    sin_z = np.sin(zen_r)
    sun_x = sin_z * np.sin(az_r)
    sun_y = sin_z * np.cos(az_r)
    sun_z = np.cos(zen_r)

    cos_threshold = float(np.cos(np.deg2rad(radius_deg)))

    out = np.zeros((t, s, s), dtype=np.float32)
    for i in range(t):
        if not bool(above[i]):
            continue
        cos_a = rx * sun_x[i] + ry * sun_y[i] + rz * sun_z[i]
        out[i] = ((cos_a >= cos_threshold) & valid).astype(np.float32)
    return out

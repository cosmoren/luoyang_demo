"""Per-frame sun_mask geometry for calibrated all-sky fisheye cameras.

Pure-numpy utilities that consume the (cx, cy, f, alpha0) parameters fitted by
:mod:`SPMF_preprocessing.fisheye_calib.fisheye_model` and produce a binary sun
mask in the runtime sky-tensor pixel grid.

The fit is performed in the *horizontally flipped* image space at native
resolution (1536 for Folsom). Folsom's raw JPGs have image-right = West, but the
calibration tooling first applies ``u_flipped = (W-1) - u_raw`` so that
image-right = East before fitting. Runtime training images stay **raw and
unflipped** (resized to e.g. 224×224); code consuming these params projects in
flip space then mirrors ``u`` back via ``u_raw = (native-1) - u_flipped`` for
both the sun-pixel location and the per-pixel Euclidean sun-mask disc.

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
* ``image_size`` -- runtime square sky-tensor size (e.g. 224); fit params and
  projected coords are remapped with ``(image_size - 1) / (native_size - 1)`` so
  they align with a uniformly resized JPG (the ``(W-1) - u`` flip does not
  commute with ``image_size / native_size`` scaling).

Masking method
--------------
:func:`compute_sun_mask` projects the sun onto the image via
:func:`project_sun_to_pixel`, then marks pixels inside a **Euclidean pixel
circle** of radius ``R = f_s * beta`` where ``beta = deg2rad(radius_deg)``.
Frames with the sun below the horizon are all-zero.

Gaussian modes (``gaussian_pixel``, ``gaussian_angular``) produce soft masks in
``[0, 1]`` clipped after the Gaussian. If the projected sun pixel falls outside
``[0, W-1] x [0, H-1]``, the entire frame is zero (even when above the horizon).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent

DEFAULT_SUN_MASK_SIGMA_PX: float = 15.0
DEFAULT_SUN_MASK_SIGMA_DEG: float = 10.0

_SUN_MASK_HARD_MODES: frozenset[str] = frozenset({"sun_only", "sun_halo"})
_SUN_MASK_GAUSSIAN_MODES: frozenset[str] = frozenset(
    {"gaussian_pixel", "gaussian_angular"}
)


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
    """Scale ``(cx, cy, f)`` from ``fit['native_size']`` to ``image_size``; alpha0 unchanged.

    Uses ``(image_size - 1) / (native_size - 1)`` so pixel coords stay consistent with
    resizing a native JPG via uniform scale (same convention as ``(W-1) - u`` flip).
    """
    native = int(fit["native_size"])
    if native <= 0:
        raise ValueError(f"fisheye_sunmask: native_size must be > 0 (got {native})")
    if image_size <= 0:
        raise ValueError(f"fisheye_sunmask: image_size must be > 0 (got {image_size})")
    scale = (float(image_size) - 1.0) / (float(native) - 1.0)
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

    # Project in native fit space, then remap with (W-1)/(N-1) so coords match a
    # uniformly resized JPG (the (W-1) flip does not commute with W/N scaling).
    native = int(fit["native_size"])
    cx_n = float(fit["cx"])
    cy_n = float(fit["cy"])
    f_n = float(fit["f"])
    r_pix_n = f_n * zen_r
    u_flipped = cx_n + r_pix_n * np.sin(az_r - alpha0)
    v_native = cy_n - r_pix_n * np.cos(az_r - alpha0)
    u_native = (float(native) - 1.0) - u_flipped

    if image_size == native:
        u_raw = u_native
        v_raw = v_native
    else:
        scale = (float(image_size) - 1.0) / (float(native) - 1.0)
        u_raw = u_native * scale
        v_raw = v_native * scale

    u_raw = np.where(above, u_raw, np.nan)
    v_raw = np.where(above, v_raw, np.nan)
    return u_raw, v_raw, above, (cx_s, cy_s, f_s)


def _scaled_optical_center_and_f(
    fit: dict, H: int, W: int
) -> tuple[float, float, float]:
    """Optical center ``(cx, cy)`` and ``f`` scaled to raw-image ``(H, W)``.

    Mirrors ``cx`` from flipped calibration space to raw coords, matching
    :func:`SPMF_preprocessing.fisheye_calib.fisheye_raymap.compute_ray_map`.
    """
    native = int(fit["native_size"])
    s = min(int(H), int(W))
    scale = (float(s) - 1.0) / (float(native) - 1.0)
    cx_raw = (float(native) - 1.0) - float(fit["cx"])
    cx_s = cx_raw * scale
    cy_s = float(fit["cy"]) * scale
    f_s = float(fit["f"]) * scale
    if not (f_s > 0.0):
        raise ValueError(
            f"fisheye_sunmask: scaled focal length must be > 0 (got {f_s})"
        )
    return cx_s, cy_s, f_s


def _sun_pixel_in_image_bounds(u: float, v: float, W: int, H: int) -> bool:
    return 0.0 <= float(u) <= float(W - 1) and 0.0 <= float(v) <= float(H - 1)


def _compute_image_axis_rays(H: int, W: int, fit: dict) -> np.ndarray:
    """Per-pixel unit rays ``[3, H, W]`` in the image-axis frame (see fisheye_raymap)."""
    cx_s, cy_s, f_s = _scaled_optical_center_and_f(fit, H, W)
    ys, xs = np.meshgrid(np.arange(int(H)), np.arange(int(W)), indexing="ij")
    dx = xs.astype(np.float64) - cx_s
    dy = ys.astype(np.float64) - cy_s
    r = np.sqrt(dx * dx + dy * dy)
    theta = r / f_s
    phi = np.arctan2(dy, dx)
    sin_t = np.sin(theta)
    ray_x = sin_t * np.cos(phi)
    ray_y = sin_t * np.sin(phi)
    ray_z = np.cos(theta)
    valid = r <= (f_s * np.pi / 2.0)
    ray_x = np.where(valid, ray_x, 0.0)
    ray_y = np.where(valid, ray_y, 0.0)
    ray_z = np.where(valid, ray_z, 0.0)
    return np.stack([ray_x, ray_y, ray_z], axis=0).astype(np.float64)


def _sun_ray_image_axis(
    u_sun: float, v_sun: float, cx_s: float, cy_s: float, f_s: float
) -> np.ndarray:
    """Unit ray towards the sun in the image-axis frame."""
    dx = float(u_sun) - float(cx_s)
    dy = float(v_sun) - float(cy_s)
    r = float(np.hypot(dx, dy))
    theta = r / float(f_s)
    phi = float(np.arctan2(dy, dx))
    sin_t = float(np.sin(theta))
    ray = np.array(
        [sin_t * np.cos(phi), sin_t * np.sin(phi), float(np.cos(theta))],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(ray))
    if norm > 0.0:
        ray /= norm
    return ray


def compute_sun_mask_gaussian_pixel(
    az_deg: np.ndarray,
    zen_deg: np.ndarray,
    image_size: int,
    fit: dict,
    sigma_px: float = DEFAULT_SUN_MASK_SIGMA_PX,
) -> np.ndarray:
    """Per-frame ``[T, H, W]`` soft sun mask via a pixel-space Gaussian."""
    az = np.asarray(az_deg, dtype=np.float64)
    zen = np.asarray(zen_deg, dtype=np.float64)
    _validate_sun_mask_inputs(az, zen, image_size)
    if not (sigma_px > 0.0):
        raise ValueError(
            f"fisheye_sunmask.compute_sun_mask_gaussian_pixel: sigma_px must be > 0 "
            f"(got {sigma_px})"
        )

    t = int(az.shape[0])
    s = int(image_size)
    u_sun, v_sun, above, _ = project_sun_to_pixel(az, zen, s, fit)

    u_grid, v_grid = np.meshgrid(
        np.arange(s, dtype=np.float64),
        np.arange(s, dtype=np.float64),
        indexing="xy",
    )
    inv_two_sigma_sq = 1.0 / (2.0 * float(sigma_px) * float(sigma_px))

    out = np.zeros((t, s, s), dtype=np.float32)
    for i in range(t):
        if not bool(above[i]):
            continue
        us = float(u_sun[i])
        vs = float(v_sun[i])
        if not _sun_pixel_in_image_bounds(us, vs, s, s):
            continue
        du = u_grid - us
        dv = v_grid - vs
        out[i] = np.exp(-(du * du + dv * dv) * inv_two_sigma_sq).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def compute_sun_mask_gaussian_angular(
    az_deg: np.ndarray,
    zen_deg: np.ndarray,
    image_size: int,
    fit: dict,
    sigma_deg: float = DEFAULT_SUN_MASK_SIGMA_DEG,
) -> np.ndarray:
    """Per-frame ``[T, H, W]`` soft sun mask via an angular Gaussian in ray space."""
    az = np.asarray(az_deg, dtype=np.float64)
    zen = np.asarray(zen_deg, dtype=np.float64)
    _validate_sun_mask_inputs(az, zen, image_size)
    if not (sigma_deg > 0.0):
        raise ValueError(
            f"fisheye_sunmask.compute_sun_mask_gaussian_angular: sigma_deg must be > 0 "
            f"(got {sigma_deg})"
        )

    t = int(az.shape[0])
    s = int(image_size)
    u_sun, v_sun, above, _ = project_sun_to_pixel(az, zen, s, fit)
    cx_s, cy_s, f_s = _scaled_optical_center_and_f(fit, s, s)
    rays = _compute_image_axis_rays(s, s, fit)
    sigma_rad = float(np.deg2rad(sigma_deg))
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_rad * sigma_rad)

    out = np.zeros((t, s, s), dtype=np.float32)
    for i in range(t):
        if not bool(above[i]):
            continue
        us = float(u_sun[i])
        vs = float(v_sun[i])
        if not _sun_pixel_in_image_bounds(us, vs, s, s):
            continue
        sun_ray = _sun_ray_image_axis(us, vs, cx_s, cy_s, f_s)
        dot = (
            rays[0] * sun_ray[0] + rays[1] * sun_ray[1] + rays[2] * sun_ray[2]
        )
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        out[i] = np.exp(-(angle_rad * angle_rad) * inv_two_sigma_sq).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _validate_sun_mask_inputs(
    az: np.ndarray, zen: np.ndarray, image_size: int
) -> None:
    if az.ndim != 1 or zen.ndim != 1:
        raise ValueError(
            f"fisheye_sunmask: az_deg/zen_deg must be 1-D, got shapes {az.shape}, {zen.shape}"
        )
    if az.shape[0] != zen.shape[0]:
        raise ValueError(
            f"fisheye_sunmask: az/zen length mismatch {az.shape[0]} vs {zen.shape[0]}"
        )
    if image_size <= 0:
        raise ValueError(
            f"fisheye_sunmask: image_size must be > 0 (got {image_size})"
        )


def compute_sun_mask(
    az_deg: np.ndarray,
    zen_deg: np.ndarray,
    image_size: int,
    radius_deg: float,
    fit: dict,
    *,
    mode: str = "sun_halo",
    sigma_px: float | None = None,
    sigma_deg: float | None = None,
) -> np.ndarray:
    """Per-frame ``[T, image_size, image_size]`` ``float32`` sun mask.

    For hard-disc modes (``sun_only`` / ``sun_halo``), each frame above the
    horizon and in-bounds:

    1. Project the sun to ``(u_sun, v_sun)`` via :func:`project_sun_to_pixel`.
    2. Set ``R = f_s * deg2rad(radius_deg)``.
    3. Mark ``mask[v, u] = 1`` when ``(u - u_sun)^2 + (v - v_sun)^2 <= R^2``.

    Frames with the sun below the horizon (``zen_deg >= 90``) are all-zero.
    Projected sun pixels outside ``[0, W-1] x [0, H-1]`` yield all-zero frames.
    The lens math (``cx, cy, f, alpha0`` + the calibration's horizontal flip)
    comes from ``fit`` (see :func:`load_fisheye_fit`); the runtime grid is
    ``image_size`` square and the fit is scaled from ``fit['native_size']``.

    ``mode`` selects the mask family:

    * ``sun_only`` / ``sun_halo`` — hard Euclidean disc (``radius_deg``).
    * ``gaussian_pixel`` — soft pixel Gaussian (``sigma_px``).
    * ``gaussian_angular`` — soft angular Gaussian in image-axis ray space
      (``sigma_deg``).
    """
    mode = str(mode).strip()
    if mode in _SUN_MASK_GAUSSIAN_MODES:
        if mode == "gaussian_pixel":
            if sigma_px is None:
                sigma_px = DEFAULT_SUN_MASK_SIGMA_PX
            return compute_sun_mask_gaussian_pixel(
                az_deg, zen_deg, image_size, fit, sigma_px=float(sigma_px)
            )
        if sigma_deg is None:
            sigma_deg = DEFAULT_SUN_MASK_SIGMA_DEG
        return compute_sun_mask_gaussian_angular(
            az_deg, zen_deg, image_size, fit, sigma_deg=float(sigma_deg)
        )
    if mode not in _SUN_MASK_HARD_MODES:
        raise ValueError(
            f"fisheye_sunmask.compute_sun_mask: unknown mode {mode!r}; "
            f"expected one of {sorted(_SUN_MASK_HARD_MODES | _SUN_MASK_GAUSSIAN_MODES)}"
        )

    az = np.asarray(az_deg, dtype=np.float64)
    zen = np.asarray(zen_deg, dtype=np.float64)
    _validate_sun_mask_inputs(az, zen, image_size)
    if not (radius_deg > 0.0):
        raise ValueError(
            f"fisheye_sunmask.compute_sun_mask: radius_deg must be > 0 "
            f"(got {radius_deg})"
        )

    t = int(az.shape[0])
    s = int(image_size)

    u_sun, v_sun, above, (_cx_s, _cy_s, f_s) = project_sun_to_pixel(
        az, zen, s, fit
    )
    radius_px = float(f_s) * float(np.deg2rad(radius_deg))
    radius_px_sq = radius_px * radius_px

    u_grid, v_grid = np.meshgrid(
        np.arange(s, dtype=np.float64),
        np.arange(s, dtype=np.float64),
        indexing="xy",
    )

    out = np.zeros((t, s, s), dtype=np.float32)
    for i in range(t):
        if not bool(above[i]):
            continue
        us = float(u_sun[i])
        vs = float(v_sun[i])
        if not _sun_pixel_in_image_bounds(us, vs, s, s):
            continue
        du = u_grid - us
        dv = v_grid - vs
        out[i] = (du * du + dv * dv <= radius_px_sq).astype(np.float32)
    return out

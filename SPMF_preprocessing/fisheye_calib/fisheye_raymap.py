"""Calibration-driven per-pixel ray map for all-sky fisheye cameras.

Pure-numpy companion to :mod:`SPMF_preprocessing.fisheye_calib.fisheye_sunmask`
that produces a ``[3, H, W]`` unit-vector ray field plus a ``[1, H, W]``
validity mask, both derived from the optical center and focal length fitted by
:mod:`SPMF_preprocessing.fisheye_calib.fisheye_model`.

The ray field lives in the **image-axis frame**:

* ``ray[0]`` is the image ``+x`` (column) component,
* ``ray[1]`` is the image ``+y`` (row) component,
* ``ray[2]`` is the out-of-image (towards the lens entrance) component.

This frame is intentionally world-agnostic: the calibration's ``alpha0`` yaw
offset does *not* enter here -- it describes how to align the *world*
(East/North/Up) frame with the image, which is irrelevant when all you want is
"what direction does each pixel look in, in image-axis coordinates". The
horizontal flip *does* enter via mirroring ``cx`` from flipped calibration
space to raw image coords (see Conventions below). The image-axis ray field is
fully determined by the optical center ``(cx, cy)`` and focal length ``f``.

Conventions
-----------
* Runtime training images are **raw, unflipped** Folsom JPGs resized to ``(H, W)``
  (typically 224×224); geometry below maps those pixels back through the fit.
* Equidistant fisheye: pixel distance ``r`` from the optical center maps to
  zenith angle ``theta = r / f`` (radians).
* The fisheye image circle is the locus ``r <= f * pi/2`` (90° from optical
  axis); pixels outside it are flagged invalid and their rays are zeroed.
* ``fit`` carries ``cx``, ``cy``, ``f`` measured at ``fit['native_size']``
  (typically 1536 for Folsom) in the **horizontally flipped** calibration
  space (same convention as :mod:`fisheye_sunmask`). Runtime images are
  **raw** (unflipped), so ``cx`` is mirrored before scaling:
  ``cx_raw = (native_size - 1) - cx_fit``; ``cy`` is unchanged.
* Scaling to the runtime grid uses ``(min(H, W) - 1) / (native_size - 1)``,
  matching :func:`fisheye_sunmask.project_sun_to_pixel` and a uniformly
  resized JPG (the ``(W-1) - u`` flip does not commute with ``W / N`` scaling).
"""

from __future__ import annotations

import numpy as np

from SPMF_preprocessing.fisheye_calib.fisheye_sunmask import load_fisheye_fit  # noqa: F401  (re-exported for callers)


def compute_ray_map(
    H: int, W: int, fit: dict, native_size: int = 1536
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated equidistant fisheye ray_map.

    Returns (ray, valid):
      ray:   [3, H, W] float32 unit vectors per pixel, in IMAGE-AXIS frame
             (ray[0] = image +x, ray[1] = image +y, ray[2] = out-of-image).
             Outside the fisheye circle: zero.
      valid: [1, H, W] float32 in {0, 1}, 1 inside the calibrated fisheye
             circle (pixel distance from (cx, cy) <= f * pi/2).

    Uses fit['cx'], fit['cy'], fit['f'] (all stored at fit['native_size'] in
    flipped calibration space). For raw-image coords, ``cx`` is mirrored
    (``cx_raw = (native - 1) - cx_fit``); ``cy`` is unchanged. All three are
    scaled with ``(min(H, W) - 1) / (native - 1)`` to match
    :func:`fisheye_sunmask.project_sun_to_pixel`. ``fit['alpha0']`` and the
    world-frame yaw do NOT enter here -- only the optical center and focal
    length define the image-axis ray field.
    """
    if H <= 0 or W <= 0:
        raise ValueError(f"compute_ray_map: H, W must be > 0 (got {H}, {W})")
    for key in ("cx", "cy", "f"):
        if key not in fit:
            raise KeyError(f"compute_ray_map: fit dict missing required key {key!r}")

    native = int(fit.get("native_size", native_size))
    if native <= 0:
        raise ValueError(f"compute_ray_map: native_size must be > 0 (got {native})")

    s = min(int(H), int(W))
    scale = (float(s) - 1.0) / (float(native) - 1.0)
    cx_raw = (float(native) - 1.0) - float(fit["cx"])
    cx_s = cx_raw * scale
    cy_s = float(fit["cy"]) * scale
    f_s = float(fit["f"]) * scale
    if not (f_s > 0.0):
        raise ValueError(f"compute_ray_map: scaled focal length must be > 0 (got {f_s})")

    ys, xs = np.meshgrid(np.arange(int(H)), np.arange(int(W)), indexing="ij")
    dx = xs.astype(np.float64) - cx_s
    dy = ys.astype(np.float64) - cy_s

    r = np.sqrt(dx * dx + dy * dy)
    theta = r / f_s
    # phi = arctan2(dy, dx) so that (cos phi, sin phi) = (dx/r, dy/r). The
    # image-axis ray then has the same in-plane direction as the pixel offset
    # from the optical center, which is the only sensible choice for an
    # axis-aligned ray field (no world-frame rotations).
    phi = np.arctan2(dy, dx)

    sin_t = np.sin(theta)
    ray_x = sin_t * np.cos(phi)  # image +x
    ray_y = sin_t * np.sin(phi)  # image +y
    ray_z = np.cos(theta)        # out-of-image

    valid = (r <= (f_s * np.pi / 2.0)).astype(np.float32)
    ray_x = (ray_x * valid).astype(np.float32)
    ray_y = (ray_y * valid).astype(np.float32)
    ray_z = (ray_z * valid).astype(np.float32)

    ray = np.stack([ray_x, ray_y, ray_z], axis=0).astype(np.float32)
    valid = valid[None, :, :]  # [1, H, W]
    return ray, valid

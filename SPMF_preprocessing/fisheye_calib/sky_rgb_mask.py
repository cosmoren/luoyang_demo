"""RGB gating masks for Folsom sky training (keep disc, zero outside).

Separate from the optional ``sun_mask`` extra channel: this module zeros RGB
pixels outside a Euclidean disc while leaving ray_map / sun_mask channels
untouched (applied in the dataloader before :func:`_build_sky_channels`).

Modes (radii in px at 224×224 unless overridden):
  * ``none``       — no gating (byte-identical to legacy)
  * ``valid_disc`` — disc at mirrored optical center, radius 110
  * ``tight_disc`` — disc at optical center, radius 80
  * ``sun_halo``   — disc at projected sun per frame, radius 50
  * ``sun_only``   — disc at projected sun per frame, radius 30
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from SPMF_preprocessing.fisheye_calib.fisheye_sunmask import project_sun_to_pixel

SKY_RGB_MASK_MODES: tuple[str, ...] = (
    "none",
    "valid_disc",
    "tight_disc",
    "sun_halo",
    "sun_only",
)

_REFERENCE_SPATIAL_SIZE: int = 224

# Default Euclidean radii (px) at ``_REFERENCE_SPATIAL_SIZE``.
DEFAULT_SKY_RGB_MASK_RADIUS_PX_AT_224: dict[str, float] = {
    "valid_disc": 110.0,
    "tight_disc": 80.0,
    "sun_halo": 50.0,
    "sun_only": 30.0,
}

_OPTICAL_CENTER_MODES: frozenset[str] = frozenset({"valid_disc", "tight_disc"})
_SUN_CENTER_MODES: frozenset[str] = frozenset({"sun_halo", "sun_only"})


def normalize_sky_rgb_mask_mode(raw: Any) -> str:
    """Validate and canonicalize a ``sky_rgb_mask_mode`` config / CLI value."""
    if raw is None:
        return "none"
    mode = str(raw).strip()
    if mode not in SKY_RGB_MASK_MODES:
        raise ValueError(
            f"sky_rgb_mask_mode must be one of {list(SKY_RGB_MASK_MODES)}, got {raw!r}"
        )
    return mode


def optical_center_raw(fit: dict, h: int, w: int) -> tuple[float, float]:
    """Mirrored optical center ``(cx, cy)`` in raw-image pixel coords at ``(h, w)``."""
    native = int(fit["native_size"])
    s = min(int(h), int(w))
    scale = (float(s) - 1.0) / (float(native) - 1.0)
    cx_raw = (float(native) - 1.0) - float(fit["cx"])
    return cx_raw * scale, float(fit["cy"]) * scale


def resolve_sky_rgb_mask_radius_px(
    mode: str,
    h: int,
    w: int,
    *,
    radius_px_override: float | None = None,
    radii_px_at_224: dict[str, float] | None = None,
) -> float:
    """Resolve the Euclidean disc radius in pixels for ``mode`` at ``(h, w)``."""
    mode = normalize_sky_rgb_mask_mode(mode)
    if mode == "none":
        raise ValueError("resolve_sky_rgb_mask_radius_px: mode 'none' has no radius")
    if radius_px_override is not None:
        radius_224 = float(radius_px_override)
    else:
        table = radii_px_at_224 if radii_px_at_224 is not None else DEFAULT_SKY_RGB_MASK_RADIUS_PX_AT_224
        if mode not in table:
            raise ValueError(
                f"resolve_sky_rgb_mask_radius_px: no radius for mode {mode!r} "
                f"(table keys: {sorted(table)})"
            )
        radius_224 = float(table[mode])
    if not (radius_224 > 0.0):
        raise ValueError(
            f"resolve_sky_rgb_mask_radius_px: radius must be > 0 (got {radius_224})"
        )
    s = min(int(h), int(w))
    scale = (float(s) - 1.0) / (float(_REFERENCE_SPATIAL_SIZE) - 1.0)
    return radius_224 * scale


def _disc_mask_2d(
    h: int,
    w: int,
    cx: float,
    cy: float,
    radius_px: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[H, W]`` bool mask: True inside the Euclidean disc."""
    ys = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, w)
    du = xs - float(cx)
    dv = ys - float(cy)
    r2 = float(radius_px) * float(radius_px)
    return (du * du + dv * dv) <= r2


def apply_sky_rgb_mask(
    rgb_chw: torch.Tensor,
    mode: str,
    *,
    fit: dict,
    frame_timestamps: list | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    optical_center: tuple[float, float] | None = None,
    radius_px_override: float | None = None,
    radii_px_at_224: dict[str, float] | None = None,
    az_deg: np.ndarray | None = None,
    zen_deg: np.ndarray | None = None,
) -> torch.Tensor:
    """Gate RGB: keep pixels inside the mode disc, zero channels outside.

    Accepts ``[T, 3, H, W]`` or ``[3, H, W]`` float tensors. Returns a tensor
    with the same shape and dtype. Mode ``none`` returns ``rgb_chw`` unchanged.
    """
    mode = normalize_sky_rgb_mask_mode(mode)
    if mode == "none":
        return rgb_chw

    if rgb_chw.ndim == 3:
        return apply_sky_rgb_mask(
            rgb_chw.unsqueeze(0),
            mode,
            fit=fit,
            frame_timestamps=frame_timestamps,
            latitude=latitude,
            longitude=longitude,
            optical_center=optical_center,
            radius_px_override=radius_px_override,
            radii_px_at_224=radii_px_at_224,
            az_deg=az_deg,
            zen_deg=zen_deg,
        ).squeeze(0)

    if rgb_chw.ndim != 4 or rgb_chw.shape[1] != 3:
        raise ValueError(
            f"apply_sky_rgb_mask: expected [T, 3, H, W] or [3, H, W], got {tuple(rgb_chw.shape)}"
        )

    t_dim, _, h_dim, w_dim = rgb_chw.shape
    radius_px = resolve_sky_rgb_mask_radius_px(
        mode,
        h_dim,
        w_dim,
        radius_px_override=radius_px_override,
        radii_px_at_224=radii_px_at_224,
    )
    device = rgb_chw.device
    dtype = rgb_chw.dtype

    if mode in _OPTICAL_CENTER_MODES:
        if optical_center is None:
            cx, cy = optical_center_raw(fit, h_dim, w_dim)
        else:
            cx, cy = float(optical_center[0]), float(optical_center[1])
        keep = _disc_mask_2d(h_dim, w_dim, cx, cy, radius_px, device=device, dtype=dtype)
        mask = keep.view(1, 1, h_dim, w_dim).expand(t_dim, 3, h_dim, w_dim)
        return rgb_chw * mask

    if mode in _SUN_CENTER_MODES:
        if az_deg is None or zen_deg is None:
            if frame_timestamps is None:
                raise ValueError(
                    f"apply_sky_rgb_mask: mode {mode!r} requires frame_timestamps or az_deg/zen_deg"
                )
            if latitude is None or longitude is None:
                raise ValueError(
                    f"apply_sky_rgb_mask: mode {mode!r} requires latitude/longitude "
                    "when az_deg/zen_deg are not provided"
                )
            from modules.solar_encoder import compute_solar_features

            feats = compute_solar_features(frame_timestamps, latitude, longitude)
            az_deg = np.asarray(feats["azimuth"], dtype=np.float64)
            zen_deg = np.asarray(feats["zenith"], dtype=np.float64)

        az = np.asarray(az_deg, dtype=np.float64)
        zen = np.asarray(zen_deg, dtype=np.float64)
        if az.shape[0] != t_dim or zen.shape[0] != t_dim:
            raise ValueError(
                f"apply_sky_rgb_mask: az/zen length {az.shape[0]}/{zen.shape[0]} "
                f"does not match T={t_dim}"
            )

        image_size = min(h_dim, w_dim)
        u_sun, v_sun, above, _ = project_sun_to_pixel(az, zen, image_size, fit)
        out = torch.zeros_like(rgb_chw)
        r2 = float(radius_px) * float(radius_px)
        ys = torch.arange(h_dim, device=device, dtype=dtype).view(h_dim, 1)
        xs = torch.arange(w_dim, device=device, dtype=dtype).view(1, w_dim)
        for i in range(t_dim):
            if not bool(above[i]):
                continue
            cx_i = float(u_sun[i])
            cy_i = float(v_sun[i])
            if not (np.isfinite(cx_i) and np.isfinite(cy_i)):
                continue
            du = xs - cx_i
            dv = ys - cy_i
            keep = (du * du + dv * dv) <= r2
            out[i] = rgb_chw[i] * keep.unsqueeze(0)
        return out

    raise ValueError(f"apply_sky_rgb_mask: unhandled mode {mode!r}")

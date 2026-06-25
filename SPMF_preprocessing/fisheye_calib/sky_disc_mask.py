"""Sky-disc gating for Folsom sky training (keep disc, zero RGB outside).

Separate from the optional ``sun_mask`` extra channel: this module zeros RGB
pixels outside a Euclidean disc while leaving ray_map / sun_mask channels
untouched (applied in the dataloader before :func:`_build_sky_channels`).

Modes (radii in px at 224×224 unless overridden):
  * ``none``         — no gating (byte-identical to legacy)
  * ``valid_disc``   — disc at mirrored optical center, radius 110
  * ``tight_disc``   — disc at optical center, radius 80
  * ``sun_halo``     — disc at projected sun per frame, radius 50
  * ``sun_only``     — disc at projected sun per frame, radius 30
  * ``manual_loose`` — keep region inside hand-drawn loose red ring (224² mask)
  * ``manual_tight`` — keep region inside hand-drawn tight red ring (224² mask)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from SPMF_preprocessing.fisheye_calib.fisheye_sunmask import project_sun_to_pixel

SKY_DISC_MASK_MODES: tuple[str, ...] = (
    "none",
    "valid_disc",
    "tight_disc",
    "sun_halo",
    "sun_only",
    "manual_loose",
    "manual_tight",
)

_REFERENCE_SPATIAL_SIZE: int = 224

# Default Euclidean radii (px) at ``_REFERENCE_SPATIAL_SIZE``.
DEFAULT_SKY_DISC_MASK_RADIUS_PX_AT_224: dict[str, float] = {
    "valid_disc": 110.0,
    "tight_disc": 80.0,
    "sun_halo": 50.0,
    "sun_only": 30.0,
}

_OPTICAL_CENTER_MODES: frozenset[str] = frozenset({"valid_disc", "tight_disc"})
_SUN_CENTER_MODES: frozenset[str] = frozenset({"sun_halo", "sun_only"})
_MANUAL_MASK_MODES: frozenset[str] = frozenset({"manual_loose", "manual_tight"})

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANUAL_MASK_PLAYGROUND_DIR = _PROJECT_ROOT / "playground" / "2026-06-25_manual_sky_masks"
_MANUAL_ANNOTATION_FILES: dict[str, str] = {
    "manual_loose": "loose.png",
    "manual_tight": "tight.png",
}
_MANUAL_KEEP_MASK_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


def normalize_sky_disc_mask_mode(raw: Any) -> str:
    """Validate and canonicalize a ``sky_disc_mask_mode`` config / CLI value."""
    if raw is None:
        return "none"
    mode = str(raw).strip()
    if mode not in SKY_DISC_MASK_MODES:
        raise ValueError(
            f"sky_disc_mask_mode must be one of {list(SKY_DISC_MASK_MODES)}, got {raw!r}"
        )
    return mode


def _red_annotation_mask(rgb_hwc: np.ndarray) -> np.ndarray:
    """Red stroke pixels in hand-annotated fisheye masks (R>180, G<100, B<100)."""
    r = rgb_hwc[..., 0]
    g = rgb_hwc[..., 1]
    b = rgb_hwc[..., 2]
    return (r > 180) & (g < 100) & (b < 100)


def _black_corner_mask(rgb_hwc: np.ndarray) -> np.ndarray:
    """Square corners outside the fisheye disc."""
    return rgb_hwc.sum(axis=-1) < 30


def extract_keep_mask_from_annotation(rgb_hwc: np.ndarray) -> np.ndarray:
    """Build a ``[H, W]`` bool keep-mask from a red-ring annotation PNG.

    Keep region = connected sky component inside the inner edge of the red
    stroke ring (red pixels and black corners are removed).
    """
    from scipy import ndimage

    rgb = np.asarray(rgb_hwc)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"extract_keep_mask_from_annotation: expected [H, W, 3], got {rgb.shape}"
        )
    h, w = rgb.shape[:2]
    red = _red_annotation_mask(rgb)
    black = _black_corner_mask(rgb)
    traversable = ~(red | black)
    labeled, _ = ndimage.label(traversable)

    cy, cx = h // 2, w // 2
    seed_label = 0
    for radius in range(0, max(h, w)):
        found = False
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dy) != radius and abs(dx) != radius:
                    continue
                y, x = cy + dy, cx + dx
                if 0 <= y < h and 0 <= x < w and traversable[y, x]:
                    seed_label = int(labeled[y, x])
                    found = True
                    break
            if found:
                break
        if found:
            break

    if seed_label == 0:
        return np.zeros((h, w), dtype=bool)
    return (labeled == seed_label) & traversable


def _manual_mask_paths(mode: str) -> tuple[Path, Path]:
    """Return ``(annotation_png, keep_npy)`` paths for a manual mask mode."""
    if mode not in _MANUAL_MASK_MODES:
        raise ValueError(f"_manual_mask_paths: not a manual mode: {mode!r}")
    stem = mode  # manual_loose -> manual_loose_keep.npy
    annotation = _MANUAL_MASK_PLAYGROUND_DIR / "inputs" / _MANUAL_ANNOTATION_FILES[mode]
    keep_npy = _MANUAL_MASK_PLAYGROUND_DIR / "outputs" / f"{stem}_keep.npy"
    return annotation, keep_npy


def build_and_save_manual_keep_mask(mode: str) -> np.ndarray:
    """Extract keep-mask from annotation PNG and save as ``.npy``."""
    from PIL import Image

    mode = normalize_sky_disc_mask_mode(mode)
    if mode not in _MANUAL_MASK_MODES:
        raise ValueError(f"build_and_save_manual_keep_mask: not a manual mode: {mode!r}")

    annotation_path, keep_npy_path = _manual_mask_paths(mode)
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"build_and_save_manual_keep_mask: annotation not found: {annotation_path}"
        )

    rgb = np.array(Image.open(annotation_path).convert("RGB"), dtype=np.uint8)
    keep = extract_keep_mask_from_annotation(rgb)
    keep_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(keep_npy_path, keep.astype(np.float32))
    return keep


def load_manual_keep_mask(
    mode: str,
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load (or build) a ``[H, W]`` float keep-mask for ``manual_*`` modes."""
    mode = normalize_sky_disc_mask_mode(mode)
    if mode not in _MANUAL_MASK_MODES:
        raise ValueError(f"load_manual_keep_mask: not a manual mode: {mode!r}")

    cache_key = (mode, int(h), int(w))
    if cache_key in _MANUAL_KEEP_MASK_CACHE:
        return _MANUAL_KEEP_MASK_CACHE[cache_key]

    _, keep_npy_path = _manual_mask_paths(mode)
    if not keep_npy_path.is_file():
        build_and_save_manual_keep_mask(mode)

    keep_np = np.load(keep_npy_path)
    if keep_np.shape != (h, w):
        keep_t = torch.from_numpy(keep_np.astype(np.float32)).view(1, 1, *keep_np.shape)
        keep_t = F.interpolate(keep_t, size=(h, w), mode="nearest")
        keep_t = keep_t.view(h, w)
    else:
        keep_t = torch.from_numpy(keep_np.astype(np.float32))

    keep_t = keep_t.to(device=device, dtype=dtype)
    _MANUAL_KEEP_MASK_CACHE[cache_key] = keep_t
    return keep_t


def optical_center_raw(fit: dict, h: int, w: int) -> tuple[float, float]:
    """Mirrored optical center ``(cx, cy)`` in raw-image pixel coords at ``(h, w)``."""
    native = int(fit["native_size"])
    s = min(int(h), int(w))
    scale = (float(s) - 1.0) / (float(native) - 1.0)
    cx_raw = (float(native) - 1.0) - float(fit["cx"])
    return cx_raw * scale, float(fit["cy"]) * scale


def resolve_sky_disc_mask_radius_px(
    mode: str,
    h: int,
    w: int,
    *,
    radius_px_override: float | None = None,
    radii_px_at_224: dict[str, float] | None = None,
) -> float:
    """Resolve the Euclidean disc radius in pixels for ``mode`` at ``(h, w)``."""
    mode = normalize_sky_disc_mask_mode(mode)
    if mode == "none":
        raise ValueError("resolve_sky_disc_mask_radius_px: mode 'none' has no radius")
    if mode in _MANUAL_MASK_MODES:
        raise ValueError(
            f"resolve_sky_disc_mask_radius_px: mode {mode!r} uses a precomputed mask, not a radius"
        )
    if radius_px_override is not None:
        radius_224 = float(radius_px_override)
    else:
        table = radii_px_at_224 if radii_px_at_224 is not None else DEFAULT_SKY_DISC_MASK_RADIUS_PX_AT_224
        if mode not in table:
            raise ValueError(
                f"resolve_sky_disc_mask_radius_px: no radius for mode {mode!r} "
                f"(table keys: {sorted(table)})"
            )
        radius_224 = float(table[mode])
    if not (radius_224 > 0.0):
        raise ValueError(
            f"resolve_sky_disc_mask_radius_px: radius must be > 0 (got {radius_224})"
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


def apply_sky_disc_mask(
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
    mode = normalize_sky_disc_mask_mode(mode)
    if mode == "none":
        return rgb_chw

    if rgb_chw.ndim == 3:
        return apply_sky_disc_mask(
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
            f"apply_sky_disc_mask: expected [T, 3, H, W] or [3, H, W], got {tuple(rgb_chw.shape)}"
        )

    t_dim, _, h_dim, w_dim = rgb_chw.shape
    device = rgb_chw.device
    dtype = rgb_chw.dtype

    if mode in _MANUAL_MASK_MODES:
        keep = load_manual_keep_mask(mode, h_dim, w_dim, device=device, dtype=dtype)
        mask = keep.view(1, 1, h_dim, w_dim).expand(t_dim, 3, h_dim, w_dim)
        return rgb_chw * mask

    radius_px = resolve_sky_disc_mask_radius_px(
        mode,
        h_dim,
        w_dim,
        radius_px_override=radius_px_override,
        radii_px_at_224=radii_px_at_224,
    )

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
                    f"apply_sky_disc_mask: mode {mode!r} requires frame_timestamps or az_deg/zen_deg"
                )
            if latitude is None or longitude is None:
                raise ValueError(
                    f"apply_sky_disc_mask: mode {mode!r} requires latitude/longitude "
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
                f"apply_sky_disc_mask: az/zen length {az.shape[0]}/{zen.shape[0]} "
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

    raise ValueError(f"apply_sky_disc_mask: unhandled mode {mode!r}")

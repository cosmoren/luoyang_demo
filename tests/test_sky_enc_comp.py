from __future__ import annotations

import torch
import torch.nn.functional as F

from modules.SkyCompressor import SkyTwoStageCompressor
from modules.SkyEncoder import (
    SkyAlternatingIntraInterFrameAttention,
    SkyPatchSpatiotemporalEmbed,
    patchify_spatiotemporal_sky_images,
)


def _fmt_shape(x: torch.Tensor) -> str:
    return str(list(x.shape))


def run_sky_pipeline_case(
    *,
    case_name: str,
    batch_size: int,
    num_frames: int,
    input_h: int,
    input_w: int,
    embed_dim: int = 64,
    patch_size: int = 16,
    image_size: int = 112,
    num_sky_queries: int = 48,
) -> None:
    print(f"\n=== {case_name} ===")

    sky_patch_embed = SkyPatchSpatiotemporalEmbed(
        embed_dim=embed_dim, patch_size=patch_size, image_size=image_size
    )
    sky_alt_attn = SkyAlternatingIntraInterFrameAttention(
        embed_dim=embed_dim, num_heads=8, num_cycles=4, dropout=0.0
    )
    sky_two_stage_compressor = SkyTwoStageCompressor(
        dim=embed_dim,
        num_frames=num_frames,
        num_patches=(image_size // patch_size) ** 2,
        num_frame_queries=8,
        num_sky_queries=num_sky_queries,
        num_heads=8,
        use_patch_pos=True,
        use_frame_time=True,
        use_coarse_spatial=True,
        coarse_query_grid_hw=(6, 8),
    )

    skimg_tensor = torch.randn(batch_size, num_frames, 3, input_h, input_w)
    skimg_timefeats = torch.randn(batch_size, num_frames, 9)
    print(f"Stage 0 (raw input): {_fmt_shape(skimg_tensor)}")

    sky_hr = F.interpolate(
        skimg_tensor.reshape(batch_size * num_frames, 3, input_h, input_w),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).view(batch_size, num_frames, 3, image_size, image_size)
    print(f"Stage 1 (resize): {_fmt_shape(sky_hr)}")
    assert sky_hr.shape == (batch_size, num_frames, 3, image_size, image_size)
    assert torch.isfinite(sky_hr).all().item(), "resize produced non-finite values"

    sky_patch_tokens = patchify_spatiotemporal_sky_images(
        sky_hr, sky_patch_embed, timefeats=skimg_timefeats
    )
    print(f"Stage 2 (patchify): {_fmt_shape(sky_patch_tokens)}")
    expected_patches = (image_size // patch_size) ** 2
    assert sky_patch_tokens.shape == (batch_size, num_frames, expected_patches, embed_dim)
    assert torch.isfinite(sky_patch_tokens).all().item(), "patchify produced non-finite values"

    sky_patch_tokens = sky_alt_attn(sky_patch_tokens)
    print(f"Stage 3 (alt-attn): {_fmt_shape(sky_patch_tokens)}")
    assert sky_patch_tokens.shape == (batch_size, num_frames, expected_patches, embed_dim)
    assert torch.isfinite(sky_patch_tokens).all().item(), "alt-attn produced non-finite values"

    sky_compressed = sky_two_stage_compressor(sky_patch_tokens)
    print(f"Stage 4 (compress): {_fmt_shape(sky_compressed)}")
    assert sky_compressed.shape == (batch_size, num_sky_queries, embed_dim)
    assert torch.isfinite(sky_compressed).all().item(), "compressor produced non-finite values"

    print(f"{case_name}: PASS")


def main() -> None:
    run_sky_pipeline_case(
        case_name="Case A - base (B=2, T=24, H=W=64)",
        batch_size=2,
        num_frames=24,
        input_h=64,
        input_w=64,
    )
    run_sky_pipeline_case(
        case_name="Case B - resize stress (B=2, T=24, H=W=80)",
        batch_size=2,
        num_frames=24,
        input_h=80,
        input_w=80,
    )
    run_sky_pipeline_case(
        case_name="Case C - alternate batch/frames (B=1, T=12, H=W=96)",
        batch_size=1,
        num_frames=12,
        input_h=96,
        input_w=96,
    )
    print("\nAll sky pipeline scenarios passed.")


if __name__ == "__main__":
    main()
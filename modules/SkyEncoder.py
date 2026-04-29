"""Sky-image patch spatiotemporal embedding and alternating intra/inter-frame attention."""

from typing import Optional

import torch
import torch.nn as nn


class SkyPatchSpatiotemporalEmbed(nn.Module):
    """
    Patchify ``[B, T, 3, H, W]`` frames and add:
    - learned spatial position embedding per patch
    - temporal embedding from ``timefeats`` ``[B, T, 9]`` via MLP
    """

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, image_size: int = 112):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        self.timefeats_mlp = nn.Sequential(
            nn.Linear(9, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, timefeats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, 3, H, W]`` where ``H=W=image_size``.
            timefeats: ``[B, T, 9]`` time features aligned with frames.
        Returns:
            ``[B, T, P, D]`` where ``P=num_patches`` and ``D=embed_dim``.
        """
        bsz, num_frames, channels, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"expected H=W={self.image_size}, got {height}x{width}")
        if channels != 3:
            raise ValueError(f"expected 3 input channels, got {channels}")
        if timefeats is None:
            raise ValueError("timefeats is required with shape [B, T, 9]")

        x_bt = x.reshape(bsz * num_frames, channels, height, width)
        tokens = self.patch_embed(x_bt)  # [B*T, D, gh, gw]
        tokens = tokens.flatten(2).transpose(1, 2)  # [B*T, P, D]
        tokens = tokens.view(bsz, num_frames, self.num_patches, self.embed_dim)
        tokens = tokens + self.spatial_pos_embed

        time_tokens = self.timefeats_mlp(timefeats.to(dtype=tokens.dtype)).unsqueeze(2)  # [B,T,1,D]
        tokens = tokens + time_tokens
        return tokens


def patchify_spatiotemporal_sky_images(
    x: torch.Tensor,
    embedder: SkyPatchSpatiotemporalEmbed,
    timefeats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Patchify sky-image frames and add spatial/temporal embeddings."""
    return embedder(x, timefeats=timefeats)


class _SkyTokenTransformerBlock(nn.Module):
    """Pre-LN self-attention + MLP on token sequence ``[N, L, D]``."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SkyAlternatingIntraInterFrameAttention(nn.Module):
    """
    Refines ``[B, T, P, D]`` by alternating:
    - intra-frame attention over patches ``P``
    - inter-frame attention over timesteps ``T``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_cycles: int = 4,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cycles = num_cycles
        self.intra_blocks = nn.ModuleList(
            _SkyTokenTransformerBlock(embed_dim, num_heads, dropout, mlp_ratio)
            for _ in range(num_cycles)
        )
        self.inter_blocks = nn.ModuleList(
            _SkyTokenTransformerBlock(embed_dim, num_heads, dropout, mlp_ratio)
            for _ in range(num_cycles)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_frames, num_patches, dim = x.shape
        for intra_block, inter_block in zip(self.intra_blocks, self.inter_blocks):
            intra_tokens = x.reshape(bsz * num_frames, num_patches, dim)
            intra_tokens = intra_block(intra_tokens)
            x = intra_tokens.reshape(bsz, num_frames, num_patches, dim)

            inter_tokens = x.permute(0, 2, 1, 3).contiguous().reshape(bsz * num_patches, num_frames, dim)
            inter_tokens = inter_block(inter_tokens)
            x = inter_tokens.reshape(bsz, num_patches, num_frames, dim).permute(0, 2, 1, 3).contiguous()
        return x

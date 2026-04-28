"""Satellite / video patch spatiotemporal embedding and alternating intra–inter frame attention."""

import math
from typing import Optional

import torch
import torch.nn as nn

class VideoPatchSpatiotemporalEmbed(nn.Module):
    """
    Patchify ``[B, T, 3, H, W]`` (default ``H=W=112``, ``patch_size=16`` → 196 patches/frame),
    then add **learned** spatial position embedding per patch and temporal encoding
    (MLP on ``timefeats`` ``[B, T, 9]`` broadcast to patches, or sinusoidal if ``timefeats`` is None).
    """

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, image_size: int = 112):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")
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
            x: ``[B, T, 3, H, W]`` with ``H=W=self.image_size`` (typically 112).
            timefeats: optional ``[B, T, 9]``. MLP → ``[B, T, embed_dim]``, then ``unsqueeze(2)``
                → ``[B, T, 1, embed_dim]``, added (broadcast) to all patch positions.
            If ``None``, uses normalized frame indices and sinusoidal encoding (same as before).
        Returns:
            ``[B, T, num_patches, embed_dim]``.
        """
        B, T, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"expected H=W={self.image_size}, got {H}x{W}")
        if C != 3:
            raise ValueError(f"expected 3 input channels, got {C}")

        x_bt = x.reshape(B * T, C, H, W)
        tok = self.patch_embed(x_bt)  # [B*T, D, gh, gw]
        tok = tok.flatten(2).transpose(1, 2)  # [B*T, P, D]
        tok = tok.view(B, T, self.num_patches, self.embed_dim)
        tok = tok + self.spatial_pos_embed
        time_tok = self.timefeats_mlp(timefeats.to(dtype=tok.dtype)).unsqueeze(2)  # [B, T, 1, D]
        tok = tok + time_tok

        return tok


def patchify_spatiotemporal_images(
    x: torch.Tensor,
    embedder: VideoPatchSpatiotemporalEmbed,
    timefeats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Patchify video frames and add spatial + temporal encodings (see :class:`VideoPatchSpatiotemporalEmbed`).

    Args:
        x: ``[B, T, 3, 112, 112]`` (``112`` must match ``embedder.image_size``).
        embedder: holds Conv patch projection and spatial positional parameters.
        timefeats: optional ``[B, T, 9]``.

    Returns:
        ``[B, T, 196, embed_dim]`` when ``patch_size=16``, ``image_size=112``.
    """
    return embedder(x, timefeats=timefeats)


class _TokenTransformerBlock(nn.Module):
    """Pre-LN self-attention + MLP on a token sequence ``[N, L, D]``."""

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
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z2 = self.norm1(z)
        attn_out, _ = self.attn(z2, z2, z2, need_weights=False)
        z = z + attn_out
        z2 = self.norm2(z)
        z = z + self.mlp(z2)
        return z


class AlternatingIntraInterFrameAttention(nn.Module):
    """
    Refines patch tokens ``[B, T, P, D]`` by **alternating**:

    1. **Intra-frame** (spatial): self-attention over ``P`` patches, independently for each
       batch index and time step (sequence length ``P``).
    2. **Inter-frame** (temporal): self-attention over ``T`` frames, independently for each
       batch index and patch index (sequence length ``T``).

    Repeats for ``num_cycles`` (intra → inter each cycle), mixing spatial and temporal context.
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
            _TokenTransformerBlock(embed_dim, num_heads, dropout, mlp_ratio)
            for _ in range(num_cycles)
        )
        self.inter_blocks = nn.ModuleList(
            _TokenTransformerBlock(embed_dim, num_heads, dropout, mlp_ratio)
            for _ in range(num_cycles)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, P, D]`` patch tokens (e.g. from :class:`VideoPatchSpatiotemporalEmbed`).
        Returns:
            Same shape ``[B, T, P, D]``.
        """
        B, T, P, D = x.shape
        for intra, inter in zip(self.intra_blocks, self.inter_blocks):
            h = x.reshape(B * T, P, D)
            h = intra(h)
            x = h.reshape(B, T, P, D)

            h = x.permute(0, 2, 1, 3).contiguous().reshape(B * P, T, D)
            h = inter(h)
            x = h.reshape(B, P, T, D).permute(0, 2, 1, 3).contiguous()
        return x

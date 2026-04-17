"""Satellite / video patch spatiotemporal embedding and alternating intra–inter frame attention."""

import math
from typing import Optional

import torch
import torch.nn as nn


def _sinusoidal_encoding_1d(pos: torch.Tensor, dim: int, base: float = 10_000.0) -> torch.Tensor:
    """
    ``pos``: [B, T] continuous coordinates (e.g. normalized timestamps).
    Returns: [B, T, dim] with dim even (sin/cos pairs).
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal dim must be even, got {dim}")
    device = pos.device
    half = dim // 2
    pos_f = pos.float()
    freqs = torch.exp(
        -math.log(base) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    )
    ang = pos_f.unsqueeze(-1) * freqs.view(1, 1, -1)
    enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return enc.to(dtype=pos.dtype)


class VideoPatchSpatiotemporalEmbed(nn.Module):
    """
    Patchify ``[B, T, 3, H, W]`` (default ``H=W=224``, ``patch_size=16`` → 196 patches/frame),
    then add **learned** spatial position embedding per patch and **sinusoidal** temporal encoding
    (from optional non-uniform ``time_offsets`` or normalized frame indices).
    """

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, image_size: int = 224):
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

    def forward(self, x: torch.Tensor, time_offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, 3, H, W]`` with ``H=W=self.image_size`` (typically 224).
            time_offsets: optional ``[B, T]`` same unit per batch (e.g. seconds since clip start);
                used for non-uniform sampling. If ``None``, uses ``linspace(0, 1, T)`` per batch.
        Returns:
            ``[B, T, num_patches, embed_dim]`` patch tokens with spatial + temporal encodings added.
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

        if time_offsets is None:
            tpos = torch.linspace(0.0, 1.0, steps=T, device=x.device, dtype=x.dtype).view(1, T).expand(B, -1)
        else:
            if time_offsets.shape != (B, T):
                raise ValueError(f"time_offsets must be [B, T]={B, T}, got {tuple(time_offsets.shape)}")
            tpos = time_offsets.to(dtype=x.dtype, device=x.device)
            tpos = tpos - tpos[:, :1]
            denom = (tpos[:, -1:] - tpos[:, :1]).clamp(min=1e-6)
            tpos = tpos / denom

        temb = _sinusoidal_encoding_1d(tpos, self.embed_dim).to(dtype=tok.dtype)
        tok = tok + temb.unsqueeze(2)
        return tok


def patchify_spatiotemporal_images(
    x: torch.Tensor,
    embedder: VideoPatchSpatiotemporalEmbed,
    time_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Patchify video frames and add spatial + temporal encodings (see :class:`VideoPatchSpatiotemporalEmbed`).

    Args:
        x: ``[B, T, 3, 224, 224]`` (``224`` must match ``embedder.image_size``).
        embedder: holds Conv patch projection and spatial positional parameters.
        time_offsets: optional ``[B, T]`` for non-uniform temporal spacing.

    Returns:
        ``[B, T, 196, embed_dim]`` when ``patch_size=16``, ``image_size=224``.
    """
    return embedder(x, time_offsets=time_offsets)


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

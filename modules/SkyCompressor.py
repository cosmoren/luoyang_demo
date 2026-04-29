import math

import torch
import torch.nn as nn


def build_1d_sincos_pos_embed(length: int, dim: int, device=None) -> torch.Tensor:
    """Return sinusoidal 1D positional embedding with shape ``[length, dim]``."""
    assert dim % 2 == 0
    positions = torch.arange(length, device=device).float()
    omega = torch.arange(dim // 2, device=device).float()
    omega = 1.0 / (10000 ** (omega / (dim // 2)))
    out = torch.einsum("l,d->ld", positions, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device=None) -> torch.Tensor:
    """Return sinusoidal 2D positional embedding with shape ``[h*w, dim]``."""
    assert dim % 4 == 0
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    grid_y = grid_y.reshape(-1).float()
    grid_x = grid_x.reshape(-1).float()

    omega = torch.arange(dim // 4, device=device).float()
    omega = 1.0 / (10000 ** (omega / (dim // 4)))

    out_x = torch.einsum("n,d->nd", grid_x, omega)
    out_y = torch.einsum("n,d->nd", grid_y, omega)
    return torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)],
        dim=1,
    )


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        y, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class SkyFrameQueryCompressor(nn.Module):
    """
    Compress each frame:
    ``[B, T, P, D] -> [B, T, Qf, D]``
    """

    def __init__(
        self,
        dim: int,
        num_patches: int = 196,
        num_frame_queries: int = 8,
        num_heads: int = 8,
        use_patch_pos: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.num_frame_queries = num_frame_queries
        self.use_patch_pos = use_patch_pos

        self.frame_queries = nn.Parameter(torch.randn(1, num_frame_queries, dim) * 0.02)
        self.cross = CrossAttnBlock(dim, num_heads=num_heads)

        if use_patch_pos:
            side = int(math.sqrt(num_patches))
            assert side * side == num_patches, "num_patches should be square when using 2D pos"
            patch_pos = build_2d_sincos_pos_embed(side, side, dim)
            self.register_buffer(
                "patch_pos_embed",
                patch_pos.unsqueeze(0).unsqueeze(0),
                persistent=False,
            )  # [1,1,P,D]
        else:
            self.patch_pos_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_frames, num_patches, dim = x.shape
        assert num_patches == self.num_patches

        if self.use_patch_pos and self.patch_pos_embed is not None:
            x = x + self.patch_pos_embed[:, :, :num_patches, :]

        q = self.frame_queries.expand(bsz * num_frames, -1, -1)  # [B*T, Qf, D]
        kv = x.reshape(bsz * num_frames, num_patches, dim)  # [B*T, P, D]
        z = self.cross(q, kv)
        return z.reshape(bsz, num_frames, self.num_frame_queries, dim)


class SkyTwoStageCompressor(nn.Module):
    """
    Input:  ``[B, T, P, D]`` (for example ``[B,24,196,192]``)
    Output: ``[B, Qs, D]`` (for example ``[B,48,192]``)
    """

    def __init__(
        self,
        dim: int = 192,
        num_frames: int = 24,
        num_patches: int = 196,
        num_frame_queries: int = 8,
        num_sky_queries: int = 48,
        num_heads: int = 8,
        use_patch_pos: bool = True,
        use_frame_time: bool = True,
        use_coarse_spatial: bool = True,
        coarse_query_grid_hw: tuple[int, int] = (6, 8),
    ):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.num_frame_queries = num_frame_queries
        self.num_sky_queries = num_sky_queries
        self.use_frame_time = use_frame_time
        self.use_coarse_spatial = use_coarse_spatial

        self.frame_compressor = SkyFrameQueryCompressor(
            dim=dim,
            num_patches=num_patches,
            num_frame_queries=num_frame_queries,
            num_heads=num_heads,
            use_patch_pos=use_patch_pos,
        )

        if use_frame_time:
            frame_pos = build_1d_sincos_pos_embed(num_frames, dim)
            self.register_buffer(
                "frame_time_embed",
                frame_pos.unsqueeze(0).unsqueeze(2),
                persistent=False,
            )  # [1,T,1,D]
        else:
            self.frame_time_embed = None

        self.sky_queries = nn.Parameter(torch.randn(1, num_sky_queries, dim) * 0.02)

        coarse_time_embed = build_1d_sincos_pos_embed(num_sky_queries, dim)
        self.register_buffer("coarse_time_embed", coarse_time_embed.unsqueeze(0), persistent=False)

        if use_coarse_spatial:
            h, w = coarse_query_grid_hw
            assert h * w == num_sky_queries
            coarse_spatial = build_2d_sincos_pos_embed(h, w, dim)
            self.register_buffer(
                "coarse_spatial_embed",
                coarse_spatial.unsqueeze(0),
                persistent=False,
            )
        else:
            self.coarse_spatial_embed = None

        self.global_cross = CrossAttnBlock(dim, num_heads=num_heads)
        self.post_block = SelfAttnBlock(dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor, return_stage1: bool = False):
        bsz, num_frames, _, dim = x.shape
        assert num_frames == self.num_frames

        frame_tokens = self.frame_compressor(x)  # [B,T,Qf,D]
        if self.use_frame_time and self.frame_time_embed is not None:
            frame_tokens = frame_tokens + self.frame_time_embed[:, :num_frames]

        memory = frame_tokens.reshape(bsz, num_frames * self.num_frame_queries, dim)

        q = self.sky_queries.expand(bsz, -1, -1)
        q = q + self.coarse_time_embed[:, : self.num_sky_queries]
        if self.use_coarse_spatial and self.coarse_spatial_embed is not None:
            q = q + self.coarse_spatial_embed[:, : self.num_sky_queries]

        z = self.global_cross(q, memory)
        z = self.post_block(z)

        if return_stage1:
            return z, frame_tokens
        return z

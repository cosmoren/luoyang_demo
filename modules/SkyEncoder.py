"""Sky-image patch spatiotemporal embedding and alternating intra/inter-frame attention."""

from typing import Optional

import torch
import torch.nn as nn

from modules.SkyCompressor import build_continuous_time_embed

# DINOv2 hub model name → feature dim / patch size
_DINOV2_SPECS: dict[str, tuple[int, int]] = {
    "dinov2_vits14": (384, 14),
    "dinov2_vitb14": (768, 14),
    "dinov2_vitl14": (1024, 14),
    "dinov2_vitg14": (1536, 14),
}


class SkyDINOv2PatchSpatiotemporalEmbed(nn.Module):
    """
    Extract sky-image patch tokens with a frozen/finetunable DINOv2 backbone.

    Input ``[B, T, C, H, W]`` (C>=3; optional 4th channel used as ASI mask on RGB)
    → RGB ImageNet-normalized → DINOv2 patch tokens → Linear project to ``embed_dim``
    → add spatial + continuous temporal embeddings.

    Output shape matches the Conv patch embedder: ``[B, T, P, D]`` with
    ``P = (image_size / patch_size)^2`` (256 for 224 / 14).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        image_size: int = 224,
        model_name: str = "dinov2_vits14",
        freeze_backbone: bool = True,
        pretrained: bool = True,
        in_channels: int = 4,
    ):
        super().__init__()
        if model_name not in _DINOV2_SPECS:
            raise ValueError(
                f"Unknown DINOv2 model {model_name!r}; choose from {sorted(_DINOV2_SPECS)}"
            )
        backbone_dim, patch_size = _DINOV2_SPECS[model_name]
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by DINOv2 patch_size ({patch_size})"
            )

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.backbone_dim = backbone_dim
        self.freeze_backbone = bool(freeze_backbone)

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
        )
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.proj = nn.Linear(backbone_dim, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # ImageNet normalization expected by DINOv2
        self.register_buffer(
            "_img_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_img_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone in eval when frozen so BN/dropout (if any) stay fixed.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Take RGB (optionally mask with ASI channel), map to ImageNet-normalized [N,3,H,W]."""
        if x.shape[1] < 3:
            raise ValueError(f"expected at least 3 channels, got {x.shape[1]}")
        rgb = x[:, :3]
        if x.shape[1] >= 4:
            # 4th channel is ASI valid mask in [0,1]; zero out invalid pixels
            rgb = rgb * x[:, 3:4]
        # Heuristic: raw uint8-like floats in 0..255
        if float(rgb.detach().max()) > 1.5:
            rgb = rgb / 255.0
        rgb = (rgb - self._img_mean) / self._img_std
        return rgb

    def forward(self, x: torch.Tensor, timefeats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, C, H, W]`` with ``H=W=image_size``.
            timefeats: ``[B, T, 1]`` delta_t per frame.
        Returns:
            ``[B, T, P, D]``.
        """
        bsz, num_frames, channels, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"expected H=W={self.image_size}, got {height}x{width}")
        if channels < 3:
            raise ValueError(f"expected at least 3 input channels, got {channels}")
        if timefeats is None:
            raise ValueError("timefeats is required with shape [B, T, 1]")

        x_bt = x.reshape(bsz * num_frames, channels, height, width)
        rgb = self._prepare_rgb(x_bt)

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone.forward_features(rgb)
        else:
            feats = self.backbone.forward_features(rgb)

        patch_tokens = feats["x_norm_patchtokens"]  # [B*T, P, backbone_dim]
        if patch_tokens.shape[1] != self.num_patches:
            raise ValueError(
                f"DINOv2 returned {patch_tokens.shape[1]} patches, expected {self.num_patches} "
                f"for image_size={self.image_size}, patch_size={self.patch_size}"
            )

        tokens = self.proj(patch_tokens.to(dtype=self.proj.weight.dtype))  # [B*T, P, D]
        tokens = tokens.view(bsz, num_frames, self.num_patches, self.embed_dim)
        tokens = tokens + self.spatial_pos_embed
        time_tokens = build_continuous_time_embed(
            timefeats.squeeze(-1).to(dtype=torch.float32), self.embed_dim
        ).to(dtype=tokens.dtype).unsqueeze(2)
        tokens = tokens + time_tokens
        return tokens


class SkyPatchSpatiotemporalEmbed(nn.Module):
    """
    Patchify ``[B, T, 3, H, W]`` frames and add:
    - learned spatial position embedding per patch
    - sinusoidal temporal embedding built from a single ``delta_t`` scalar per frame
      (``timefeats`` ``[B, T, 1]``), via :func:`modules.SkyCompressor.build_continuous_time_embed`.

    Mirrors :class:`modules.SatEncoder.VideoPatchSpatiotemporalEmbed` for symmetry.
    """

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, image_size: int = 112,
                 in_channels: int = 3):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, timefeats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, 3, H, W]`` where ``H=W=image_size``.
            timefeats: ``[B, T, 1]`` single ``delta_t`` scalar per frame.
        Returns:
            ``[B, T, P, D]`` where ``P=num_patches`` and ``D=embed_dim``.
        """
        bsz, num_frames, channels, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"expected H=W={self.image_size}, got {height}x{width}")
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {channels}")
        if timefeats is None:
            raise ValueError("timefeats is required with shape [B, T, 1]")

        x_bt = x.reshape(bsz * num_frames, channels, height, width)
        tokens = self.patch_embed(x_bt)  # [B*T, D, gh, gw]
        tokens = tokens.flatten(2).transpose(1, 2)  # [B*T, P, D]
        tokens = tokens.view(bsz, num_frames, self.num_patches, self.embed_dim)
        tokens = tokens + self.spatial_pos_embed
        # timefeats: [B, T, 1] -> squeeze to [B, T] -> build_continuous_time_embed -> [B, T, D] -> unsqueeze -> [B, T, 1, D]
        time_tokens = build_continuous_time_embed(
            timefeats.squeeze(-1).to(dtype=torch.float32), self.embed_dim
        ).to(dtype=tokens.dtype).unsqueeze(2)
        tokens = tokens + time_tokens
        return tokens


def patchify_spatiotemporal_sky_images(
    x: torch.Tensor,
    embedder: SkyPatchSpatiotemporalEmbed | SkyDINOv2PatchSpatiotemporalEmbed,
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

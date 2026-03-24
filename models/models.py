"""
1D temporal CNN for encoding time-series (e.g. PV history with mask).
Input shape: (batch, in_channels, seq_len). Optional mask: (batch, seq_len).
"""

import torch
import torch.nn as nn
from typing import Optional
import torch
from .timesformer import TimeSformerFeatureExtractor, TimesformerConfig
import logging
import os
import contextlib
import io

class TemporalCNN1d(nn.Module):
    """
    1D temporal CNN. Input [B, C, T], output [B, C_out, T].
    B: batch, C: channels, T: time steps. C_out is given by out_channels.
    Optional mask: (batch, seq_len) is multiplied with x element-wise before conv.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: list = (32, 64, 64),
        kernel_size: int = 3,
        out_channels: int = 1,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = list(hidden_channels)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        layers = []
        c_in = in_channels
        for c_out in self.hidden_channels:
            layers.append(
                nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            c_in = c_out
        self.conv_blocks = nn.Sequential(*layers)
        self._out_channels = c_in
        self.conv_out = nn.Conv1d(self._out_channels, self.out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, T], C channels, T time steps
            mask: (batch, seq_len), 1=valid 0=masked. If given, x = x * mask before conv.
        Returns:
            [B, C_out, T]
        """
        if mask is not None:
            x = x * mask.to(x.dtype)

        out = self.conv_blocks(x)   # (B, _out_channels, T)
        out = self.conv_out(out)    # (B, C_out, T)
        return out


class MLP(nn.Module):
    """
    Small MLP: Linear -> ReLU -> [Linear -> ReLU] -> Linear.
    Input: [..., in_dim]. Output: [..., out_dim].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple = (512, 256, 256, 128),
        out_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim]. Returns: [..., out_dim]."""
        return self.mlp(x)


class FC(nn.Module):
    """
    Single linear (fully connected) output layer.
    Input: [..., in_dim]. Output: [..., out_dim].
    """

    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim]. Returns: [..., out_dim]."""
        return self.fc(x)


class CrossAttention(nn.Module):
    """
    Cross attention via nn.MultiheadAttention: query from one sequence, key/value from another.
    Inputs: query [B, Lq, query_dim], key [B, Lkv, key_dim], value [B, Lkv, value_dim].
    Output: [B, Lq, embed_dim].
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: Optional[int] = None,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        value_dim = value_dim if value_dim is not None else key_dim
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(query_dim, embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=key_dim,
            vdim=value_dim,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Lq, query_dim]
            key: [B, Lkv, key_dim]
            value: [B, Lkv, value_dim]
            key_value_mask: [B, Lkv], 1=valid 0=masked. True in key_padding_mask means ignore.
        Returns:
            [B, Lq, embed_dim]
        """
        q = self.w_q(query)
        key_padding_mask = None
        if key_value_mask is not None:
            key_padding_mask = (key_value_mask == 0)
        out, _ = self.mha(q, key, value, key_padding_mask=key_padding_mask)
        return out


class pv_forecasting_model(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: list = (32, 64, 64), kernel_size: int = 3, out_dim: Optional[int] = None, use_batchnorm: bool = True, dropout: float = 0.0, dev_dn_list: Optional[list] = None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = list(hidden_channels)
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.TCN = TemporalCNN1d(in_channels=8, out_channels=64, use_batchnorm=use_batchnorm, dropout=dropout)
        self.cross_attention = CrossAttention(query_dim=8, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.fc = FC(in_dim=64, out_dim=1)

    def forward(self, device_id: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None, history_solar_features: Optional[torch.Tensor] = None, forecast_solar_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x_masked = x * mask.to(x.dtype)
        pv_history = torch.cat([x_masked, x, history_solar_features.permute(0, 2, 1)], dim=1)  # [B, 8, T]
        pv_hist_mem = self.TCN(pv_history, mask)      # [B, C_out, T]()
        KV_hist_mem = pv_hist_mem.permute(0, 2, 1)   # [B, T, C_out]

        forcast_pv_features = self.cross_attention(query=forecast_solar_features, key=KV_hist_mem, value=KV_hist_mem)   #[B,T,D]
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forcast_pv_features.shape[1], 1)
        fused = torch.cat([forcast_pv_features, inverter_features], dim=2)
        pv_feats = self.pv_feats_head(fused)
        pv = self.fc(pv_feats)

        return pv.squeeze(-1)


class pv_forecasting_model_vit(nn.Module):
    def __init__(self, use_batchnorm: bool = True, dropout: float = 0.0, dev_dn_list: Optional[list] = None):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        logging.getLogger("transformers").setLevel(logging.ERROR)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        # TimeSformer for satellite images
        config = TimesformerConfig(
            num_frames=24,        # 改成你要的 T
            image_size=224,
            patch_size=16,
            num_channels=3,
        )

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.sat_extractor = TimeSformerFeatureExtractor(
                pretrained="facebook/timesformer-base-finetuned-k400",
                output_format="sequence",
                normalize=True,
                config=config,
            )
        # TimeSformer for sky imager images
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.skimg_extractor = TimeSformerFeatureExtractor(
                pretrained="facebook/timesformer-base-finetuned-k400",
                output_format="sequence",
                normalize=True,
                config=config,
            )

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.TCN = TemporalCNN1d(in_channels=11, out_channels=64, use_batchnorm=use_batchnorm, dropout=dropout)
        self.cross_attention_pv = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.cross_attention_sat = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.cross_attention_skimg = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.cross_attention_nwp = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.sat_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.skimg_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.timefeats_encoder = MLP(in_dim=9, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.nwp_updim = MLP(in_dim=16, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.pv_feats_head = MLP(in_dim=272, hidden_dims=(512, 256, 128), out_dim=64, dropout=0.0)
        self.fc = FC(in_dim=64, out_dim=1)

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor, 
                pv_mask: Optional[torch.Tensor] = None, 
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None,
                nwp_timefeats: Optional[torch.Tensor] = None,) -> torch.Tensor:
        
        # PV features
        pv_masked = pv * pv_mask.to(pv.dtype)
        pv_history = torch.cat([pv_masked, pv_mask, pv_timefeats.permute(0, 2, 1)], dim=1)  # [B, 8, T]
        pv_hist_mem = self.TCN(pv_history, pv_mask)     # [B, C_out, T]()
        KV_hist_mem = pv_hist_mem.permute(0, 2, 1)   # [B, T, C_out]
        forecast_pv_features = self.cross_attention_pv(query=forecast_timefeats, key=KV_hist_mem, value=KV_hist_mem)   #[B,T,D]

        # Satellite features    
        if sat_tensor is None:
            forecast_sat_features = torch.zeros(pv.shape[0], forecast_timefeats.shape[1], 64).to(pv.device)
        else:
            sat_tensor = nn.functional.interpolate(sat_tensor.view(-1, 3, *sat_tensor.shape[-2:]), size=(224, 224), mode='bilinear', align_corners=False).view(*sat_tensor.shape[:3], 224, 224)
            sat_features = self.sat_extractor(sat_tensor[:,-24:,:,:,:])   #[B,T=xx,D=768]
            sat_timefeats_hdim = self.timefeats_encoder(sat_timefeats[:,-24:,:])
            sat_down_features = self.sat_downdim(sat_features) # [B,T=xx,D=64]
            sat_down_features = sat_down_features + sat_timefeats_hdim
            forecast_sat_features = self.cross_attention_sat(query=forecast_timefeats, key=sat_down_features, value=sat_down_features)

        # Sky imager features
        if skimg_tensor is None:
            forecast_skimg_features = torch.zeros(pv.shape[0], forecast_timefeats.shape[1], 64).to(pv.device)
        else:
            skimg_tensor = nn.functional.interpolate(skimg_tensor.view(-1, 3    , *skimg_tensor.shape[-2:]), size=(224, 224), mode='bilinear', align_corners=False).view(*skimg_tensor.shape[:3], 224, 224)
            skimg_features = self.skimg_extractor(skimg_tensor)   #[B,T=12,D=768]
            skimg_down_features = self.skimg_downdim(skimg_features) # [B,T=12,D=64]
            forecast_skimg_features = self.cross_attention_skimg(query=forecast_timefeats, key=skimg_down_features, value=skimg_down_features)
            skimg_timefeats_hdim = self.timefeats_encoder(skimg_timefeats)
            forecast_skimg_features = forecast_skimg_features + skimg_timefeats_hdim

        # NWP features
        if nwp_tensor is None:
            forecast_nwp_features = torch.zeros(pv.shape[0], forecast_timefeats.shape[1], 64).to(pv.device)
        else:
            nwp_fc_wtime = torch.cat([nwp_tensor.permute(0, 2, 1), nwp_timefeats], dim=-1) # [4, 192, 16]
            nwp_fc_mem = self.nwp_updim(nwp_fc_wtime)
            forecast_nwp_features = self.cross_attention_nwp(query=forecast_timefeats, key=nwp_fc_mem, value=nwp_fc_mem)

        # Inverter features (embeddings)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_sat_features.shape[1], 1)

        # Fuse and predict
        fused = torch.cat([forecast_pv_features, forecast_sat_features, forecast_skimg_features, forecast_nwp_features, inverter_features], dim=2)   # [B=1,T=192,C=272]
        pv_feats = self.pv_feats_head(fused)
        pv = self.fc(pv_feats)

        return pv.squeeze(-1)
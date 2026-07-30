"""
1D temporal CNN for encoding time-series (e.g. PV history with mask).
Input shape: (batch, in_channels, seq_len). Optional mask: (batch, seq_len).
"""

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from tabm import TabM

from modules.SatEncoder import (
    AlternatingIntraInterFrameAttention,
    VideoPatchSpatiotemporalEmbed,
    patchify_spatiotemporal_images,
)
from modules.SatCompressor import SatelliteTwoStageCompressor
from modules.SkyEncoder import (
    SkyAlternatingIntraInterFrameAttention,
    SkyDINOv2PatchSpatiotemporalEmbed,
    SkyPatchSpatiotemporalEmbed,
    patchify_spatiotemporal_sky_images,
)
from modules.SkyCompressor import SkyTwoStageCompressor
from modules.SimVPEncoder import SimVPFeatureExtractor
from dataloader.folsom import _FOLSOM_NWP_FEATURE_COLS
from .timesformer import TimeSformerFeatureExtractor, TimesformerConfig
import logging
import os
import contextlib
import io


# Per-feature normalizers applied to columns of ``nwp_tensor`` in
# ``pv_forecasting_model_vit_imgs``. Order-agnostic dispatch: the column index for
# each feature comes from ``_FOLSOM_NWP_FEATURE_COLS`` (see ``dataloader/folsom.py``);
# the trainer feeds the raw tensor (no channel remap), and the model loops over the
# resolved ``self.nwp_features`` list applying these closures in feature order.
#
# Constants (data-fit, post-refactor; ``temperature`` differs from the pre-refactor
# ``(t - 288.15) / 10`` to match the merged Folsom NWP distribution):
#   * dwsw          (W/m^2): (x/1000 - 0.5) * 2 -> ~[-1, +1.5] at peak sun
#   * cloud_cover   (%)    : (x - 50)/50         -> [-1, +1]
#   * precipitation (mm/h) : log1p(x.clamp(>=0))/3 -> ~[0, 2] for typical events
#   * pressure      (Pa)   : (x - 100500)/500    -> roughly [-3, +3]
#   * wind-u/wind-v (m/s)  : x / 5
#   * temperature   (K)    : (x - 295)/12        -> ~[-2, +2] across Folsom annual range
#   * rel_humidity  (%)    : (x - 50)/50         -> [-1, +1]
#
# The trailing invalid-mask channel (``nwp_tensor[:, :, -1]``) is passed through as-is
# when ``use_invalid_mask`` is set on the model.
NWP_FEATURE_NORMALIZERS = {
    "dwsw":          lambda x: (x / 1000.0 - 0.5) * 2.0,
    "cloud_cover":   lambda x: (x - 50.0) / 50.0,
    "precipitation": lambda x: torch.log1p(x.clamp(min=0)) / 3.0,
    "pressure":      lambda x: (x - 100500.0) / 500.0,
    "wind-u":        lambda x: x / 5.0,
    "wind-v":        lambda x: x / 5.0,
    "temperature":   lambda x: (x - 295.0) / 12.0,
    "rel_humidity":  lambda x: (x - 50.0) / 50.0,
}
# Sanity: every canonical NWP feature must have a normalizer.
assert set(NWP_FEATURE_NORMALIZERS) == set(_FOLSOM_NWP_FEATURE_COLS), (
    "NWP_FEATURE_NORMALIZERS and _FOLSOM_NWP_FEATURE_COLS must cover the same features: "
    f"normalizers={sorted(NWP_FEATURE_NORMALIZERS)} vs feature_cols={sorted(_FOLSOM_NWP_FEATURE_COLS)}"
)

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
        kt = self.fc(x)
        return kt


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

        # PV branch: x_masked + x (2 ch) + pv_timefeats (8 solar + 1 delta_t per step = 9)
        self._pv_timefeat_dim = 9
        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        tcn_in = 2 + self._pv_timefeat_dim
        self.TCN = TemporalCNN1d(in_channels=tcn_in, out_channels=64, use_batchnorm=use_batchnorm, dropout=dropout)
        self.cross_attention = CrossAttention(
            query_dim=self._pv_timefeat_dim, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout
        )
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.fc = FC(in_dim=64, out_dim=1)

    def forward(
        self,
        device_id: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pv_timefeats: Optional[torch.Tensor] = None,
        forecast_timefeats: Optional[torch.Tensor] = None,
        history_solar_features: Optional[torch.Tensor] = None,
        forecast_solar_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T_in] PV history (power).
            mask: [B, 1, T_in] valid mask.
            pv_timefeats: [B, T_in, C_tf] aligned with history (C_tf=9 from dataloader).
            forecast_timefeats: [B, T_out, C_tf] query timesteps for prediction.
            history_solar_features / forecast_solar_features: legacy names ([B, T, 6] / [B, T, 8]),
                zero-padded to C_tf=9 when shorter.
        """
        if pv_timefeats is None:
            pv_timefeats = history_solar_features
        if forecast_timefeats is None:
            forecast_timefeats = forecast_solar_features
        if pv_timefeats is None or forecast_timefeats is None:
            raise ValueError(
                "pv_timefeats and forecast_timefeats (or legacy history_solar_features / forecast_solar_features) are required"
            )

        def _pad_timefeat(t: torch.Tensor, want: int) -> torch.Tensor:
            c = t.size(-1)
            if c == want:
                return t
            if c < want:
                pad = want - c
                return torch.nn.functional.pad(t, (0, pad))
            return t[..., :want]

        pv_timefeats = _pad_timefeat(pv_timefeats, self._pv_timefeat_dim)
        forecast_timefeats = _pad_timefeat(forecast_timefeats, self._pv_timefeat_dim)

        x_masked = x * mask.to(x.dtype)
        # [B, T, C] -> [B, C, T]
        hist_tf = pv_timefeats.permute(0, 2, 1)
        pv_history = torch.cat([x_masked, x, hist_tf], dim=1)
        pv_hist_mem = self.TCN(pv_history, mask)
        KV_hist_mem = pv_hist_mem.permute(0, 2, 1)

        forcast_pv_features = self.cross_attention(query=forecast_timefeats, key=KV_hist_mem, value=KV_hist_mem)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forcast_pv_features.shape[1], 1)
        fused = torch.cat([forcast_pv_features, inverter_features], dim=2)
        pv_feats = self.pv_feats_head(fused)
        pv = self.fc(pv_feats)

        return pv.squeeze(-1)


# Using PV history to forecast PV
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
            image_size=112,
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
        # self.cross_attention_sat = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        # self.cross_attention_skimg = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        # self.sat_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        # self.skimg_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        # self.timefeats_encoder = MLP(in_dim=9, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.fc = FC(in_dim=64, out_dim=1)

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor, 
                pv_mask: Optional[torch.Tensor] = None, 
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None,
                nwp_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # PV features
        pv_masked = pv * pv_mask.to(pv.dtype)
        pv_history = torch.cat([pv_masked, pv_mask, pv_timefeats.permute(0, 2, 1)], dim=1)  # [B, C=11, T]
        pv_hist_mem = self.TCN(pv_history, pv_mask)     # [B, C_out, T]()
        KV_hist_mem = pv_hist_mem.permute(0, 2, 1)   # [B, T, C_out]
        forecast_pv_features = self.cross_attention_pv(query=forecast_timefeats, key=KV_hist_mem, value=KV_hist_mem)   #[B,T,D]

        '''
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
        '''

        # Inverter features (embeddings)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_pv_features.shape[1], 1)

        # Fuse and predict
        fused = torch.cat([forecast_pv_features, inverter_features], dim=2)   # [B=1,T=192,C=80]
        pv_feats = self.pv_feats_head(fused)
        delta_kt = self.fc(pv_feats)

        

        return pv.squeeze(-1)


# Using PV history and NWP to forecast PV, solar features and NWP features are used as query
class pv_forecasting_model_vit_nwp(nn.Module):
    def __init__(self, use_batchnorm: bool = True, dropout: float = 0.0, dev_dn_list: Optional[list] = None):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.TCN = TemporalCNN1d(in_channels=11, out_channels=64, use_batchnorm=use_batchnorm, dropout=dropout)
        self.query_mlp = MLP(in_dim=13, hidden_dims=(64, 64), out_dim=64, dropout=0.0)
        self.cross_attention_pv = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        # self.cross_attention_sat = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        # self.cross_attention_skimg = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        # self.sat_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        # self.skimg_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        # self.timefeats_encoder = MLP(in_dim=9, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.fc = FC(in_dim=64, out_dim=1)

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor, 
                pv_mask: Optional[torch.Tensor] = None, 
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # PV features
        pv_masked = pv * pv_mask.to(pv.dtype)
        pv_history = torch.cat([pv_masked, pv_mask, pv_timefeats.permute(0, 2, 1)], dim=1)  # [B, C=11, T]
        pv_hist_mem = self.TCN(pv_history, pv_mask)     # [B, C_out, T]()
        KV_hist_mem = pv_hist_mem.permute(0, 2, 1)   # [B, T, C_out]

        ssrd_normalized = (nwp_tensor[:,:,0]/1000 - 0.5)*2
        msl_normalized = (nwp_tensor[:,:,1]-101325)/1000
        t2m_normalized = (nwp_tensor[:,:,2]-288.15)/10
        forecast_ssrd_timefeats = torch.cat([forecast_timefeats, ssrd_normalized.unsqueeze(2), msl_normalized.unsqueeze(2), t2m_normalized.unsqueeze(2), nwp_tensor[:,:,-1].unsqueeze(2)], dim=2)
        forecast_query = self.query_mlp(forecast_ssrd_timefeats)
        forecast_pv_features = self.cross_attention_pv(query=forecast_query, key=KV_hist_mem, value=KV_hist_mem)   #[B,T,D]

        '''
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
        '''

        # Inverter features (embeddings)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_pv_features.shape[1], 1)

        # Fuse and predict
        fused = torch.cat([forecast_pv_features, inverter_features], dim=2)   # [B=1,T=192,C=80]
        pv_feats = self.pv_feats_head(fused)
        pv = self.fc(pv_feats)

        return pv.squeeze(-1)


# Using PV history and NWP to forecast PV, solar features and NWP features are used as query
# This is the best pv branch model
class pv_forecasting_model_vit_nwp_short(nn.Module):
    def __init__(self, use_batchnorm: bool = True, dropout: float = 0.0, dev_dn_list: Optional[list] = None):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.TCN = TemporalCNN1d(in_channels=11, out_channels=64, use_batchnorm=use_batchnorm, dropout=dropout)

        self.time_mlp = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.corase_idx = [18, 54, 90, 126, 162, 198, 234, 270, 295, 308, 322, 335, 349, 362, 376, 389,
                           403, 416, 430, 443, 457, 470, 484, 497, 505, 508, 511, 514, 517, 520, 523, 526,
                           529, 532, 535, 538, 541, 544, 547, 550, 553, 556, 559, 562, 565, 568, 571, 574]
        self.learnable_pv_queries = nn.Parameter(torch.randn(1, 48, 64))
        self.cross_attention_pv_compression = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        
        self.query_mlp = MLP(in_dim=12, hidden_dims=(64, 64), out_dim=64, dropout=0.0)
        self.cross_attention_pv = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        # self.cross_attention_sat = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        # self.cross_attention_skimg = CrossAttention(query_dim=9, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        # self.sat_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        # self.skimg_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        # self.timefeats_encoder = MLP(in_dim=9, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.fc = FC(in_dim=64, out_dim=1)

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor, 
                pv_mask: Optional[torch.Tensor] = None, 
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # PV history features
        _ = nwp_history  # wired from dataloader/trainer for future model use
        pv_masked = pv * pv_mask.to(pv.dtype)
        pv_history = torch.cat([pv_masked, pv_mask, pv_timefeats.permute(0, 2, 1)], dim=1)  # [B, C=11, T]
        pv_hist_mem = self.TCN(pv_history, pv_mask)     # [B, C_out, T]()
        KV_hist_mem = pv_hist_mem.permute(0, 2, 1)   # [B, T, C_out]

        # PV time features compression
        pv_timefeats_corase = pv_timefeats[:, self.corase_idx, :]
        pv_timefeats_corase = self.time_mlp(pv_timefeats_corase)
        learnable_pv_queries = self.learnable_pv_queries.repeat(pv_timefeats_corase.shape[0], 1, 1)
        corase_queries = learnable_pv_queries + pv_timefeats_corase
        KV_hist_mem_compressed = self.cross_attention_pv_compression(query=corase_queries, key=KV_hist_mem, value=KV_hist_mem) # [B,48,D=64] 48 pv tokens
        KV_hist_mem_compressed = KV_hist_mem_compressed + corase_queries
        
        # Forecast features
        ssrd_normalized = (nwp_tensor[:,:,0]/1000 - 0.5)*2
        # msl_normalized = (nwp_tensor[:,:,1]-101325)/1000
        t2m_normalized = (nwp_tensor[:,:,2]-288.15)/10
        forecast_ssrd_timefeats = torch.cat([forecast_timefeats, ssrd_normalized.unsqueeze(2), t2m_normalized.unsqueeze(2), nwp_tensor[:,:,-1].unsqueeze(2)], dim=2)
        forecast_query = self.query_mlp(forecast_ssrd_timefeats)
        forecast_pv_features = self.cross_attention_pv(query=forecast_query, key=KV_hist_mem_compressed, value=KV_hist_mem_compressed)   #[B,T,D]

        # Inverter features (embeddings)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_pv_features.shape[1], 1)

        # Fuse and predict
        fused = torch.cat([forecast_pv_features, inverter_features], dim=2)   # [B=1,T=192,C=80]
        pv_feats = self.pv_feats_head(fused)
        pv = self.fc(pv_feats)

        return pv.squeeze(-1)



# Default per-feature NWP selection for ``pv_forecasting_model_vit_imgs``. Matches the
# post-refactor "minimal" preset and the pre-refactor hardcoded behaviour (which fed
# ssrd-like + temperature-like channels through ``query_mlp``); used as the legacy
# default when an older checkpoint does not record ``nwp_features``.
_DEFAULT_VIT_IMGS_NWP_FEATURES: tuple[str, ...] = ("dwsw", "temperature")
_DEFAULT_VIT_IMGS_NWP_USE_INVALID_MASK: bool = False


# Using PV history and NWP to forecast PV, solar features and NWP features are used as query
class pv_forecasting_model_vit_imgs(nn.Module):
    """
    Like ``pv_forecasting_model_vit_nwp`` but satellite frames go through
    :func:`modules.SatEncoder.patchify_spatiotemporal_images` and
    :class:`modules.SatEncoder.AlternatingIntraInterFrameAttention`, then cross-attend into the forecast query.

    NWP forecast-query channels are configurable: ``nwp_features`` selects which columns
    of ``_FOLSOM_NWP_FEATURE_COLS`` to read from ``nwp_tensor`` (each normalised via
    :data:`NWP_FEATURE_NORMALIZERS`), and ``use_invalid_mask`` toggles passing the trailing
    per-step invalid mask channel (``nwp_tensor[:, :, -1]``) through as-is.
    """

    def __init__(
        self,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        dev_dn_list: Optional[list] = None,
        nwp_features: Optional[list[str]] = None,
        use_invalid_mask: bool = _DEFAULT_VIT_IMGS_NWP_USE_INVALID_MASK,
        nwp_dropout_prob: float = 0.0,
        nwp_history_dropout_prob: float = 0.0,
        # Folsom Zarr RGB+image_valid = 4 (knobs off). Trainer must pass dataset.sky_in_channels.
        sky_in_channels: int = 4,
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        if sky_in_channels < 1:
            raise ValueError(f"sky_in_channels must be >= 1, got {sky_in_channels}")
        self.sky_in_channels = int(sky_in_channels)

        if nwp_features is None:
            nwp_features = list(_DEFAULT_VIT_IMGS_NWP_FEATURES)
        nwp_features = list(nwp_features)
        unknown = [n for n in nwp_features if n not in NWP_FEATURE_NORMALIZERS]
        if unknown:
            raise ValueError(
                f"Unknown NWP feature(s) {unknown}; valid features are "
                f"{sorted(NWP_FEATURE_NORMALIZERS)}"
            )
        dupes = [n for n in nwp_features if nwp_features.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate NWP feature(s) in nwp_features: {sorted(set(dupes))}")
        self.nwp_features: list[str] = nwp_features
        self.use_invalid_mask: bool = bool(use_invalid_mask)
        self.nwp_dropout_prob: float = max(0.0, min(1.0, float(nwp_dropout_prob)))
        self.nwp_history_dropout_prob: float = max(
            0.0, min(1.0, float(nwp_history_dropout_prob))
        )
        # Pre-resolve column indices in ``nwp_tensor`` (last dim = 8 features + 1 mask).
        self._nwp_feature_indices: list[int] = [
            _FOLSOM_NWP_FEATURE_COLS.index(name) for name in self.nwp_features
        ]
        # Forecast-query input dim = 3 time feats + N NWP features (+ 1 if mask passed through).
        query_mlp_in_dim = 3 + 4

        dim = 64
        self.tabm_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # TabM modality embedding
        self.pv_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # PV modality embedding
        self.sat_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Satellite image modality embedding
        self.sky_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Sky imagem odality embedding

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.pv_hist_in_channels = 8
        self.nwp_future_steps = 16
        self.nwp_future_channels = 2
        self.nwp_future_flat_dim = self.nwp_future_steps * self.nwp_future_channels
        self.TCN = TemporalCNN1d(
            in_channels=self.pv_hist_in_channels,
            out_channels=64,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.corase_idx = [18, 54, 90, 126, 162, 198, 234, 270, 295, 308, 322, 335, 349, 362, 376, 389,
                           403, 416, 430, 443, 457, 470, 484, 497, 505, 508, 511, 514, 517, 520, 523, 526,
                           529, 532, 535, 538, 541, 544, 547, 550, 553, 556, 559, 562, 565, 568, 571, 574]
        # This model path uses 576-step history (derived from coarse idx settings).
        self.pv_hist_len = max(self.corase_idx) + 2
        self.learnable_pv_queries = nn.Parameter(torch.randn(1, 48, 64))
        self.cross_attention_pv_compression = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)        
        self.query_mlp = MLP(in_dim=query_mlp_in_dim, hidden_dims=(64, 64), out_dim=64, dropout=0.0)

        self.sat_embed_dim = 64
        self.sat_patch_embed = VideoPatchSpatiotemporalEmbed(
            embed_dim=self.sat_embed_dim, patch_size=16, image_size=112
        )
        self.sat_alt_attn = AlternatingIntraInterFrameAttention(
            embed_dim=self.sat_embed_dim, num_heads=8, num_cycles=4, dropout=dropout
        )

        self.sat_two_stage_compressor = SatelliteTwoStageCompressor(
                                            dim=self.sat_embed_dim,
                                            num_frames=24,
                                            num_patches=49,
                                            num_frame_queries=8,   # 196 -> 8 per frame
                                            num_sat_queries=48,    # 24*8=192 -> 48 final tokens
                                            num_heads=8,
                                            use_patch_pos=True,
                                            use_frame_time=True,
                                            use_coarse_spatial=True,
                                            coarse_query_grid_hw=(6, 8),)
        
        self.sky_embed_dim = 64
        self.sky_patch_embed = SkyPatchSpatiotemporalEmbed(
            embed_dim=self.sky_embed_dim,
            patch_size=16,
            image_size=224,
            in_channels=self.sky_in_channels,
        )
        self.sky_alt_attn = SkyAlternatingIntraInterFrameAttention(
            embed_dim=self.sky_embed_dim, num_heads=8, num_cycles=4, dropout=dropout
        )
        self.sky_two_stage_compressor = SkyTwoStageCompressor(
                                            dim=self.sky_embed_dim,
                                            num_frames=30,
                                            num_patches=196,
                                            num_frame_queries=8,
                                            num_sky_queries=48,
                                            num_heads=8,
                                            use_patch_pos=True,
                                            use_frame_time=True,
                                            use_coarse_spatial=True,
                                            coarse_query_grid_hw=(6, 8),)

        self.cross_attention_pv = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)  # PV 64 + sat 64 + inverter 16
        self.fc = FC(in_dim=64, out_dim=1)
        # TabM Branch 1: flatten PV history (+ padded NWP future) → single-step kt [B, 1].
        self.pv_tabm_head = TabM(
            n_num_features=2336,
            cat_cardinalities=None,
            d_out=1,
            n_blocks=3,
            d_block=512,
            dropout=dropout,
            k=32,
            arch_type="tabm",
            start_scaling_init="normal",
        )
        # Cached feature right before TabM final output projection.
        self.pv_tabm_preoutput_features: Optional[torch.Tensor] = None
        self._pv_tabm_output_pre_hook_handle = self.pv_tabm_head.output.register_forward_pre_hook(
            self._capture_pv_tabm_preoutput_features
        )

    @staticmethod
    def _apply_channel_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if (not training) or p <= 0.0:
            return x
        keep = 1.0 - float(p)
        if keep <= 0.0:
            return torch.zeros_like(x)
        bsz, _, channels = x.shape
        mask = (torch.rand((bsz, 1, channels), device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def _capture_pv_tabm_preoutput_features(self, module: nn.Module, inputs: tuple) -> None:
        _ = module
        if not inputs:
            self.pv_tabm_preoutput_features = None
            return
        x = inputs[0]
        self.pv_tabm_preoutput_features = x if isinstance(x, torch.Tensor) else None

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor, 
                pv_mask: Optional[torch.Tensor] = None, 
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None,
                nwp_history: Optional[torch.Tensor] = None,
                nwp_forecast_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        pv_masked = pv * pv_mask.to(pv.dtype)

        pv_timefeats = pv_timefeats[:, :, [2,3,8]]
        forecast_timefeats = forecast_timefeats[:, :, [2,3,8]]

        # Read history features for PV branch; if missing, fallback to zeros.
        if nwp_history is None:
            nwp_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_history_ghi = ((nwp_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        if nwp_forecast_history is None:
            nwp_forecast_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_forecast_history_ghi = ((nwp_forecast_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        delta_nwp_history_ghi = nwp_history_ghi - nwp_forecast_history_ghi

        pv_history = torch.cat(
            [
                pv_masked,
                pv_mask,
                pv_timefeats.permute(0, 2, 1),
                nwp_history_ghi,
                nwp_forecast_history_ghi,
                delta_nwp_history_ghi,
            ],
            dim=1,
        )  # [B, C=8, T]

        nwp_ssrd_normalized = (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0
        nwp_t2m_normalized = (nwp_tensor[:, :, 2] - 288.15) / 10.0

        
        pv_hist_flat = pv_history[:,[0,5,6,7],:].reshape(pv_history.shape[0], -1)
        nwp_tensor_for_tabm = torch.stack(
            [
                nwp_ssrd_normalized,
                nwp_t2m_normalized,
            ],
            dim=2,
        )
        nwp_tensor_flat = nwp_tensor_for_tabm.reshape(nwp_tensor_for_tabm.shape[0], -1)
        cur_nwp_dim = nwp_tensor_flat.shape[1]
        if cur_nwp_dim < self.nwp_future_flat_dim:
            pad = torch.zeros(
                nwp_tensor_flat.shape[0],
                self.nwp_future_flat_dim - cur_nwp_dim,
                device=nwp_tensor_flat.device,
                dtype=nwp_tensor_flat.dtype,
            )
            nwp_tensor_flat = torch.cat([nwp_tensor_flat, pad], dim=1)
        elif cur_nwp_dim > self.nwp_future_flat_dim:
            nwp_tensor_flat = nwp_tensor_flat[:, : self.nwp_future_flat_dim]

        x_num = torch.cat([pv_hist_flat, nwp_tensor_flat], dim=1)

        self.pv_tabm_preoutput_features = None
        tabm_out = self.pv_tabm_head(x_num=x_num, x_cat=None)  # [B, K, 1]
        kt_tabm = tabm_out.mean(dim=1)  # [B, 1]

        # Forecast queries: Luoyang total variant only uses two NWP channels:
        # ssrd (idx=0) and t2m (idx=2), where nwp_tensor layout is
        # [ssrd, msl, t2m, u10, v10, u100, v100, nan_mask].

        nwp_channels = [forecast_timefeats]
        if nwp_tensor is not None:
            nwp_feats = torch.stack(
                [
                    (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0,
                    nwp_tensor[:, :, 1],
                    (nwp_tensor[:, :, 2] - 288.15) / 10.0,
                    nwp_tensor[:, :, 3],
                ],
                dim=2,
            )
            nwp_feats = self._apply_channel_dropout(
                nwp_feats,
                p=self.nwp_dropout_prob,
                training=self.training,
            )
            nwp_channels.append(nwp_feats)
        else:
            zero_nwp_feat = torch.zeros(
                forecast_timefeats.shape[:2],
                device=forecast_timefeats.device,
                dtype=forecast_timefeats.dtype,
            )
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # ssrd
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # t2m
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # ssrd
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # t2m

        forecast_ssrd_timefeats = torch.cat(nwp_channels, dim=2)
        
        forecast_query = self.query_mlp(forecast_ssrd_timefeats)

        # satellite images encoder
        B, T_out, _ = forecast_query.shape

        if sat_tensor is None or sat_tensor.max() == 0:
            sat_compressed = torch.zeros(B, 48, 64, device=pv.device, dtype=pv.dtype) + self.sat_mod_embed
            sat_mask = torch.zeros(B, 48, device=pv.device, dtype=pv.dtype)
        else:
            B_sat, T_sat, C_sat, H_sat, W_sat = sat_tensor.shape
            if C_sat != 3:
                raise ValueError(f"sat_tensor expected 3 channels, got {C_sat}")
            sat_hr = nn.functional.interpolate(
                sat_tensor.reshape(B_sat * T_sat, C_sat, H_sat, W_sat),
                size=(112, 112),
                mode="bilinear",
                align_corners=False,
            ).view(B_sat, T_sat, 3, 112, 112)

            sat_timefeats = sat_timefeats[:, :, [2,3,8]]
            sat_patch_tokens = patchify_spatiotemporal_images(sat_hr, self.sat_patch_embed, timefeats=sat_timefeats[:,:,-1].unsqueeze(2))
            sat_patch_tokens = self.sat_alt_attn(sat_patch_tokens)  # [B,T=24,P=49,D=64]
            sat_start = sat_timefeats[:, 0, -1]   # [B]
            sat_end   = sat_timefeats[:, -1, -1]  # [B]
            sat_steps = torch.linspace(0, 1, 48, device=sat_patch_tokens.device)
            sat_query_times = sat_start[:, None] + (sat_end - sat_start)[:, None] * sat_steps  # [B, 48]
            sat_compressed = self.sat_two_stage_compressor(sat_patch_tokens, sat_query_times) + self.sat_mod_embed  # [B,P=48,D=64]

            sat_timefeats_48 = F.interpolate(
                    sat_timefeats.transpose(1, 2),       # -> [B, F=3, T=24]  (interp 要求 [N, C, L])
                    size=48,
                    mode="linear",
                    align_corners=True,                  # 头尾对齐 -> 保留首末时刻原值
                ).transpose(1, 2)

            sat_timefeats_48_hd = self.time_mlp(sat_timefeats_48)
            sat_compressed = sat_compressed + sat_timefeats_48_hd
            
            # print('sat_patch_tokens: ', sat_patch_tokens.shape, sat_compressed.shape)
            sat_mask = torch.ones(B, sat_compressed.shape[1], device=pv.device, dtype=pv.dtype)

        if sat_valid_mask is not None:
            if sat_valid_mask.dim() != 1 or sat_valid_mask.shape[0] != B:
                raise ValueError(f"sat_valid_mask expected [B], got {sat_valid_mask.shape}")
            sat_valid_mask = sat_valid_mask.to(device=pv.device, dtype=pv.dtype).unsqueeze(1)
            sat_compressed = sat_compressed * sat_valid_mask.unsqueeze(2)
            sat_mask = sat_mask * sat_valid_mask

        # sky images encoder (Branch 2 residual memory)
        if skimg_tensor is None or skimg_tensor.max() == 0:
            sky_compressed = torch.zeros(B, 48, 64, device=pv.device, dtype=pv.dtype) + self.sky_mod_embed
            sky_mask = torch.zeros(B, 48, device=pv.device, dtype=pv.dtype)
        else:
            B_sky, T_sky, C_sky, H_sky, W_sky = skimg_tensor.shape
            if C_sky != self.sky_in_channels:
                raise ValueError(
                    f"skimg_tensor channels {C_sky} != model sky_in_channels {self.sky_in_channels}"
                )
            skimg_timefeats = skimg_timefeats[:, :, [2, 3, 8]]
            if H_sky != 224 or W_sky != 224:
                sky_hr = nn.functional.interpolate(
                    skimg_tensor.reshape(B_sky * T_sky, C_sky, H_sky, W_sky),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,).view(B_sky, T_sky, C_sky, 224, 224)
            else:
                sky_hr = skimg_tensor

            sky_patch_tokens = patchify_spatiotemporal_sky_images(
                sky_hr,
                self.sky_patch_embed,
                timefeats=skimg_timefeats[:, :, -1].unsqueeze(2),
            )
            sky_patch_tokens = self.sky_alt_attn(sky_patch_tokens)
            sky_start = skimg_timefeats[:, 0, -1]   # [B]
            sky_end   = skimg_timefeats[:, -1, -1]  # [B]
            sky_steps = torch.linspace(0, 1, 48, device=sky_patch_tokens.device)
            sky_query_times = sky_start[:, None] + (sky_end - sky_start)[:, None] * sky_steps  # [B, 48]
            sky_compressed = self.sky_two_stage_compressor(sky_patch_tokens, sky_query_times) + self.sky_mod_embed  # [B,P=48,D=64]

            sky_timefeats_48 = F.interpolate(
                    skimg_timefeats.transpose(1, 2),       # -> [B, F=3, T=30]
                    size=48,
                    mode="linear",
                    align_corners=True,                  # 头尾对齐 -> 保留首末时刻原值
                ).transpose(1, 2)

            sky_timefeats_48_hd = self.time_mlp(sky_timefeats_48)
            sky_compressed = sky_compressed + sky_timefeats_48_hd
            sky_compressed = torch.nan_to_num(sky_compressed, nan=0.0, posinf=0.0, neginf=0.0)

            sky_mask = torch.ones(B, 48, device=pv.device, dtype=pv.dtype)

        if skimg_valid_mask is not None:
            if skimg_valid_mask.dim() != 1 or skimg_valid_mask.shape[0] != B:
                raise ValueError(f"skimg_valid_mask expected [B], got {skimg_valid_mask.shape}")
            skimg_valid_mask = skimg_valid_mask.to(device=pv.device, dtype=pv.dtype).unsqueeze(1)
            sky_compressed = sky_compressed * skimg_valid_mask.unsqueeze(2)
            sky_mask = sky_mask * skimg_valid_mask

        hist_mem_compressed = torch.cat(
            [sat_compressed, sky_compressed], dim=1
        )
        key_value_mask = torch.cat([sat_mask, sky_mask], dim=1)

        # Per-sample handling of all-masked rows: an all-masked row makes the
        # attention softmax all -inf → NaN for that sample. Give those rows a
        # dummy all-valid mask to keep the math finite, then zero their
        # residual so samples without sat/sky contribute delta_kt = 0.
        row_has_valid = (key_value_mask.sum(dim=1, keepdim=True) > 0).to(pv.dtype)  # [B,1]
        safe_mask = torch.where(
            row_has_valid.bool(), key_value_mask, torch.ones_like(key_value_mask)
        )
        # Branch 2: single-horizon residual via last forecast query step.
        forecast_pv_features = self.cross_attention_pv(
            query=forecast_query[:, -1:, :],
            key=hist_mem_compressed,
            value=hist_mem_compressed,
            key_value_mask=safe_mask,
        )

        # Inverter features (embeddings)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_pv_features.shape[1], 1)

        # Fuse and predict: Branch1 TabM + Branch2 sky residual → [B, 1]
        fused = torch.cat([forecast_pv_features, inverter_features], dim=2)
        pv_feats = self.pv_feats_head(fused)
        delta_kt = self.fc(pv_feats) * row_has_valid.unsqueeze(2)  # [B,1,1]

        kt = kt_tabm.unsqueeze(1) + delta_kt  # [B,1,1]

        return kt.squeeze(-1)


# Using PV history and NWP to forecast PV, solar features and NWP features are used as query
class pv_forecasting_model_vit_dinov2(nn.Module):
    """
    Like ``pv_forecasting_model_vit_imgs``, but sky patch features come from
    DINOv2 (``SkyDINOv2PatchSpatiotemporalEmbed``) instead of a learned Conv
    patch embedder. Satellite branch is unchanged.
    """

    def __init__(
        self,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        dev_dn_list: Optional[list] = None,
        nwp_features: Optional[list[str]] = None,
        use_invalid_mask: bool = _DEFAULT_VIT_IMGS_NWP_USE_INVALID_MASK,
        nwp_dropout_prob: float = 0.0,
        nwp_history_dropout_prob: float = 0.0,
        dinov2_model_name: str = "dinov2_vits14",
        dinov2_freeze: bool = True,
        dinov2_pretrained: bool = True,
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        if nwp_features is None:
            nwp_features = list(_DEFAULT_VIT_IMGS_NWP_FEATURES)
        nwp_features = list(nwp_features)
        unknown = [n for n in nwp_features if n not in NWP_FEATURE_NORMALIZERS]
        if unknown:
            raise ValueError(
                f"Unknown NWP feature(s) {unknown}; valid features are "
                f"{sorted(NWP_FEATURE_NORMALIZERS)}"
            )
        dupes = [n for n in nwp_features if nwp_features.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate NWP feature(s) in nwp_features: {sorted(set(dupes))}")
        self.nwp_features: list[str] = nwp_features
        self.use_invalid_mask: bool = bool(use_invalid_mask)
        self.nwp_dropout_prob: float = max(0.0, min(1.0, float(nwp_dropout_prob)))
        self.nwp_history_dropout_prob: float = max(
            0.0, min(1.0, float(nwp_history_dropout_prob))
        )
        # Pre-resolve column indices in ``nwp_tensor`` (last dim = 8 features + 1 mask).
        self._nwp_feature_indices: list[int] = [
            _FOLSOM_NWP_FEATURE_COLS.index(name) for name in self.nwp_features
        ]
        # Forecast-query input dim = 3 time feats + N NWP features (+ 1 if mask passed through).
        query_mlp_in_dim = 3 + 4

        dim = 64
        self.tabm_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # TabM modality embedding
        self.pv_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # PV modality embedding
        self.sat_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Satellite image modality embedding
        self.sky_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Sky imagem odality embedding

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.pv_hist_in_channels = 9  # pv + pv_mask + pv_ramp + 3×timefeat + nwp_ghi + nwp_fc_ghi + delta_nwp
        self.nwp_future_steps = 16
        self.nwp_future_channels = 2
        self.nwp_future_flat_dim = self.nwp_future_steps * self.nwp_future_channels
        self.TCN = TemporalCNN1d(
            in_channels=self.pv_hist_in_channels,
            out_channels=64,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.corase_idx = [18, 54, 90, 126, 162, 198, 234, 270, 295, 308, 322, 335, 349, 362, 376, 389,
                           403, 416, 430, 443, 457, 470, 484, 497, 505, 508, 511, 514, 517, 520, 523, 526,
                           529, 532, 535, 538, 541, 544, 547, 550, 553, 556, 559, 562, 565, 568, 571, 574]
        # This model path uses 576-step history (derived from coarse idx settings).
        self.pv_hist_len = max(self.corase_idx) + 2
        self.learnable_pv_queries = nn.Parameter(torch.randn(1, 48, 64))
        self.cross_attention_pv_compression = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)        
        self.query_mlp = MLP(in_dim=query_mlp_in_dim, hidden_dims=(64, 64), out_dim=64, dropout=0.0)

        self.sat_embed_dim = 64
        self.sat_patch_embed = VideoPatchSpatiotemporalEmbed(
            embed_dim=self.sat_embed_dim, patch_size=16, image_size=112
        )
        self.sat_alt_attn = AlternatingIntraInterFrameAttention(
            embed_dim=self.sat_embed_dim, num_heads=8, num_cycles=4, dropout=dropout
        )

        self.sat_two_stage_compressor = SatelliteTwoStageCompressor(
                                            dim=self.sat_embed_dim,
                                            num_frames=24,
                                            num_patches=49,
                                            num_frame_queries=8,   # 196 -> 8 per frame
                                            num_sat_queries=48,    # 24*8=192 -> 48 final tokens
                                            num_heads=8,
                                            use_patch_pos=True,
                                            use_frame_time=True,
                                            use_coarse_spatial=True,
                                            coarse_query_grid_hw=(6, 8),)
        
        self.sky_embed_dim = 64
        # DINOv2 ViT-S/14 on 224×224 → 16×16 = 256 patch tokens (dim 384 → proj to 64).
        # Only the last N sky frames are used (not the full 30-frame window).
        self.sky_num_frames = 8
        self.sky_patch_embed = SkyDINOv2PatchSpatiotemporalEmbed(
            embed_dim=self.sky_embed_dim,
            image_size=224,
            model_name=dinov2_model_name,
            freeze_backbone=dinov2_freeze,
            pretrained=dinov2_pretrained,
            in_channels=4,
        )
        self.sky_alt_attn = SkyAlternatingIntraInterFrameAttention(
            embed_dim=self.sky_embed_dim, num_heads=8, num_cycles=4, dropout=dropout
        )
        self.sky_two_stage_compressor = SkyTwoStageCompressor(
                                            dim=self.sky_embed_dim,
                                            num_frames=self.sky_num_frames,
                                            num_patches=self.sky_patch_embed.num_patches,  # 256 for vits14@224
                                            num_frame_queries=8,
                                            num_sky_queries=48,
                                            num_heads=8,
                                            use_patch_pos=True,
                                            use_frame_time=True,
                                            use_coarse_spatial=True,
                                            coarse_query_grid_hw=(6, 8),)

        self.cross_attention_pv = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)  # PV 64 + sat 64 + inverter 16
        self.fc = FC(in_dim=64, out_dim=1)
        # TabM base prediction: same I/O as ``pv_forecasting_model_vit_simvp``
        # (flatten pv + pv_ramp over 576 steps → 1152 features).
        self.pv_tabm_head = TabM(
            n_num_features=1152,
            cat_cardinalities=None,
            d_out=1,
            n_blocks=3,
            d_block=512,
            dropout=dropout,
            k=48,
            arch_type="tabm",
            start_scaling_init="normal",
        )
        # Cached feature right before TabM final output projection.
        self.pv_tabm_preoutput_features: Optional[torch.Tensor] = None
        self._pv_tabm_output_pre_hook_handle = self.pv_tabm_head.output.register_forward_pre_hook(
            self._capture_pv_tabm_preoutput_features
        )

    @staticmethod
    def _apply_channel_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if (not training) or p <= 0.0:
            return x
        keep = 1.0 - float(p)
        if keep <= 0.0:
            return torch.zeros_like(x)
        bsz, _, channels = x.shape
        mask = (torch.rand((bsz, 1, channels), device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def _capture_pv_tabm_preoutput_features(self, module: nn.Module, inputs: tuple) -> None:
        _ = module
        if not inputs:
            self.pv_tabm_preoutput_features = None
            return
        x = inputs[0]
        self.pv_tabm_preoutput_features = x if isinstance(x, torch.Tensor) else None

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor, 
                pv_mask: Optional[torch.Tensor] = None, 
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None,
                nwp_history: Optional[torch.Tensor] = None,
                nwp_forecast_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        pv_masked = pv * pv_mask.to(pv.dtype)
        pv_ramp = pv_masked[:, :, 0:-1] - pv_masked[:, :, 1:]
        pv_ramp = torch.cat([torch.zeros_like(pv_ramp[:, :, :1]), pv_ramp], dim=2)

        pv_timefeats = pv_timefeats[:, :, [2,3,8]]
        forecast_timefeats = forecast_timefeats[:, :, [2,3,8]]

        # Read history features for PV branch; if missing, fallback to zeros.
        if nwp_history is None:
            nwp_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_history_ghi = ((nwp_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        if nwp_forecast_history is None:
            nwp_forecast_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_forecast_history_ghi = ((nwp_forecast_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        delta_nwp_history_ghi = nwp_history_ghi - nwp_forecast_history_ghi

        pv_history = torch.cat(
            [
                pv_masked,
                pv_mask,
                pv_ramp,
                pv_timefeats.permute(0, 2, 1),
                nwp_history_ghi,
                nwp_forecast_history_ghi,
                delta_nwp_history_ghi,
            ],
            dim=1,
        )  # [B, C=9, T]  — same layout as vit_simvp

        nwp_ssrd_normalized = (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0
        nwp_t2m_normalized = (nwp_tensor[:, :, 2] - 288.15) / 10.0

        # TabM input identical to vit_simvp: only pv + pv_ramp (channels 0, 2)
        pv_hist_flat = pv_history[:, [0, 2], :].reshape(pv_history.shape[0], -1)
        nwp_tensor_for_tabm = torch.stack(
            [
                nwp_ssrd_normalized,
                nwp_t2m_normalized,
            ],
            dim=2,
        )
        nwp_tensor_flat = nwp_tensor_for_tabm.reshape(nwp_tensor_for_tabm.shape[0], -1)
        cur_nwp_dim = nwp_tensor_flat.shape[1]
        if cur_nwp_dim < self.nwp_future_flat_dim:
            pad = torch.zeros(
                nwp_tensor_flat.shape[0],
                self.nwp_future_flat_dim - cur_nwp_dim,
                device=nwp_tensor_flat.device,
                dtype=nwp_tensor_flat.dtype,
            )
            nwp_tensor_flat = torch.cat([nwp_tensor_flat, pad], dim=1)
        elif cur_nwp_dim > self.nwp_future_flat_dim:
            nwp_tensor_flat = nwp_tensor_flat[:, : self.nwp_future_flat_dim]

        x_num = pv_hist_flat

        self.pv_tabm_preoutput_features = None
        tabm_out = self.pv_tabm_head(x_num=x_num, x_cat=None)  # [B, K, 1]
        kt_tabm = tabm_out.mean(dim=1)  # [B, 1]

        # return kt_tabm  # [B, 1] — keep dim so pv_pred = kt * target_p_cs * p_mean is [B, 1]

        # Forecast queries: Luoyang total variant only uses two NWP channels:
        # ssrd (idx=0) and t2m (idx=2), where nwp_tensor layout is
        # [ssrd, msl, t2m, u10, v10, u100, v100, nan_mask].

        nwp_channels = [forecast_timefeats]
        if nwp_tensor is not None:
            nwp_feats = torch.stack(
                [
                    (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0,
                    nwp_tensor[:, :, 1],
                    (nwp_tensor[:, :, 2] - 288.15) / 10.0,
                    nwp_tensor[:, :, 3],
                ],
                dim=2,
            )
            nwp_feats = self._apply_channel_dropout(
                nwp_feats,
                p=self.nwp_dropout_prob,
                training=self.training,
            )
            nwp_channels.append(nwp_feats)
        else:
            zero_nwp_feat = torch.zeros(
                forecast_timefeats.shape[:2],
                device=forecast_timefeats.device,
                dtype=forecast_timefeats.dtype,
            )
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # ssrd
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # t2m
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # ssrd
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # t2m

        forecast_ssrd_timefeats = torch.cat(nwp_channels, dim=2)
        
        forecast_query = self.query_mlp(forecast_ssrd_timefeats)

        # satellite images encoder
        B, T_out, _ = forecast_query.shape

        if sat_tensor is None or sat_tensor.max() == 0:
            sat_compressed = torch.zeros(B, 48, 64, device=pv.device, dtype=pv.dtype) + self.sat_mod_embed
            sat_mask = torch.zeros(B, 48, device=pv.device, dtype=pv.dtype)
        else:
            B_sat, T_sat, C_sat, H_sat, W_sat = sat_tensor.shape
            if C_sat != 3:
                raise ValueError(f"sat_tensor expected 3 channels, got {C_sat}")
            sat_hr = nn.functional.interpolate(
                sat_tensor.reshape(B_sat * T_sat, C_sat, H_sat, W_sat),
                size=(112, 112),
                mode="bilinear",
                align_corners=False,
            ).view(B_sat, T_sat, 3, 112, 112)

            sat_timefeats = sat_timefeats[:, :, [2,3,8]]
            sat_patch_tokens = patchify_spatiotemporal_images(sat_hr, self.sat_patch_embed, timefeats=sat_timefeats[:,:,-1].unsqueeze(2))
            sat_patch_tokens = self.sat_alt_attn(sat_patch_tokens)  # [B,T=24,P=49,D=64]
            sat_start = sat_timefeats[:, 0, -1]   # [B]
            sat_end   = sat_timefeats[:, -1, -1]  # [B]
            sat_steps = torch.linspace(0, 1, 48, device=sat_patch_tokens.device)
            sat_query_times = sat_start[:, None] + (sat_end - sat_start)[:, None] * sat_steps  # [B, 48]
            sat_compressed = self.sat_two_stage_compressor(sat_patch_tokens, sat_query_times) + self.sat_mod_embed  # [B,P=48,D=64]

            sat_timefeats_48 = F.interpolate(
                    sat_timefeats.transpose(1, 2),       # -> [B, F=3, T=24]  (interp 要求 [N, C, L])
                    size=48,
                    mode="linear",
                    align_corners=True,                  # 头尾对齐 -> 保留首末时刻原值
                ).transpose(1, 2)

            sat_timefeats_48_hd = self.time_mlp(sat_timefeats_48)
            sat_compressed = sat_compressed + sat_timefeats_48_hd
            
            # print('sat_patch_tokens: ', sat_patch_tokens.shape, sat_compressed.shape)
            sat_mask = torch.ones(B, sat_compressed.shape[1], device=pv.device, dtype=pv.dtype)

        if sat_valid_mask is not None:
            if sat_valid_mask.dim() != 1 or sat_valid_mask.shape[0] != B:
                raise ValueError(f"sat_valid_mask expected [B], got {sat_valid_mask.shape}")
            sat_valid_mask = sat_valid_mask.to(device=pv.device, dtype=pv.dtype).unsqueeze(1)
            sat_compressed = sat_compressed * sat_valid_mask.unsqueeze(2)
            sat_mask = sat_mask * sat_valid_mask

        # sky images encoder (DINOv2): use only the last sky_num_frames frames
        if skimg_tensor is None or skimg_tensor.max() == 0:
            sky_compressed = torch.zeros(B, 48, 64, device=pv.device, dtype=pv.dtype) + self.sky_mod_embed
            sky_mask = torch.zeros(B, 48, device=pv.device, dtype=pv.dtype)
        else:
            B_sky, T_sky, C_sky, H_sky, W_sky = skimg_tensor.shape
            if C_sky < 3:
                raise ValueError(f"skimg_tensor expected at least 3 channels, got {C_sky}")
            n_sky = min(self.sky_num_frames, T_sky)
            skimg_tensor = skimg_tensor[:, -n_sky:]
            skimg_timefeats = skimg_timefeats[:, -n_sky:, [2, 3, 8]]
            B_sky, T_sky, C_sky, H_sky, W_sky = skimg_tensor.shape
            if H_sky != 224 or W_sky != 224:
                sky_hr = nn.functional.interpolate(
                    skimg_tensor.reshape(B_sky * T_sky, C_sky, H_sky, W_sky),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,).view(B_sky, T_sky, C_sky, 224, 224)
            else:
                sky_hr = skimg_tensor

            sky_patch_tokens = patchify_spatiotemporal_sky_images(
                sky_hr,
                self.sky_patch_embed,
                timefeats=skimg_timefeats[:, :, -1].unsqueeze(2),
            )
            sky_patch_tokens = self.sky_alt_attn(sky_patch_tokens)
            sky_start = skimg_timefeats[:, 0, -1]   # [B]
            sky_end   = skimg_timefeats[:, -1, -1]  # [B]
            sky_steps = torch.linspace(0, 1, 48, device=sky_patch_tokens.device)
            sky_query_times = sky_start[:, None] + (sky_end - sky_start)[:, None] * sky_steps  # [B, 48]
            sky_compressed = self.sky_two_stage_compressor(sky_patch_tokens, sky_query_times) + self.sky_mod_embed  # [B,P=48,D=64]

            sky_timefeats_48 = F.interpolate(
                    skimg_timefeats.transpose(1, 2),       # -> [B, F=3, T=8]
                    size=48,
                    mode="linear",
                    align_corners=True,                  # 头尾对齐 -> 保留首末时刻原值
                ).transpose(1, 2)

            sky_timefeats_48_hd = self.time_mlp(sky_timefeats_48)
            sky_compressed = sky_compressed + sky_timefeats_48_hd
            sky_compressed = torch.nan_to_num(sky_compressed, nan=0.0, posinf=0.0, neginf=0.0)

            sky_mask = torch.ones(B, 48, device=pv.device, dtype=pv.dtype)

        if skimg_valid_mask is not None:
            if skimg_valid_mask.dim() != 1 or skimg_valid_mask.shape[0] != B:
                raise ValueError(f"skimg_valid_mask expected [B], got {skimg_valid_mask.shape}")
            skimg_valid_mask = skimg_valid_mask.to(device=pv.device, dtype=pv.dtype).unsqueeze(1)
            sky_compressed = sky_compressed * skimg_valid_mask.unsqueeze(2)
            sky_mask = sky_mask * skimg_valid_mask

        # tabm_mask = torch.ones(B, tabm_summary_token.shape[1], device=pv.device, dtype=pv.dtype)
        hist_mem_compressed = torch.cat(
            [sat_compressed, sky_compressed], dim=1
        )
        key_value_mask = torch.cat([sat_mask, sky_mask], dim=1)

        # Per-sample handling of all-masked rows: an all-masked row makes the
        # attention softmax all -inf → NaN for that sample. Give those rows a
        # dummy all-valid mask to keep the math finite, then zero their
        # residual so samples without sat/sky contribute delta_kt = 0.
        row_has_valid = (key_value_mask.sum(dim=1, keepdim=True) > 0).to(pv.dtype)  # [B,1]
        safe_mask = torch.where(
            row_has_valid.bool(), key_value_mask, torch.ones_like(key_value_mask)
        )
        forecast_pv_features = self.cross_attention_pv(query=forecast_query[:,-1:,:], key=hist_mem_compressed, value=hist_mem_compressed, key_value_mask=safe_mask)

        # Inverter features (embeddings)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_pv_features.shape[1], 1)

        # Fuse and predict
        fused = torch.cat([forecast_pv_features, inverter_features], dim=2)
        pv_feats = self.pv_feats_head(fused)
        delta_kt = self.fc(pv_feats) * row_has_valid.unsqueeze(2)  # [B,1,1]

        kt = kt_tabm.unsqueeze(1) + delta_kt # [B,1,1]

        return kt.squeeze(-1)


class pv_forecasting_model_vit_pvb(nn.Module):
    """
    TabM base prediction + PV-history residual branch only.

    Same as ``pv_forecasting_model_vit_imgs`` but with all satellite and sky
    image encoders removed.  The residual ``delta_kt`` is produced by
    cross-attending the forecast query over compressed PV-history tokens
    (TCN → coarse-frame selection → learnable cross-attention compression).
    """

    def __init__(
        self,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
        dev_dn_list: Optional[list] = None,
        nwp_features: Optional[list[str]] = None,
        use_invalid_mask: bool = _DEFAULT_VIT_IMGS_NWP_USE_INVALID_MASK,
        nwp_dropout_prob: float = 0.0,
        nwp_history_dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        if nwp_features is None:
            nwp_features = list(_DEFAULT_VIT_IMGS_NWP_FEATURES)
        nwp_features = list(nwp_features)
        unknown = [n for n in nwp_features if n not in NWP_FEATURE_NORMALIZERS]
        if unknown:
            raise ValueError(
                f"Unknown NWP feature(s) {unknown}; valid features are "
                f"{sorted(NWP_FEATURE_NORMALIZERS)}"
            )
        dupes = [n for n in nwp_features if nwp_features.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate NWP feature(s) in nwp_features: {sorted(set(dupes))}")
        self.nwp_features: list[str] = nwp_features
        self.use_invalid_mask: bool = bool(use_invalid_mask)
        self.nwp_dropout_prob: float = max(0.0, min(1.0, float(nwp_dropout_prob)))
        self.nwp_history_dropout_prob: float = max(
            0.0, min(1.0, float(nwp_history_dropout_prob))
        )
        self._nwp_feature_indices: list[int] = [
            _FOLSOM_NWP_FEATURE_COLS.index(name) for name in self.nwp_features
        ]
        query_mlp_in_dim = 3 + 4

        dim = 64
        self.pv_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.pv_hist_in_channels = 9  # pv + pv_mask + pv_ramp + 3×timefeat + nwp_ghi + nwp_fc_ghi + delta_nwp
        self.nwp_future_steps = 16
        self.nwp_future_channels = 2
        self.nwp_future_flat_dim = self.nwp_future_steps * self.nwp_future_channels
        self.TCN = TemporalCNN1d(
            in_channels=self.pv_hist_in_channels,
            out_channels=64,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.corase_idx = [18, 54, 90, 126, 162, 198, 234, 270, 295, 308, 322, 335, 349, 362, 376, 389,
                           403, 416, 430, 443, 457, 470, 484, 497, 505, 508, 511, 514, 517, 520, 523, 526,
                           529, 532, 535, 538, 541, 544, 547, 550, 553, 556, 559, 562, 565, 568, 571, 574]
        self.pv_hist_len = max(self.corase_idx) + 2
        self.learnable_pv_queries = nn.Parameter(torch.randn(1, 48, 64))
        self.cross_attention_pv_compression = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.query_mlp = MLP(in_dim=query_mlp_in_dim, hidden_dims=(64, 64), out_dim=64, dropout=0.0)

        self.cross_attention_pv = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)  # pv 64 + inverter 16
        self.fc = FC(in_dim=64, out_dim=1)

        self.pv_tabm_head = TabM(
            n_num_features=1152,
            cat_cardinalities=None,
            d_out=1,
            n_blocks=3,
            d_block=512,
            dropout=dropout,
            k=48,
            arch_type="tabm",
            start_scaling_init="normal",
        )
        self.pv_tabm_preoutput_features: Optional[torch.Tensor] = None
        self._pv_tabm_output_pre_hook_handle = self.pv_tabm_head.output.register_forward_pre_hook(
            self._capture_pv_tabm_preoutput_features
        )

    @staticmethod
    def _apply_channel_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if (not training) or p <= 0.0:
            return x
        keep = 1.0 - float(p)
        if keep <= 0.0:
            return torch.zeros_like(x)
        bsz, _, channels = x.shape
        mask = (torch.rand((bsz, 1, channels), device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def _capture_pv_tabm_preoutput_features(self, module: nn.Module, inputs: tuple) -> None:
        _ = module
        if not inputs:
            self.pv_tabm_preoutput_features = None
            return
        x = inputs[0]
        self.pv_tabm_preoutput_features = x if isinstance(x, torch.Tensor) else None

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor,
                pv_mask: Optional[torch.Tensor] = None,
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None,
                nwp_history: Optional[torch.Tensor] = None,
                nwp_forecast_history: Optional[torch.Tensor] = None) -> torch.Tensor:

        pv_masked = pv * pv_mask.to(pv.dtype)
        pv_ramp = pv_masked[:, :, 0:-1] - pv_masked[:, :, 1:]
        pv_ramp = torch.cat([torch.zeros_like(pv_ramp[:, :, :1]), pv_ramp], dim=2)

        pv_timefeats = pv_timefeats[:, :, [2, 3, 8]]
        forecast_timefeats = forecast_timefeats[:, :, [2, 3, 8]]

        if nwp_history is None:
            nwp_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_history_ghi = ((nwp_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        if nwp_forecast_history is None:
            nwp_forecast_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_forecast_history_ghi = ((nwp_forecast_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        delta_nwp_history_ghi = nwp_history_ghi - nwp_forecast_history_ghi

        pv_history = torch.cat(
            [
                pv_masked,
                pv_mask,
                pv_ramp,
                pv_timefeats.permute(0, 2, 1),
                nwp_history_ghi,
                nwp_forecast_history_ghi,
                delta_nwp_history_ghi,
            ],
            dim=1,
        )  # [B, C=9, T]

        nwp_ssrd_normalized = (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0
        nwp_t2m_normalized = (nwp_tensor[:, :, 2] - 288.15) / 10.0

        pv_hist_flat = pv_history[:, [0, 2], :].reshape(pv_history.shape[0], -1)
        nwp_tensor_for_tabm = torch.stack([nwp_ssrd_normalized, nwp_t2m_normalized], dim=2)
        nwp_tensor_flat = nwp_tensor_for_tabm.reshape(nwp_tensor_for_tabm.shape[0], -1)
        cur_nwp_dim = nwp_tensor_flat.shape[1]
        if cur_nwp_dim < self.nwp_future_flat_dim:
            pad = torch.zeros(
                nwp_tensor_flat.shape[0],
                self.nwp_future_flat_dim - cur_nwp_dim,
                device=nwp_tensor_flat.device,
                dtype=nwp_tensor_flat.dtype,
            )
            nwp_tensor_flat = torch.cat([nwp_tensor_flat, pad], dim=1)
        elif cur_nwp_dim > self.nwp_future_flat_dim:
            nwp_tensor_flat = nwp_tensor_flat[:, : self.nwp_future_flat_dim]

        x_num = pv_hist_flat

        self.pv_tabm_preoutput_features = None
        tabm_out = self.pv_tabm_head(x_num=x_num, x_cat=None)  # [B, K, 1]
        kt_tabm = tabm_out.mean(dim=1)  # [B, 1]

        # Forecast query from NWP + time features
        nwp_channels = [forecast_timefeats]
        if nwp_tensor is not None:
            nwp_feats = torch.stack(
                [
                    (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0,
                    nwp_tensor[:, :, 1],
                    (nwp_tensor[:, :, 2] - 288.15) / 10.0,
                    nwp_tensor[:, :, 3],
                ],
                dim=2,
            )
            nwp_feats = self._apply_channel_dropout(nwp_feats, p=self.nwp_dropout_prob, training=self.training)
            nwp_channels.append(nwp_feats)
        else:
            zero_nwp_feat = torch.zeros(
                forecast_timefeats.shape[:2], device=forecast_timefeats.device, dtype=forecast_timefeats.dtype
            )
            for _ in range(4):
                nwp_channels.append(zero_nwp_feat.unsqueeze(2))

        forecast_query = self.query_mlp(torch.cat(nwp_channels, dim=2))  # [B, T_out, 64]
        B = forecast_query.shape[0]

        # PV history branch: TCN → coarse-frame selection → cross-attention compression
        pv_tcn_out = self.TCN(pv_history)  # [B, 64, T]
        pv_coarse = pv_tcn_out[:, :, self.corase_idx].permute(0, 2, 1)  # [B, 48, 64]
        pv_queries = self.learnable_pv_queries.expand(B, -1, -1)         # [B, 48, 64]
        pv_compressed = self.cross_attention_pv_compression(
            query=pv_queries, key=pv_coarse, value=pv_coarse
        ) + self.pv_mod_embed  # [B, 48, 64]

        # Cross-attend forecast query over PV tokens → residual delta_kt
        forecast_pv_features = self.cross_attention_pv(
            query=forecast_query[:, -1:, :], key=pv_compressed, value=pv_compressed
        )  # [B, 1, 64]

        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_pv_features.shape[1], 1)
        fused = torch.cat([forecast_pv_features, inverter_features], dim=2)  # [B, 1, 80]
        pv_feats = self.pv_feats_head(fused)
        delta_kt = self.fc(pv_feats)  # [B, 1, 1]; PV always valid

        kt = kt_tabm.unsqueeze(1) + delta_kt  # [B, 1, 1]
        return kt.squeeze(-1)


class pv_forecasting_model_vit_simvp(nn.Module):
    """
    Same as ``pv_forecasting_model_vit_pvb`` (TabM base prediction + PV-history
    residual branch), plus a SimVP-derived spatiotemporal module on the
    historical sky images (``skimg_tensor``).

    The SimVP module (Encoder + Translator, no frame decoder; see
    https://github.com/A4Bio/SimVP) outputs future latent features
    ``sky_future_feats`` of shape ``[B, out_frames, hid_S, H', W']`` (default
    ``out_frames=1``: a single future feature frame).  How these features
    are consumed downstream is left to be implemented.
    """

    def __init__(
        self,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
        dev_dn_list: Optional[list] = None,
        nwp_features: Optional[list[str]] = None,
        use_invalid_mask: bool = _DEFAULT_VIT_IMGS_NWP_USE_INVALID_MASK,
        nwp_dropout_prob: float = 0.0,
        nwp_history_dropout_prob: float = 0.0,
        simvp_in_frames: int = 8,
        simvp_in_channels: int = 4,  # sky images: RGB + asi_mask
        simvp_out_frames: int = 1,   # single future feature frame (t0 + 15 min)
        simvp_hid_S: int = 16,
        simvp_hid_T: int = 256,
        simvp_N_S: int = 4,
        simvp_N_T: int = 8,
        simvp_img_size: Optional[int] = None,  # downsample sky images to this HxW before SimVP (None = no resize)
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.simvp_in_frames: int = int(simvp_in_frames)
        self.simvp_img_size: Optional[int] = int(simvp_img_size) if simvp_img_size is not None else None

        if nwp_features is None:
            nwp_features = list(_DEFAULT_VIT_IMGS_NWP_FEATURES)
        nwp_features = list(nwp_features)
        unknown = [n for n in nwp_features if n not in NWP_FEATURE_NORMALIZERS]
        if unknown:
            raise ValueError(
                f"Unknown NWP feature(s) {unknown}; valid features are "
                f"{sorted(NWP_FEATURE_NORMALIZERS)}"
            )
        dupes = [n for n in nwp_features if nwp_features.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate NWP feature(s) in nwp_features: {sorted(set(dupes))}")
        self.nwp_features: list[str] = nwp_features
        self.use_invalid_mask: bool = bool(use_invalid_mask)
        self.nwp_dropout_prob: float = max(0.0, min(1.0, float(nwp_dropout_prob)))
        self.nwp_history_dropout_prob: float = max(
            0.0, min(1.0, float(nwp_history_dropout_prob))
        )
        self._nwp_feature_indices: list[int] = [
            _FOLSOM_NWP_FEATURE_COLS.index(name) for name in self.nwp_features
        ]
        query_mlp_in_dim = 3 + 4

        dim = 64
        self.pv_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.pv_hist_in_channels = 9  # pv + pv_mask + pv_ramp + 3×timefeat + nwp_ghi + nwp_fc_ghi + delta_nwp
        self.nwp_future_steps = 16
        self.nwp_future_channels = 2
        self.nwp_future_flat_dim = self.nwp_future_steps * self.nwp_future_channels
        self.TCN = TemporalCNN1d(
            in_channels=self.pv_hist_in_channels,
            out_channels=64,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.corase_idx = [18, 54, 90, 126, 162, 198, 234, 270, 295, 308, 322, 335, 349, 362, 376, 389,
                           403, 416, 430, 443, 457, 470, 484, 497, 505, 508, 511, 514, 517, 520, 523, 526,
                           529, 532, 535, 538, 541, 544, 547, 550, 553, 556, 559, 562, 565, 568, 571, 574]
        self.pv_hist_len = max(self.corase_idx) + 2
        self.learnable_pv_queries = nn.Parameter(torch.randn(1, 48, 64))
        self.cross_attention_pv_compression = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.query_mlp = MLP(in_dim=query_mlp_in_dim, hidden_dims=(64, 64), out_dim=64, dropout=0.0)

        self.cross_attention_pv = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)  # pv 64 + inverter 16
        self.fc = FC(in_dim=64, out_dim=1)

        # SimVP-derived module on historical satellite frames:
        # Encoder + Translator only, outputs future latent features (no frame decoder).
        self.simvp = SimVPFeatureExtractor(
            in_frames=simvp_in_frames,
            in_channels=simvp_in_channels,
            out_frames=simvp_out_frames,
            hid_S=simvp_hid_S,
            hid_T=simvp_hid_T,
            N_S=simvp_N_S,
            N_T=simvp_N_T,
        )
        # Conv head over the future sky feature map → residual delta_kt_sky.
        # 56 → 28 → 14 → 7 → global average pool.
        self.sky_head = nn.Sequential(
            nn.Conv2d(simvp_hid_S, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.sky_fc = FC(in_dim=64, out_dim=1)
        # Zero-init so the sky branch contributes nothing at the start of
        # training; TabM + PV branches converge first, sky residual ramps up.
        nn.init.zeros_(self.sky_fc.fc.weight)
        nn.init.zeros_(self.sky_fc.fc.bias)

        self.pv_tabm_head = TabM(
            n_num_features=1152,
            cat_cardinalities=None,
            d_out=1,
            n_blocks=3,
            d_block=512,
            dropout=dropout,
            k=48,
            arch_type="tabm",
            start_scaling_init="normal",
        )
        self.pv_tabm_preoutput_features: Optional[torch.Tensor] = None
        self._pv_tabm_output_pre_hook_handle = self.pv_tabm_head.output.register_forward_pre_hook(
            self._capture_pv_tabm_preoutput_features
        )

    @staticmethod
    def _apply_channel_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if (not training) or p <= 0.0:
            return x
        keep = 1.0 - float(p)
        if keep <= 0.0:
            return torch.zeros_like(x)
        bsz, _, channels = x.shape
        mask = (torch.rand((bsz, 1, channels), device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def _capture_pv_tabm_preoutput_features(self, module: nn.Module, inputs: tuple) -> None:
        _ = module
        if not inputs:
            self.pv_tabm_preoutput_features = None
            return
        x = inputs[0]
        self.pv_tabm_preoutput_features = x if isinstance(x, torch.Tensor) else None

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor,
                pv_mask: Optional[torch.Tensor] = None,
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None,
                nwp_history: Optional[torch.Tensor] = None,
                nwp_forecast_history: Optional[torch.Tensor] = None) -> torch.Tensor:

        pv_masked = pv * pv_mask.to(pv.dtype)
        pv_ramp = pv_masked[:, :, 0:-1] - pv_masked[:, :, 1:]
        pv_ramp = torch.cat([torch.zeros_like(pv_ramp[:, :, :1]), pv_ramp], dim=2)

        pv_timefeats = pv_timefeats[:, :, [2, 3, 8]]
        forecast_timefeats = forecast_timefeats[:, :, [2, 3, 8]]

        if nwp_history is None:
            nwp_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_history_ghi = ((nwp_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        if nwp_forecast_history is None:
            nwp_forecast_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_forecast_history_ghi = ((nwp_forecast_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        delta_nwp_history_ghi = nwp_history_ghi - nwp_forecast_history_ghi

        pv_history = torch.cat(
            [
                pv_masked,
                pv_mask,
                pv_ramp,
                pv_timefeats.permute(0, 2, 1),
                nwp_history_ghi,
                nwp_forecast_history_ghi,
                delta_nwp_history_ghi,
            ],
            dim=1,
        )  # [B, C=9, T]

        nwp_ssrd_normalized = (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0
        nwp_t2m_normalized = (nwp_tensor[:, :, 2] - 288.15) / 10.0

        pv_hist_flat = pv_history[:, [0, 2], :].reshape(pv_history.shape[0], -1)
        nwp_tensor_for_tabm = torch.stack([nwp_ssrd_normalized, nwp_t2m_normalized], dim=2)
        nwp_tensor_flat = nwp_tensor_for_tabm.reshape(nwp_tensor_for_tabm.shape[0], -1)
        cur_nwp_dim = nwp_tensor_flat.shape[1]
        if cur_nwp_dim < self.nwp_future_flat_dim:
            pad = torch.zeros(
                nwp_tensor_flat.shape[0],
                self.nwp_future_flat_dim - cur_nwp_dim,
                device=nwp_tensor_flat.device,
                dtype=nwp_tensor_flat.dtype,
            )
            nwp_tensor_flat = torch.cat([nwp_tensor_flat, pad], dim=1)
        elif cur_nwp_dim > self.nwp_future_flat_dim:
            nwp_tensor_flat = nwp_tensor_flat[:, : self.nwp_future_flat_dim]

        x_num = pv_hist_flat

        self.pv_tabm_preoutput_features = None
        tabm_out = self.pv_tabm_head(x_num=x_num, x_cat=None)  # [B, K, 1]
        kt_tabm = tabm_out.mean(dim=1)  # [B, 1]

        '''
        # Forecast query from NWP + time features
        nwp_channels = [forecast_timefeats]
        if nwp_tensor is not None:
            nwp_feats = torch.stack(
                [
                    (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0,
                    nwp_tensor[:, :, 1],
                    (nwp_tensor[:, :, 2] - 288.15) / 10.0,
                    nwp_tensor[:, :, 3],
                ],
                dim=2,
            )
            nwp_feats = self._apply_channel_dropout(nwp_feats, p=self.nwp_dropout_prob, training=self.training)
            nwp_channels.append(nwp_feats)
        else:
            zero_nwp_feat = torch.zeros(
                forecast_timefeats.shape[:2], device=forecast_timefeats.device, dtype=forecast_timefeats.dtype
            )
            for _ in range(4):
                nwp_channels.append(zero_nwp_feat.unsqueeze(2))

        forecast_query = self.query_mlp(torch.cat(nwp_channels, dim=2))  # [B, T_out, 64]
        B = forecast_query.shape[0]

        # SimVP branch: historical sky images → future latent features at t0+15min
        # skimg_tensor: [B, T_sky, 4, H, W] → sky_future_feats: [B, out_frames=1, hid_S, H', W']
        # Select the last simvp_in_frames frames with step=2 (every other frame), including the last frame.
        # e.g. simvp_in_frames=8 → slice [-15::2] gives indices -15,-13,...,-1 (8 frames).
        frame_start = -(self.simvp_in_frames * 2 - 1)
        sky_frames = skimg_tensor[:, frame_start::2, :, :, :]  # [B, simvp_in_frames, C, H, W]
        if self.simvp_img_size is not None:
            B_s, T_s, C_s, H_s, W_s = sky_frames.shape
            sky_frames = torch.nn.functional.interpolate(
                sky_frames.reshape(B_s * T_s, C_s, H_s, W_s),
                size=(self.simvp_img_size, self.simvp_img_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(B_s, T_s, C_s, self.simvp_img_size, self.simvp_img_size)
        sky_future_feats = self.simvp(sky_frames)

        sky_pooled = self.sky_head(sky_future_feats[:, 0]).flatten(1)  # [B, 64]
        delta_kt_sky = self.sky_fc(sky_pooled).unsqueeze(1)  # [B, 1, 1]
        # Mask out samples without sky images (their skimg_tensor is all zeros,
        # which would still produce a spurious feature through conv biases).
        if skimg_valid_mask is not None:
            delta_kt_sky = delta_kt_sky * skimg_valid_mask.to(delta_kt_sky.dtype).view(-1, 1, 1)
        '''

        kt = kt_tabm.unsqueeze(1) # + delta_kt_sky
        return kt.squeeze(-1)


# Using PV history and NWP to forecast PV, solar features and NWP features are used as query
class pv_forecasting_model_tabm_FiLM(nn.Module):
    """
    Like ``pv_forecasting_model_vit_nwp`` but satellite frames go through
    :func:`modules.SatEncoder.patchify_spatiotemporal_images` and
    :class:`modules.SatEncoder.AlternatingIntraInterFrameAttention`, then cross-attend into the forecast query.

    NWP forecast-query channels are configurable: ``nwp_features`` selects which columns
    of ``_FOLSOM_NWP_FEATURE_COLS`` to read from ``nwp_tensor`` (each normalised via
    :data:`NWP_FEATURE_NORMALIZERS`), and ``use_invalid_mask`` toggles passing the trailing
    per-step invalid mask channel (``nwp_tensor[:, :, -1]``) through as-is.
    """

    def __init__(
        self,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        dev_dn_list: Optional[list] = None,
        nwp_features: Optional[list[str]] = None,
        use_invalid_mask: bool = _DEFAULT_VIT_IMGS_NWP_USE_INVALID_MASK,
        nwp_dropout_prob: float = 0.0,
        nwp_history_dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        if nwp_features is None:
            nwp_features = list(_DEFAULT_VIT_IMGS_NWP_FEATURES)
        nwp_features = list(nwp_features)
        unknown = [n for n in nwp_features if n not in NWP_FEATURE_NORMALIZERS]
        if unknown:
            raise ValueError(
                f"Unknown NWP feature(s) {unknown}; valid features are "
                f"{sorted(NWP_FEATURE_NORMALIZERS)}"
            )
        dupes = [n for n in nwp_features if nwp_features.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate NWP feature(s) in nwp_features: {sorted(set(dupes))}")
        self.nwp_features: list[str] = nwp_features
        self.use_invalid_mask: bool = bool(use_invalid_mask)
        self.nwp_dropout_prob: float = max(0.0, min(1.0, float(nwp_dropout_prob)))
        self.nwp_history_dropout_prob: float = max(
            0.0, min(1.0, float(nwp_history_dropout_prob))
        )
        # Pre-resolve column indices in ``nwp_tensor`` (last dim = 8 features + 1 mask).
        self._nwp_feature_indices: list[int] = [
            _FOLSOM_NWP_FEATURE_COLS.index(name) for name in self.nwp_features
        ]
        # Forecast-query input dim = 3 time feats + N NWP features (+ 1 if mask passed through).
        query_mlp_in_dim = 3 + 4

        dim = 64
        self.tabm_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # TabM modality embedding
        self.pv_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # PV modality embedding
        self.sat_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Satellite image modality embedding
        self.sky_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Sky imagem odality embedding

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.pv_hist_in_channels = 8
        self.nwp_future_steps = 16
        self.nwp_future_channels = 2
        self.nwp_future_flat_dim = self.nwp_future_steps * self.nwp_future_channels
        self.TCN = TemporalCNN1d(
            in_channels=self.pv_hist_in_channels,
            out_channels=64,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.corase_idx = [18, 54, 90, 126, 162, 198, 234, 270, 295, 308, 322, 335, 349, 362, 376, 389,
                           403, 416, 430, 443, 457, 470, 484, 497, 505, 508, 511, 514, 517, 520, 523, 526,
                           529, 532, 535, 538, 541, 544, 547, 550, 553, 556, 559, 562, 565, 568, 571, 574]
        # This model path uses 576-step history (derived from coarse idx settings).
        self.pv_hist_len = max(self.corase_idx) + 2
        self.learnable_pv_queries = nn.Parameter(torch.randn(1, 48, 64))
        self.cross_attention_pv_compression = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)        
        self.query_mlp = MLP(in_dim=query_mlp_in_dim, hidden_dims=(64, 64), out_dim=64, dropout=0.0)

        self.sat_embed_dim = 64
        self.sat_patch_embed = VideoPatchSpatiotemporalEmbed(
            embed_dim=self.sat_embed_dim, patch_size=16, image_size=112
        )
        self.sat_alt_attn = AlternatingIntraInterFrameAttention(
            embed_dim=self.sat_embed_dim, num_heads=8, num_cycles=4, dropout=dropout
        )

        self.sat_two_stage_compressor = SatelliteTwoStageCompressor(
                                            dim=self.sat_embed_dim,
                                            num_frames=24,
                                            num_patches=49,
                                            num_frame_queries=8,   # 196 -> 8 per frame
                                            num_sat_queries=48,    # 24*8=192 -> 48 final tokens
                                            num_heads=8,
                                            use_patch_pos=True,
                                            use_frame_time=True,
                                            use_coarse_spatial=True,
                                            coarse_query_grid_hw=(6, 8),)
        
        self.sky_embed_dim = 64
        self.sky_patch_embed = SkyPatchSpatiotemporalEmbed(
            embed_dim=self.sky_embed_dim, patch_size=16, image_size=224, in_channels=4
        )
        self.sky_alt_attn = SkyAlternatingIntraInterFrameAttention(
            embed_dim=self.sky_embed_dim, num_heads=8, num_cycles=4, dropout=dropout
        )
        self.sky_two_stage_compressor = SkyTwoStageCompressor(
                                            dim=self.sky_embed_dim,
                                            num_frames=30,
                                            num_patches=196,
                                            num_frame_queries=8,
                                            num_sky_queries=48,
                                            num_heads=8,
                                            use_patch_pos=True,
                                            use_frame_time=True,
                                            use_coarse_spatial=True,
                                            coarse_query_grid_hw=(6, 8),)

        self.cross_attention_pv = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)  # PV 64 + sat 64 + inverter 16
        self.fc = FC(in_dim=64, out_dim=1)
        # FiLM generator: tabm_summary_feature [B,64] -> gamma, beta each [B,64].
        # Zero-init -> (1+gamma)=1, beta=0 at start, equivalent to original model (warm start).
        self.film_generator = nn.Linear(512, 2 * 64)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)
        # New PV-only 4h branch implemented with TabM:
        # flatten pv_history to table features and predict kt at t0+4h.
        self.pv_tabm_head = TabM(
            n_num_features=2336,
            cat_cardinalities=None,
            d_out=1,
            n_blocks=3,
            d_block=512,
            dropout=dropout,
            k=48,
            arch_type="tabm",
            start_scaling_init="normal",
        )
        # Cached feature right before TabM final output projection.
        self.pv_tabm_preoutput_features: Optional[torch.Tensor] = None
        self._pv_tabm_output_pre_hook_handle = self.pv_tabm_head.output.register_forward_pre_hook(
            self._capture_pv_tabm_preoutput_features
        )

    @staticmethod
    def _apply_channel_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if (not training) or p <= 0.0:
            return x
        keep = 1.0 - float(p)
        if keep <= 0.0:
            return torch.zeros_like(x)
        bsz, _, channels = x.shape
        mask = (torch.rand((bsz, 1, channels), device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def _capture_pv_tabm_preoutput_features(self, module: nn.Module, inputs: tuple) -> None:
        _ = module
        if not inputs:
            self.pv_tabm_preoutput_features = None
            return
        x = inputs[0]
        self.pv_tabm_preoutput_features = x if isinstance(x, torch.Tensor) else None

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor, 
                pv_mask: Optional[torch.Tensor] = None, 
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None,
                nwp_history: Optional[torch.Tensor] = None,
                nwp_forecast_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        pv_masked = pv * pv_mask.to(pv.dtype)

        pv_timefeats = pv_timefeats[:, :, [2,3,8]]
        forecast_timefeats = forecast_timefeats[:, :, [2,3,8]]

        # Read history features for PV branch; if missing, fallback to zeros.
        if nwp_history is None:
            nwp_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_history_ghi = ((nwp_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        if nwp_forecast_history is None:
            nwp_forecast_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_forecast_history_ghi = ((nwp_forecast_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        delta_nwp_history_ghi = nwp_history_ghi - nwp_forecast_history_ghi

        pv_history = torch.cat(
            [
                pv_masked,
                pv_mask,
                pv_timefeats.permute(0, 2, 1),
                nwp_history_ghi,
                nwp_forecast_history_ghi,
                delta_nwp_history_ghi,
            ],
            dim=1,
        )  # [B, C=8, T]

        nwp_ssrd_normalized = (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0
        nwp_t2m_normalized = (nwp_tensor[:, :, 2] - 288.15) / 10.0

        
        pv_hist_flat = pv_history[:,[0,5,6,7],:].reshape(pv_history.shape[0], -1)
        nwp_tensor_for_tabm = torch.stack(
            [
                nwp_ssrd_normalized,
                nwp_t2m_normalized,
            ],
            dim=2,
        )
        nwp_tensor_flat = nwp_tensor_for_tabm.reshape(nwp_tensor_for_tabm.shape[0], -1)
        cur_nwp_dim = nwp_tensor_flat.shape[1]
        if cur_nwp_dim < self.nwp_future_flat_dim:
            pad = torch.zeros(
                nwp_tensor_flat.shape[0],
                self.nwp_future_flat_dim - cur_nwp_dim,
                device=nwp_tensor_flat.device,
                dtype=nwp_tensor_flat.dtype,
            )
            nwp_tensor_flat = torch.cat([nwp_tensor_flat, pad], dim=1)
        elif cur_nwp_dim > self.nwp_future_flat_dim:
            nwp_tensor_flat = nwp_tensor_flat[:, : self.nwp_future_flat_dim]

        x_num = torch.cat([pv_hist_flat, nwp_tensor_flat], dim=1)

        self.pv_tabm_preoutput_features = None
        tabm_out = self.pv_tabm_head(x_num=x_num, x_cat=None)  # [B, K, 1]
        if self.pv_tabm_preoutput_features is None:
            raise RuntimeError("pv_tabm_preoutput_features hook capture failed.")
        tabm_summary_feature = self.pv_tabm_preoutput_features  # [B, 1, 64]
        kt_tabm = tabm_out.mean(dim=1)  # [B, 1]

        # return kt_tabm  # [B, 1] — keep dim so pv_pred = kt * target_p_cs * p_mean is [B, 1]

        # Forecast queries: Luoyang total variant only uses two NWP channels:
        # ssrd (idx=0) and t2m (idx=2), where nwp_tensor layout is
        # [ssrd, msl, t2m, u10, v10, u100, v100, nan_mask].

        nwp_channels = [forecast_timefeats]
        if nwp_tensor is not None:
            nwp_feats = torch.stack(
                [
                    (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0,
                    nwp_tensor[:, :, 1],
                    (nwp_tensor[:, :, 2] - 288.15) / 10.0,
                    nwp_tensor[:, :, 3],
                ],
                dim=2,
            )
            nwp_feats = self._apply_channel_dropout(
                nwp_feats,
                p=self.nwp_dropout_prob,
                training=self.training,
            )
            nwp_channels.append(nwp_feats)
        else:
            zero_nwp_feat = torch.zeros(
                forecast_timefeats.shape[:2],
                device=forecast_timefeats.device,
                dtype=forecast_timefeats.dtype,
            )
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # ssrd
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # t2m
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # ssrd
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # t2m

        forecast_ssrd_timefeats = torch.cat(nwp_channels, dim=2)
        
        forecast_query = self.query_mlp(forecast_ssrd_timefeats)

        # satellite images encoder
        B, T_out, _ = forecast_query.shape

        if sat_tensor is None or sat_tensor.max() == 0:
            sat_compressed = torch.zeros(B, 48, 64, device=pv.device, dtype=pv.dtype) + self.sat_mod_embed
            sat_mask = torch.zeros(B, 48, device=pv.device, dtype=pv.dtype)
        else:
            B_sat, T_sat, C_sat, H_sat, W_sat = sat_tensor.shape
            if C_sat != 3:
                raise ValueError(f"sat_tensor expected 3 channels, got {C_sat}")
            sat_hr = nn.functional.interpolate(
                sat_tensor.reshape(B_sat * T_sat, C_sat, H_sat, W_sat),
                size=(112, 112),
                mode="bilinear",
                align_corners=False,
            ).view(B_sat, T_sat, 3, 112, 112)

            sat_timefeats = sat_timefeats[:, :, [2,3,8]]
            sat_patch_tokens = patchify_spatiotemporal_images(sat_hr, self.sat_patch_embed, timefeats=sat_timefeats[:,:,-1].unsqueeze(2))
            sat_patch_tokens = self.sat_alt_attn(sat_patch_tokens)  # [B,T=24,P=49,D=64]
            sat_start = sat_timefeats[:, 0, -1]   # [B]
            sat_end   = sat_timefeats[:, -1, -1]  # [B]
            sat_steps = torch.linspace(0, 1, 48, device=sat_patch_tokens.device)
            sat_query_times = sat_start[:, None] + (sat_end - sat_start)[:, None] * sat_steps  # [B, 48]
            sat_compressed = self.sat_two_stage_compressor(sat_patch_tokens, sat_query_times) + self.sat_mod_embed  # [B,P=48,D=64]

            sat_timefeats_48 = F.interpolate(
                    sat_timefeats.transpose(1, 2),       # -> [B, F=3, T=24]  (interp 要求 [N, C, L])
                    size=48,
                    mode="linear",
                    align_corners=True,                  # 头尾对齐 -> 保留首末时刻原值
                ).transpose(1, 2)

            sat_timefeats_48_hd = self.time_mlp(sat_timefeats_48)
            sat_compressed = sat_compressed + sat_timefeats_48_hd
            
            # print('sat_patch_tokens: ', sat_patch_tokens.shape, sat_compressed.shape)
            sat_mask = torch.ones(B, sat_compressed.shape[1], device=pv.device, dtype=pv.dtype)

        if sat_valid_mask is not None:
            if sat_valid_mask.dim() != 1 or sat_valid_mask.shape[0] != B:
                raise ValueError(f"sat_valid_mask expected [B], got {sat_valid_mask.shape}")
            sat_valid_mask = sat_valid_mask.to(device=pv.device, dtype=pv.dtype).unsqueeze(1)
            sat_compressed = sat_compressed * sat_valid_mask.unsqueeze(2)
            sat_mask = sat_mask * sat_valid_mask

        # sky images encoder
        if skimg_tensor is None or skimg_tensor.max() == 0:
            sky_compressed = torch.zeros(B, 48, 64, device=pv.device, dtype=pv.dtype) + self.sky_mod_embed
            sky_mask = torch.zeros(B, 48, device=pv.device, dtype=pv.dtype)
        else:
            B_sky, T_sky, C_sky, H_sky, W_sky = skimg_tensor.shape
            if C_sky < 3:
                raise ValueError(f"skimg_tensor expected at least 3 channels, got {C_sky}")
            skimg_timefeats = skimg_timefeats[:, :, [2, 3, 8]]
            if H_sky != 224 or W_sky != 224:
                sky_hr = nn.functional.interpolate(
                    skimg_tensor.reshape(B_sky * T_sky, C_sky, H_sky, W_sky),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,).view(B_sky, T_sky, C_sky, 224, 224)
            else:
                sky_hr = skimg_tensor

            sky_patch_tokens = patchify_spatiotemporal_sky_images(
                sky_hr,
                self.sky_patch_embed,
                timefeats=skimg_timefeats[:, :, -1].unsqueeze(2),
            )
            sky_patch_tokens = self.sky_alt_attn(sky_patch_tokens)
            sky_start = skimg_timefeats[:, 0, -1]   # [B]
            sky_end   = skimg_timefeats[:, -1, -1]  # [B]
            sky_steps = torch.linspace(0, 1, 48, device=sky_patch_tokens.device)
            sky_query_times = sky_start[:, None] + (sky_end - sky_start)[:, None] * sky_steps  # [B, 48]
            sky_compressed = self.sky_two_stage_compressor(sky_patch_tokens, sky_query_times) + self.sky_mod_embed  # [B,P=48,D=64]

            sky_timefeats_48 = F.interpolate(
                    skimg_timefeats.transpose(1, 2),       # -> [B, F=3, T=30]
                    size=48,
                    mode="linear",
                    align_corners=True,                  # 头尾对齐 -> 保留首末时刻原值
                ).transpose(1, 2)

            sky_timefeats_48_hd = self.time_mlp(sky_timefeats_48)
            sky_compressed = sky_compressed + sky_timefeats_48_hd
            sky_compressed = torch.nan_to_num(sky_compressed, nan=0.0, posinf=0.0, neginf=0.0)

            sky_mask = torch.ones(B, 48, device=pv.device, dtype=pv.dtype)

        if skimg_valid_mask is not None:
            if skimg_valid_mask.dim() != 1 or skimg_valid_mask.shape[0] != B:
                raise ValueError(f"skimg_valid_mask expected [B], got {skimg_valid_mask.shape}")
            skimg_valid_mask = skimg_valid_mask.to(device=pv.device, dtype=pv.dtype).unsqueeze(1)
            sky_compressed = sky_compressed * skimg_valid_mask.unsqueeze(2)
            sky_mask = sky_mask * skimg_valid_mask

        # PV history branch: TCN → coarse-frame selection → cross-attention compression
        pv_tcn_out = self.TCN(pv_history)  # [B, 64, T]
        pv_coarse = pv_tcn_out[:, :, self.corase_idx].permute(0, 2, 1)  # [B, 48, 64]
        pv_queries = self.learnable_pv_queries.expand(B, -1, -1)  # [B, 48, 64]
        pv_compressed = self.cross_attention_pv_compression(
            query=pv_queries, key=pv_coarse, value=pv_coarse
        ) + self.pv_mod_embed  # [B, 48, 64]
        pv_mask = torch.ones(B, 48, device=pv.device, dtype=pv.dtype)  # PV always available

        hist_mem_compressed = torch.cat(
            [pv_compressed, sat_compressed, sky_compressed], dim=1
        )
        key_value_mask = torch.cat([pv_mask, sat_mask, sky_mask], dim=1)

        # PV is always valid so key_value_mask.sum() > 0 is guaranteed.
        # Still apply per-sample safe mask for robustness.
        row_has_valid = (key_value_mask.sum(dim=1, keepdim=True) > 0).to(pv.dtype)  # [B,1]
        safe_mask = torch.where(
            row_has_valid.bool(), key_value_mask, torch.ones_like(key_value_mask)
        )
        forecast_pv_features = self.cross_attention_pv(query=forecast_query[:,-1:,:], key=hist_mem_compressed, value=hist_mem_compressed, key_value_mask=safe_mask)

        # Inverter features (embeddings)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_pv_features.shape[1], 1)

        # FiLM: tabm_summary_feature [B,32,512] modulates forecast_pv_features [B,1,64]
        # before MLP fusion, so the conditioning passes through the non-linearity.
        # Zero-init on film_generator ensures identity at init (warm start).
        film_params = self.film_generator(tabm_summary_feature.mean(dim=1))  # [B, 128]
        gamma, beta = film_params.chunk(2, dim=-1)                           # each [B, 64]
        forecast_pv_features = (1.0 + gamma).unsqueeze(1) * forecast_pv_features + beta.unsqueeze(1)  # [B, 1, 64]

        # Fuse and predict
        fused = torch.cat([forecast_pv_features, inverter_features], dim=2)
        pv_feats = self.pv_feats_head(fused)                                 # [B, 1, 64]

        # Image gate: 1 if any sat/sky token is valid for that sample, else 0.
        # When both sat and sky are absent, gate=0 → delta_kt=0 → pure TabM output.
        sat_any = sat_mask.any(dim=1).to(pv.dtype)               # [B]
        sky_any = sky_mask.any(dim=1).to(pv.dtype)               # [B]
        img_gate = (sat_any + sky_any).clamp(0, 1).view(B, 1, 1)  # [B, 1, 1]

        delta_kt = self.fc(pv_feats) * img_gate  # [B, 1, 1]; zero when no images

        kt = kt_tabm.unsqueeze(1) + delta_kt  # [B, 1, 1]

        return kt.squeeze(-1)



# Using PV history and NWP to forecast PV, solar features and NWP features are used as query
class pv_forecasting_model_vit_gate(nn.Module):
    """
    Like ``pv_forecasting_model_vit_nwp`` but satellite frames go through
    :func:`modules.SatEncoder.patchify_spatiotemporal_images` and
    :class:`modules.SatEncoder.AlternatingIntraInterFrameAttention`, then cross-attend into the forecast query.

    NWP forecast-query channels are configurable: ``nwp_features`` selects which columns
    of ``_FOLSOM_NWP_FEATURE_COLS`` to read from ``nwp_tensor`` (each normalised via
    :data:`NWP_FEATURE_NORMALIZERS`), and ``use_invalid_mask`` toggles passing the trailing
    per-step invalid mask channel (``nwp_tensor[:, :, -1]``) through as-is.
    """

    def __init__(
        self,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        dev_dn_list: Optional[list] = None,
        nwp_features: Optional[list[str]] = None,
        use_invalid_mask: bool = _DEFAULT_VIT_IMGS_NWP_USE_INVALID_MASK,
        nwp_dropout_prob: float = 0.0,
        nwp_history_dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        if nwp_features is None:
            nwp_features = list(_DEFAULT_VIT_IMGS_NWP_FEATURES)
        nwp_features = list(nwp_features)
        unknown = [n for n in nwp_features if n not in NWP_FEATURE_NORMALIZERS]
        if unknown:
            raise ValueError(
                f"Unknown NWP feature(s) {unknown}; valid features are "
                f"{sorted(NWP_FEATURE_NORMALIZERS)}"
            )
        dupes = [n for n in nwp_features if nwp_features.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate NWP feature(s) in nwp_features: {sorted(set(dupes))}")
        self.nwp_features: list[str] = nwp_features
        self.use_invalid_mask: bool = bool(use_invalid_mask)
        self.nwp_dropout_prob: float = max(0.0, min(1.0, float(nwp_dropout_prob)))
        self.nwp_history_dropout_prob: float = max(
            0.0, min(1.0, float(nwp_history_dropout_prob))
        )
        # Pre-resolve column indices in ``nwp_tensor`` (last dim = 8 features + 1 mask).
        self._nwp_feature_indices: list[int] = [
            _FOLSOM_NWP_FEATURE_COLS.index(name) for name in self.nwp_features
        ]
        # Forecast-query input dim = 3 time feats + N NWP features (+ 1 if mask passed through).
        query_mlp_in_dim = 3 + 4

        dim = 64
        self.tabm_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # TabM modality embedding
        self.pv_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # PV modality embedding
        self.sat_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Satellite image modality embedding
        self.sky_mod_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)   # Sky imagem odality embedding

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.pv_hist_in_channels = 8
        self.nwp_future_steps = 16
        self.nwp_future_channels = 2
        self.nwp_future_flat_dim = self.nwp_future_steps * self.nwp_future_channels
        self.TCN = TemporalCNN1d(
            in_channels=self.pv_hist_in_channels,
            out_channels=64,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.corase_idx = [18, 54, 90, 126, 162, 198, 234, 270, 295, 308, 322, 335, 349, 362, 376, 389,
                           403, 416, 430, 443, 457, 470, 484, 497, 505, 508, 511, 514, 517, 520, 523, 526,
                           529, 532, 535, 538, 541, 544, 547, 550, 553, 556, 559, 562, 565, 568, 571, 574]
        # This model path uses 576-step history (derived from coarse idx settings).
        self.pv_hist_len = max(self.corase_idx) + 2
        self.learnable_pv_queries = nn.Parameter(torch.randn(1, 48, 64))
        self.cross_attention_pv_compression = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)        
        self.query_mlp = MLP(in_dim=query_mlp_in_dim, hidden_dims=(64, 64), out_dim=64, dropout=0.0)

        self.sat_embed_dim = 64
        self.sat_patch_embed = VideoPatchSpatiotemporalEmbed(
            embed_dim=self.sat_embed_dim, patch_size=16, image_size=112
        )
        self.sat_alt_attn = AlternatingIntraInterFrameAttention(
            embed_dim=self.sat_embed_dim, num_heads=8, num_cycles=4, dropout=dropout
        )

        self.sat_two_stage_compressor = SatelliteTwoStageCompressor(
                                            dim=self.sat_embed_dim,
                                            num_frames=24,
                                            num_patches=49,
                                            num_frame_queries=8,   # 196 -> 8 per frame
                                            num_sat_queries=48,    # 24*8=192 -> 48 final tokens
                                            num_heads=8,
                                            use_patch_pos=True,
                                            use_frame_time=True,
                                            use_coarse_spatial=True,
                                            coarse_query_grid_hw=(6, 8),)
        
        self.sky_embed_dim = 64
        self.sky_patch_embed = SkyPatchSpatiotemporalEmbed(
            embed_dim=self.sky_embed_dim, patch_size=16, image_size=224, in_channels=4
        )
        self.sky_alt_attn = SkyAlternatingIntraInterFrameAttention(
            embed_dim=self.sky_embed_dim, num_heads=8, num_cycles=4, dropout=dropout
        )
        self.sky_two_stage_compressor = SkyTwoStageCompressor(
                                            dim=self.sky_embed_dim,
                                            num_frames=30,
                                            num_patches=196,
                                            num_frame_queries=8,
                                            num_sky_queries=48,
                                            num_heads=8,
                                            use_patch_pos=True,
                                            use_frame_time=True,
                                            use_coarse_spatial=True,
                                            coarse_query_grid_hw=(6, 8),)

        self.cross_attention_pv = CrossAttention(query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout)
        self.pv_feats_head = MLP(in_dim=80, hidden_dims=(128, 64), out_dim=64, dropout=0.0)  # PV 64 + sat 64 + inverter 16
        self.fc = FC(in_dim=64, out_dim=1)
        # New PV-only 4h branch implemented with TabM:
        # flatten pv_history to table features and predict kt at t0+4h.
        self.pv_tabm_head = TabM(
            n_num_features=2336,
            cat_cardinalities=None,
            d_out=1,
            n_blocks=3,
            d_block=512,
            dropout=dropout,
            k=32,
            arch_type="tabm",
            start_scaling_init="normal",
        )
        # Cached feature right before TabM final output projection.
        self.pv_tabm_preoutput_features: Optional[torch.Tensor] = None
        self._pv_tabm_output_pre_hook_handle = self.pv_tabm_head.output.register_forward_pre_hook(
            self._capture_pv_tabm_preoutput_features
        )
        # Gate: uses TabM mean, TabM std, NWP ssrd and t2m to produce a [0,1] gate
        # that controls how much the sat/sky residual delta_kt contributes.
        # Input: [kt_tabm(1), kt_tabm_std(1), nwp_ssrd(1), nwp_t2m(1)] -> [B, 4]
        self.gate_head = MLP(in_dim=4, hidden_dims=(32, 16), out_dim=1, dropout=0.0)

    @staticmethod
    def _apply_channel_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if (not training) or p <= 0.0:
            return x
        keep = 1.0 - float(p)
        if keep <= 0.0:
            return torch.zeros_like(x)
        bsz, _, channels = x.shape
        mask = (torch.rand((bsz, 1, channels), device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def _capture_pv_tabm_preoutput_features(self, module: nn.Module, inputs: tuple) -> None:
        _ = module
        if not inputs:
            self.pv_tabm_preoutput_features = None
            return
        x = inputs[0]
        self.pv_tabm_preoutput_features = x if isinstance(x, torch.Tensor) else None

    def forward(self, device_id: torch.Tensor, pv: torch.Tensor, 
                pv_mask: Optional[torch.Tensor] = None, 
                pv_timefeats: Optional[torch.Tensor] = None,
                forecast_timefeats: Optional[torch.Tensor] = None,
                sat_tensor: Optional[torch.Tensor] = None,
                sat_timefeats: Optional[torch.Tensor] = None,
                skimg_tensor: Optional[torch.Tensor] = None,
                skimg_timefeats: Optional[torch.Tensor] = None,
                sat_valid_mask: Optional[torch.Tensor] = None,
                skimg_valid_mask: Optional[torch.Tensor] = None,
                nwp_tensor: Optional[torch.Tensor] = None,
                nwp_history: Optional[torch.Tensor] = None,
                nwp_forecast_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        pv_masked = pv * pv_mask.to(pv.dtype)

        pv_timefeats = pv_timefeats[:, :, [2,3,8]]
        forecast_timefeats = forecast_timefeats[:, :, [2,3,8]]

        # Read history features for PV branch; if missing, fallback to zeros.
        if nwp_history is None:
            nwp_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_history_ghi = ((nwp_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        if nwp_forecast_history is None:
            nwp_forecast_history_ghi = torch.zeros_like(pv_masked)
        else:
            nwp_forecast_history_ghi = ((nwp_forecast_history[:, :, 0] / 1000.0 - 0.5) * 2.0).unsqueeze(1)
        delta_nwp_history_ghi = nwp_history_ghi - nwp_forecast_history_ghi

        pv_history = torch.cat(
            [
                pv_masked,
                pv_mask,
                pv_timefeats.permute(0, 2, 1),
                nwp_history_ghi,
                nwp_forecast_history_ghi,
                delta_nwp_history_ghi,
            ],
            dim=1,
        )  # [B, C=8, T]

        nwp_ssrd_normalized = (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0
        nwp_t2m_normalized = (nwp_tensor[:, :, 2] - 288.15) / 10.0

        
        pv_hist_flat = pv_history[:,[0,5,6,7],:].reshape(pv_history.shape[0], -1)
        nwp_tensor_for_tabm = torch.stack(
            [
                nwp_ssrd_normalized,
                nwp_t2m_normalized,
            ],
            dim=2,
        )
        nwp_tensor_flat = nwp_tensor_for_tabm.reshape(nwp_tensor_for_tabm.shape[0], -1)
        cur_nwp_dim = nwp_tensor_flat.shape[1]
        if cur_nwp_dim < self.nwp_future_flat_dim:
            pad = torch.zeros(
                nwp_tensor_flat.shape[0],
                self.nwp_future_flat_dim - cur_nwp_dim,
                device=nwp_tensor_flat.device,
                dtype=nwp_tensor_flat.dtype,
            )
            nwp_tensor_flat = torch.cat([nwp_tensor_flat, pad], dim=1)
        elif cur_nwp_dim > self.nwp_future_flat_dim:
            nwp_tensor_flat = nwp_tensor_flat[:, : self.nwp_future_flat_dim]

        x_num = torch.cat([pv_hist_flat, nwp_tensor_flat], dim=1)

        self.pv_tabm_preoutput_features = None
        tabm_out = self.pv_tabm_head(x_num=x_num, x_cat=None)  # [B, K, 1]
        kt_tabm = tabm_out.mean(dim=1)  # [B, 1]

        # The gate
        kt_tabm_std = tabm_out.std(dim=1)
        gate_features = torch.cat([kt_tabm, kt_tabm_std, nwp_ssrd_normalized, nwp_t2m_normalized], dim=1)
        gate_features = self.gate_head(gate_features)
        gate = torch.sigmoid(gate_features)

        # return kt_tabm  # [B, 1] — keep dim so pv_pred = kt * target_p_cs * p_mean is [B, 1]

        # Forecast queries: Luoyang total variant only uses two NWP channels:
        # ssrd (idx=0) and t2m (idx=2), where nwp_tensor layout is
        # [ssrd, msl, t2m, u10, v10, u100, v100, nan_mask].

        nwp_channels = [forecast_timefeats]
        if nwp_tensor is not None:
            nwp_feats = torch.stack(
                [
                    (nwp_tensor[:, :, 0] / 1000.0 - 0.5) * 2.0,
                    nwp_tensor[:, :, 1],
                    (nwp_tensor[:, :, 2] - 288.15) / 10.0,
                    nwp_tensor[:, :, 3],
                ],
                dim=2,
            )
            nwp_feats = self._apply_channel_dropout(
                nwp_feats,
                p=self.nwp_dropout_prob,
                training=self.training,
            )
            nwp_channels.append(nwp_feats)
        else:
            zero_nwp_feat = torch.zeros(
                forecast_timefeats.shape[:2],
                device=forecast_timefeats.device,
                dtype=forecast_timefeats.dtype,
            )
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # ssrd
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # t2m
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # ssrd
            nwp_channels.append(zero_nwp_feat.unsqueeze(2))  # t2m

        forecast_ssrd_timefeats = torch.cat(nwp_channels, dim=2)
        
        forecast_query = self.query_mlp(forecast_ssrd_timefeats)

        # satellite images encoder
        B, T_out, _ = forecast_query.shape

        if sat_tensor is None or sat_tensor.max() == 0:
            sat_compressed = torch.zeros(B, 48, 64, device=pv.device, dtype=pv.dtype) + self.sat_mod_embed
            sat_mask = torch.zeros(B, 48, device=pv.device, dtype=pv.dtype)
        else:
            B_sat, T_sat, C_sat, H_sat, W_sat = sat_tensor.shape
            if C_sat != 3:
                raise ValueError(f"sat_tensor expected 3 channels, got {C_sat}")
            sat_hr = nn.functional.interpolate(
                sat_tensor.reshape(B_sat * T_sat, C_sat, H_sat, W_sat),
                size=(112, 112),
                mode="bilinear",
                align_corners=False,
            ).view(B_sat, T_sat, 3, 112, 112)

            sat_timefeats = sat_timefeats[:, :, [2,3,8]]
            sat_patch_tokens = patchify_spatiotemporal_images(sat_hr, self.sat_patch_embed, timefeats=sat_timefeats[:,:,-1].unsqueeze(2))
            sat_patch_tokens = self.sat_alt_attn(sat_patch_tokens)  # [B,T=24,P=49,D=64]
            sat_start = sat_timefeats[:, 0, -1]   # [B]
            sat_end   = sat_timefeats[:, -1, -1]  # [B]
            sat_steps = torch.linspace(0, 1, 48, device=sat_patch_tokens.device)
            sat_query_times = sat_start[:, None] + (sat_end - sat_start)[:, None] * sat_steps  # [B, 48]
            sat_compressed = self.sat_two_stage_compressor(sat_patch_tokens, sat_query_times) + self.sat_mod_embed  # [B,P=48,D=64]

            sat_timefeats_48 = F.interpolate(
                    sat_timefeats.transpose(1, 2),       # -> [B, F=3, T=24]  (interp 要求 [N, C, L])
                    size=48,
                    mode="linear",
                    align_corners=True,                  # 头尾对齐 -> 保留首末时刻原值
                ).transpose(1, 2)

            sat_timefeats_48_hd = self.time_mlp(sat_timefeats_48)
            sat_compressed = sat_compressed + sat_timefeats_48_hd
            
            # print('sat_patch_tokens: ', sat_patch_tokens.shape, sat_compressed.shape)
            sat_mask = torch.ones(B, sat_compressed.shape[1], device=pv.device, dtype=pv.dtype)

        if sat_valid_mask is not None:
            if sat_valid_mask.dim() != 1 or sat_valid_mask.shape[0] != B:
                raise ValueError(f"sat_valid_mask expected [B], got {sat_valid_mask.shape}")
            sat_valid_mask = sat_valid_mask.to(device=pv.device, dtype=pv.dtype).unsqueeze(1)
            sat_compressed = sat_compressed * sat_valid_mask.unsqueeze(2)
            sat_mask = sat_mask * sat_valid_mask

        # sky images encoder
        if skimg_tensor is None or skimg_tensor.max() == 0:
            sky_compressed = torch.zeros(B, 48, 64, device=pv.device, dtype=pv.dtype) + self.sky_mod_embed
            sky_mask = torch.zeros(B, 48, device=pv.device, dtype=pv.dtype)
        else:
            B_sky, T_sky, C_sky, H_sky, W_sky = skimg_tensor.shape
            if C_sky < 3:
                raise ValueError(f"skimg_tensor expected at least 3 channels, got {C_sky}")
            skimg_timefeats = skimg_timefeats[:, :, [2, 3, 8]]
            if H_sky != 224 or W_sky != 224:
                sky_hr = nn.functional.interpolate(
                    skimg_tensor.reshape(B_sky * T_sky, C_sky, H_sky, W_sky),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,).view(B_sky, T_sky, C_sky, 224, 224)
            else:
                sky_hr = skimg_tensor

            sky_patch_tokens = patchify_spatiotemporal_sky_images(
                sky_hr,
                self.sky_patch_embed,
                timefeats=skimg_timefeats[:, :, -1].unsqueeze(2),
            )
            sky_patch_tokens = self.sky_alt_attn(sky_patch_tokens)
            sky_start = skimg_timefeats[:, 0, -1]   # [B]
            sky_end   = skimg_timefeats[:, -1, -1]  # [B]
            sky_steps = torch.linspace(0, 1, 48, device=sky_patch_tokens.device)
            sky_query_times = sky_start[:, None] + (sky_end - sky_start)[:, None] * sky_steps  # [B, 48]
            sky_compressed = self.sky_two_stage_compressor(sky_patch_tokens, sky_query_times) + self.sky_mod_embed  # [B,P=48,D=64]

            sky_timefeats_48 = F.interpolate(
                    skimg_timefeats.transpose(1, 2),       # -> [B, F=3, T=30]
                    size=48,
                    mode="linear",
                    align_corners=True,                  # 头尾对齐 -> 保留首末时刻原值
                ).transpose(1, 2)

            sky_timefeats_48_hd = self.time_mlp(sky_timefeats_48)
            sky_compressed = sky_compressed + sky_timefeats_48_hd
            sky_compressed = torch.nan_to_num(sky_compressed, nan=0.0, posinf=0.0, neginf=0.0)

            sky_mask = torch.ones(B, 48, device=pv.device, dtype=pv.dtype)

        if skimg_valid_mask is not None:
            if skimg_valid_mask.dim() != 1 or skimg_valid_mask.shape[0] != B:
                raise ValueError(f"skimg_valid_mask expected [B], got {skimg_valid_mask.shape}")
            skimg_valid_mask = skimg_valid_mask.to(device=pv.device, dtype=pv.dtype).unsqueeze(1)
            sky_compressed = sky_compressed * skimg_valid_mask.unsqueeze(2)
            sky_mask = sky_mask * skimg_valid_mask

        # tabm_mask = torch.ones(B, tabm_summary_token.shape[1], device=pv.device, dtype=pv.dtype)
        hist_mem_compressed = torch.cat(
            [sat_compressed, sky_compressed], dim=1
        )
        key_value_mask = torch.cat([sat_mask, sky_mask], dim=1)

        # Per-sample handling of all-masked rows: an all-masked row makes the
        # attention softmax all -inf → NaN for that sample. Give those rows a
        # dummy all-valid mask to keep the math finite, then zero their
        # residual so samples without sat/sky contribute delta_kt = 0.
        row_has_valid = (key_value_mask.sum(dim=1, keepdim=True) > 0).to(pv.dtype)  # [B,1]
        safe_mask = torch.where(
            row_has_valid.bool(), key_value_mask, torch.ones_like(key_value_mask)
        )
        forecast_pv_features = self.cross_attention_pv(query=forecast_query[:,-1:,:], key=hist_mem_compressed, value=hist_mem_compressed, key_value_mask=safe_mask)

        # Inverter features (embeddings)
        inverter_features = self.inverter_embedding(device_id).unsqueeze(1).repeat(1, forecast_pv_features.shape[1], 1)

        # Fuse and predict
        fused = torch.cat([forecast_pv_features, inverter_features], dim=2)
        pv_feats = self.pv_feats_head(fused)
        delta_kt = self.fc(pv_feats) * row_has_valid.unsqueeze(2)  # [B,1,1]

        kt = kt_tabm.unsqueeze(1) + gate * delta_kt # [B,1,1]

        return kt.squeeze(-1)



# Using PV history and NWP to forecast PV, solar features and NWP features are used as query

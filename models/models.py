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
                nwp_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        
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
        pv = self.fc(pv_feats)

        return pv.squeeze(-1)


# Using PV history and NWP to forecast PV, solar features and NWP features are used as query
class pv_forecasting_model_vit_nwp(nn.Module):
    def __init__(self, use_batchnorm: bool = True, dropout: float = 0.0, dev_dn_list: Optional[list] = None):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        self.inverter_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=16)
        self.TCN = TemporalCNN1d(in_channels=11, out_channels=64, use_batchnorm=use_batchnorm, dropout=dropout)
        self.query_mlp = MLP(in_dim=11, hidden_dims=(64, 64), out_dim=64, dropout=0.0)
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
                nwp_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # PV features
        pv_masked = pv * pv_mask.to(pv.dtype)
        pv_history = torch.cat([pv_masked, pv_mask, pv_timefeats.permute(0, 2, 1)], dim=1)  # [B, C=11, T]
        pv_hist_mem = self.TCN(pv_history, pv_mask)     # [B, C_out, T]()
        KV_hist_mem = pv_hist_mem.permute(0, 2, 1)   # [B, T, C_out]

        ssrd_normalized = (nwp_tensor[:,:,0]/1000 - 0.5)*2
        forecast_ssrd_timefeats = torch.cat([forecast_timefeats, ssrd_normalized.unsqueeze(2), nwp_tensor[:,:,-1].unsqueeze(2)], dim=2)
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


class irr_forecasting_model_vit_folsom(nn.Module):
    """
    Folsom multimodal forecaster: 3-channel irradiance TCN, NWP-augmented forecast queries,
    sky TimeSformer branch fused with cross-attention. Returns ``[B, T_out, 3]`` (GHI, DNI, DHI).
    """

    _timefeat_dim = 9

    def __init__(
        self,
        *,
        skyimg_window_size: int,
        skyimg_spatial_size: int = 224,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.skyimg_window_size = int(skyimg_window_size)
        self.skyimg_spatial_size = int(skyimg_spatial_size)

        logging.getLogger("transformers").setLevel(logging.ERROR)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        config = TimesformerConfig(
            num_frames=self.skyimg_window_size,
            image_size=224,
            patch_size=16,
            num_channels=3,
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.skimg_extractor = TimeSformerFeatureExtractor(
                pretrained="facebook/timesformer-base-finetuned-k400",
                output_format="sequence",
                normalize=True,
                config=config,
            )

        self.TCN = TemporalCNN1d(
            in_channels=13,
            out_channels=64,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )
        self.query_mlp = MLP(in_dim=11, hidden_dims=(64, 64), out_dim=64, dropout=0.0)
        self.cross_attention_irr = CrossAttention(
            query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout
        )
        self.cross_attention_skimg = CrossAttention(
            query_dim=64, key_dim=64, value_dim=64, embed_dim=64, num_heads=4, dropout=dropout
        )
        self.skimg_downdim = MLP(in_dim=768, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.timefeats_encoder = MLP(in_dim=self._timefeat_dim, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.irr_feats_head = MLP(in_dim=64, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.fc_out = nn.Linear(64, 3)

    def forward(
        self,
        ghi: torch.Tensor,
        dni: torch.Tensor,
        dhi: torch.Tensor,
        input_mask: torch.Tensor,
        irr_timefeats: torch.Tensor,
        forecast_timefeats: torch.Tensor,
        skimg_tensor: Optional[torch.Tensor] = None,
        skimg_timefeats: Optional[torch.Tensor] = None,
        nwp_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gm = input_mask.to(ghi.dtype)
        ghi_m = ghi.unsqueeze(1) * gm
        dni_m = dni.unsqueeze(1) * gm
        dhi_m = dhi.unsqueeze(1) * gm
        irr_tf = irr_timefeats.permute(0, 2, 1)
        irr_hist = torch.cat([ghi_m, dni_m, dhi_m, gm, irr_tf], dim=1)
        # Keep mask shape [B,1,T] so it broadcasts across all history channels in TemporalCNN1d.
        irr_mem = self.TCN(irr_hist, gm)
        kv_irr = irr_mem.permute(0, 2, 1)

        B, T_out, _ = forecast_timefeats.shape
        if nwp_tensor is None:
            nwp_front = torch.zeros(B, T_out, device=ghi.device, dtype=ghi.dtype)
            nwp_bad = torch.ones(B, T_out, device=ghi.device, dtype=ghi.dtype)
        else:
            nwp_front = (nwp_tensor[:, :, 0].clamp(min=0.0) / 1000.0 - 0.5) * 2.0
            nwp_bad = nwp_tensor[:, :, -1]

        q_in = torch.cat(
            [forecast_timefeats, nwp_front.unsqueeze(2), nwp_bad.unsqueeze(2)],
            dim=2,
        )
        forecast_query = self.query_mlp(q_in)
        irr_q = self.cross_attention_irr(query=forecast_query, key=kv_irr, value=kv_irr)

        if skimg_tensor is None or skimg_timefeats is None:
            sky_q = torch.zeros_like(irr_q)
        else:
            H0, W0 = int(skimg_tensor.shape[-2]), int(skimg_tensor.shape[-1])
            sk = nn.functional.interpolate(
                skimg_tensor.reshape(-1, 3, H0, W0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).view(ghi.shape[0], self.skyimg_window_size, 3, 224, 224)
            sk_feat = self.skimg_extractor(sk)
            sk_down = self.skimg_downdim(sk_feat)
            sk_tf = self.timefeats_encoder(skimg_timefeats)
            sk_kv = sk_down + sk_tf
            sky_q = self.cross_attention_skimg(query=forecast_query, key=sk_kv, value=sk_kv)

        fused = irr_q + sky_q
        x = self.irr_feats_head(fused)
        return self.fc_out(x)
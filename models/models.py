"""
1D temporal CNN for encoding time-series (e.g. PV history with mask).
Input shape: (batch, in_channels, seq_len). Optional mask: (batch, seq_len).
"""

import torch
import torch.nn as nn
from typing import Optional
import torch


class TemporalCNN1d(nn.Module):
    """
    1D temporal CNN. Stacks Conv1d blocks and optionally applies a mask by zeroing
    masked positions before the first conv. Output is a fixed-size encoding vector.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: list = (32, 64, 64),
        kernel_size: int = 3,
        out_dim: Optional[int] = None,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = list(hidden_channels)
        self.kernel_size = kernel_size
        self.out_dim = out_dim
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

        if out_dim is not None:
            self.proj = nn.Linear(self._out_channels, out_dim)
        else:
            self.proj = None



    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
            mask: (batch, seq_len), 1 = valid, 0 = masked. If given, masked positions are zeroed in x.
        Returns:
            (batch, out_dim) if out_dim was set, else (batch, out_channels, seq_len)
        """
        if mask is not None:
            # Zero masked positions: x * mask along time
            x = x * mask.unsqueeze(1).to(x.dtype)

        out = self.conv_blocks(x)  # (batch, out_channels, seq_len)

        if self.proj is None:
            return out

        # Global average pooling over time, then project
        out = out.mean(dim=2)  # (batch, out_channels)
        out = self.proj(out)   # (batch, out_dim)
        return out


class MLP(nn.Module):
    """Small MLP: Linear -> ReLU -> [Linear -> ReLU] -> Linear."""

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
        return self.mlp(x)


class pv_forecasting_model(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: list = (32, 64, 64), kernel_size: int = 3, out_dim: Optional[int] = None, use_batchnorm: bool = True, dropout: float = 0.0, dev_dn_list: Optional[list] = None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = list(hidden_channels)
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        if dev_dn_list is not None:
            self.dev_dn_list = list(dev_dn_list)
            self.dev_dn_to_idx = {str(dn): i for i, dn in enumerate(dev_dn_list)}
            num_devices = len(self.dev_dn_list)
        else:
            self.dev_dn_list = None
            self.dev_dn_to_idx = {}
            num_devices = 1000

        pv_features_dim = 64
        
        pv_age = torch.linspace(15, 192*15, 192)
        self.register_buffer("pv_age_feat", torch.stack([pv_age, torch.log(pv_age)], dim=1))
        
        self.pv_temporal_cnn = TemporalCNN1d(in_channels, hidden_channels, kernel_size, pv_features_dim, use_batchnorm, dropout)
        self.pv_gate = MLP(in_dim=8, hidden_dims=(), out_dim=pv_features_dim, dropout=0.0)

        self.forecast_solar_features_encoder = MLP(in_dim=8, hidden_dims=(16,32), out_dim=32, dropout=0.0)

        self.device_embedding = nn.Embedding(num_embeddings=num_devices, embedding_dim=16)
        self.device_encoder = MLP(in_dim=16, hidden_dims=(16, 16), out_dim=16, dropout=0.0)

        self.mlp = MLP(in_dim=112, hidden_dims=(128, 64), out_dim=64, dropout=0.0)
        self.fc = nn.Linear(64, 1)

    def forward(self, device_id: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None, pv_ztime: Optional[torch.Tensor] = None, solar_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        pv_features = self.pv_temporal_cnn(x, mask) # [B, 64] pv history features
        pv_age_ztime = torch.cat([self.pv_age_feat, pv_ztime], dim=1)
        pv_gate = torch.sigmoid( self.pv_gate(pv_age_ztime) )
        pv_features = pv_features * pv_gate   # [B, 64]
        
        forecast_solar_features = self.forecast_solar_features_encoder(solar_features)

        device_embedding = self.device_embedding(device_id).repeat(192, 1)
        device_features = self.device_encoder(device_embedding)

        fused = torch.cat([pv_features, forecast_solar_features, device_features], dim=1)
        forecast_pv_features = self.mlp(fused)
        pv = self.fc(forecast_pv_features)
        
        return pv
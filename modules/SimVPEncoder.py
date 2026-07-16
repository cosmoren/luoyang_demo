"""
SimVP-derived spatiotemporal feature extractor (no future-frame decoder).

Adapted from the official SimVP implementation (CVPR'22):
    https://github.com/A4Bio/SimVP

Only the spatial Encoder and the temporal Translator (Mid_Xnet) are kept.
The spatial Decoder that reconstructs future frames is dropped: the module
directly outputs the translator's future features of shape
``[B, T, hid_S, H', W']`` where ``H' = H / 2**(N_S//2)`` (stride-2 halving on
every other encoder layer).
"""

import torch
import torch.nn as nn


def stride_generator(N: int, reverse: bool = False) -> list[int]:
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    return strides[:N]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 transpose=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding,
                                           output_padding=stride // 2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super().__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    """1x1 bottleneck conv followed by parallel multi-kernel group convs."""

    def __init__(self, C_in, C_hid, C_out, incep_ker=(3, 5, 7, 11), groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1,
                                      padding=ker // 2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y = y + layer(x)
        return y


class SimVPSpatialEncoder(nn.Module):
    """SimVP spatial Encoder: stacked ConvSC blocks (stride 1/2 alternating)."""

    def __init__(self, C_in, C_hid, N_S):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):  # x: [B*T, C, H, W]
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class SimVPTranslator(nn.Module):
    """SimVP temporal Translator (Mid_Xnet): Inception-UNet over T*C channels.

    ``channel_out`` may differ from ``channel_in`` (e.g. ``T_out*C`` with
    ``T_out != T_in``) so the translator can directly emit a different number
    of future frames instead of one per input frame.
    """

    def __init__(self, channel_in, channel_hid, N_T, incep_ker=(3, 5, 7, 11), groups=8,
                 channel_out=None):
        super().__init__()
        self.N_T = N_T
        self.channel_out = channel_in if channel_out is None else int(channel_out)
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker, groups)]
        for _ in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker, groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker, groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker, groups)]
        for _ in range(1, N_T - 1):
            dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker, groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, self.channel_out, incep_ker, groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):  # x: [B, T, C, H', W']
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        return z.reshape(B, self.channel_out // C, C, H, W)


class SimVPFeatureExtractor(nn.Module):
    """
    SimVP Encoder + Translator without the frame-reconstruction Decoder.

    Input : ``[B, T, C, H, W]`` image sequence.
    Output: ``[B, out_frames, hid_S, H', W']`` future spatiotemporal features
            (the translator output in latent space).  ``out_frames`` defaults
            to ``in_frames`` (original SimVP behaviour); set it to a smaller
            value (e.g. 1) to emit fewer future frames and save the final
            translator layer's compute.
    """

    def __init__(
        self,
        in_frames: int,
        in_channels: int = 3,
        hid_S: int = 16,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 8,
        incep_ker: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
        out_frames: int | None = None,
    ):
        super().__init__()
        self.in_frames = int(in_frames)
        self.in_channels = int(in_channels)
        self.hid_S = int(hid_S)
        self.out_frames = self.in_frames if out_frames is None else int(out_frames)
        self.enc = SimVPSpatialEncoder(in_channels, hid_S, N_S)
        self.hid = SimVPTranslator(
            in_frames * hid_S, hid_T, N_T, incep_ker, groups,
            channel_out=self.out_frames * hid_S,
        )

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_raw.shape
        if T != self.in_frames:
            raise ValueError(f"expected {self.in_frames} input frames, got {T}")
        if C != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {C}")

        x = x_raw.reshape(B * T, C, H, W)
        embed, _skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)  # [B, out_frames, hid_S, H', W']
        return hid

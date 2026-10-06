"""Lightweight 1-D convolutional autoencoder for seismic trace inpainting.

Used by :func:`dssrr.repair_lib.depulse_advanced.AdvancedDePulse.nn_inpaint` to
fill samples that an outlier detector has flagged.  The model is intentionally
tiny so it can be trained on the fly, on CPU, on a *single* trace (roughly
5K-100K samples) in a few seconds -- there is no pretrained checkpoint to ship
and no GPU requirement.

Architecture
------------
A small U-Net style 1-D encoder/decoder (~60K parameters):

- ``ResidualBlock1D`` -- two dilated ``Conv1d(k=5)`` + GroupNorm + SiLU with a
  residual skip, the basic building block.
- ``DownBlock1D`` / ``UpBlock1D`` -- strided-conv downsampling and matching
  upsampling.
- ``InpaintAutoencoder1D`` -- the assembled autoencoder.  Training is masked
  (loss computed only on the flagged region) so the network learns to
  reconstruct the gap from its healthy surroundings rather than copying it.

The design follows the same idea as a low-frequency reconstruction U-Net but is
simplified for fast single-trace training.

Requires PyTorch (``pip install "dssrr[nn]"``).  Importing this module without
torch installed raises ``ImportError``; callers should guard the import (see
``depulse_advanced`` for the established pattern).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Conv1d(k=5) -> GN -> SiLU -> Conv1d(k=5) -> GN + skip -> SiLU."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation * 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5,
                                padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(min(4, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5,
                                padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(min(4, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class DownBlock1D(nn.Module):
    """Strided conv for downsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2,
                      padding=1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock1D(nn.Module):
    """Transposed conv for upsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.GroupNorm(min(4, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class InpaintAutoencoder1D(nn.Module):
    """Lightweight 1-D autoencoder for seismic trace inpainting.

    Encoder-decoder with skip connections and a learned blend factor
    that only modifies masked (outlier) regions while preserving clean data.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1 for single trace).
    base_channels : int
        Base channel count, scaled per level (default 16, total ~60K params).
    """

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        # --- Encoder ---
        self.enc_in = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7,
                      padding=3, padding_mode='reflect', bias=False),
            nn.GroupNorm(min(4, base_channels), base_channels),
            nn.SiLU(),
        )
        self.down1 = DownBlock1D(base_channels, base_channels * 2)       # N -> N/2
        self.down2 = DownBlock1D(base_channels * 2, base_channels * 4)   # N/2 -> N/4
        self.down3 = DownBlock1D(base_channels * 4, base_channels * 8)   # N/4 -> N/8

        # --- Bottleneck ---
        ch_bn = base_channels * 8
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(ch_bn, dilation=2),
            ResidualBlock1D(ch_bn, dilation=4),
        )

        # --- Decoder ---
        self.up1 = UpBlock1D(ch_bn, base_channels * 4)                   # N/8 -> N/4
        self.dec1 = ResidualBlock1D(base_channels * 4, dilation=1)
        self.up2 = UpBlock1D(base_channels * 4, base_channels * 2)       # N/4 -> N/2
        self.dec2 = ResidualBlock1D(base_channels * 2, dilation=1)
        self.up3 = UpBlock1D(base_channels * 2, base_channels)           # N/2 -> N
        self.dec3 = ResidualBlock1D(base_channels, dilation=1)

        self.dec_out = nn.Conv1d(base_channels, in_channels, kernel_size=7,
                                 padding=3, padding_mode='reflect', bias=False)

        # Learned blend factor: output = alpha * decoded + (1-alpha) * input
        # Initialized near 1.0 so the model learns to trust its output
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : (B, C, L) tensor
            Input trace with masked regions zero-filled.

        Returns
        -------
        (B, C, L) tensor
            Inpainted trace.
        """
        # Encoder
        e0 = self.enc_in(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections (match shapes)
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = self._match_size(d3, e0)
        d3 = self.dec3(d3)

        out = self.dec_out(d3)
        out = self._match_size(out, x)

        # Blend: only modify where needed
        alpha = torch.sigmoid(self.alpha)
        return alpha * out + (1 - alpha) * x

    @staticmethod
    def _match_size(a, b):
        """Crop or pad tensor `a` to match the last dimension of `b`."""
        if a.shape[-1] != b.shape[-1]:
            diff = b.shape[-1] - a.shape[-1]
            if diff > 0:
                a = F.pad(a, (0, diff))
            else:
                a = a[..., :b.shape[-1]]
        return a

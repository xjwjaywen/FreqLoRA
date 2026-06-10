"""Frequency analysis module using learnable SRM-style filters + FFT."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# SRM (Steganalysis Rich Model) kernels - proven effective for image forensics
SRM_KERNELS = [
    # Edge detector
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
    # Second-order
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32),
    # Diagonal
    np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32),
    np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32),
    # Third-order
    np.array([[0, 0, 0], [-1, 3, -3], [0, 0, 1]], dtype=np.float32),
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
    # Square
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
]


class SRMConv(nn.Module):
    """Learnable noise extraction filters initialized from SRM kernels."""

    def __init__(self, num_filters: int = 9):
        super().__init__()
        # Initialize with SRM kernels, then allow fine-tuning
        kernels = []
        for i in range(num_filters):
            k = SRM_KERNELS[i % len(SRM_KERNELS)]
            # Expand to 3 input channels (RGB)
            k3 = np.stack([k, k, k], axis=0)  # (3, 3, 3)
            kernels.append(k3)

        weight = np.stack(kernels, axis=0)  # (num_filters, 3, 3, 3)
        self.conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1, bias=False)
        self.conv.weight.data = torch.from_numpy(weight).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FrequencyAttention(nn.Module):
    """Channel attention to select discriminative frequency bands."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class FrequencyEncoder(nn.Module):
    """
    Frequency feature encoder using:
    1. Learnable SRM-style high-pass filters (noise residuals)
    2. FFT magnitude + phase spectra
    3. Channel attention for band selection
    """

    def __init__(self, out_dim: int = 256, num_srm_filters: int = 9):
        super().__init__()
        self.srm = SRMConv(num_srm_filters)
        # SRM: num_srm_filters channels + FFT: 2 channels (mag + phase)
        total_ch = num_srm_filters + 2
        self.attention = FrequencyAttention(total_ch)
        self.encoder = nn.Sequential(
            nn.Conv2d(total_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SRM noise residuals
        srm_feats = self.srm(x)

        # FFT magnitude + phase
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log1p(fft_shift.abs())
        phase = fft_shift.angle()

        # Concatenate all frequency features
        freq = torch.cat([srm_feats, magnitude, phase], dim=1)
        freq = self.attention(freq)
        return self.encoder(freq)

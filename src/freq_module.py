"""Frequency analysis module using DWT and FFT."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class DWTDecompose(nn.Module):
    """Discrete Wavelet Transform decomposition."""

    def __init__(self, wavelet="haar", level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        subbands = []
        for b in range(B):
            img_np = x[b].detach().cpu().numpy()
            bands = []
            for c in range(C):
                coeffs = pywt.wavedec2(img_np[c], self.wavelet, level=self.level)
                # Collect all detail subbands (LH, HL, HH at each level)
                for level_coeffs in coeffs[1:]:
                    for detail in level_coeffs:
                        resized = np.array(
                            torch.nn.functional.interpolate(
                                torch.from_numpy(detail).unsqueeze(0).unsqueeze(0).float(),
                                size=(H, W), mode="bilinear", align_corners=False
                            ).squeeze()
                        )
                        bands.append(resized)
                # Also include approximation
                approx = coeffs[0]
                resized_approx = np.array(
                    torch.nn.functional.interpolate(
                        torch.from_numpy(approx).unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W), mode="bilinear", align_corners=False
                    ).squeeze()
                )
                bands.append(resized_approx)
            subbands.append(np.stack(bands))

        return torch.from_numpy(np.stack(subbands)).float().to(device)


class FFTModule(nn.Module):
    """Extract FFT magnitude and phase spectra."""

    def forward(self, x: torch.Tensor) -> tuple:
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log1p(fft_shift.abs())
        phase = fft_shift.angle()
        return magnitude, phase


class FrequencyAttention(nn.Module):
    """Channel attention over frequency subbands to select discriminative bands."""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class FrequencyEncoder(nn.Module):
    """Complete frequency feature encoder: DWT + FFT + Attention."""

    def __init__(self, out_dim: int = 256, wavelet: str = "haar", dwt_level: int = 2):
        super().__init__()
        self.dwt = DWTDecompose(wavelet=wavelet, level=dwt_level)
        self.fft = FFTModule()

        # DWT: 3 channels * (3 details * dwt_level + 1 approx) = 3*(3*2+1) = 21 channels
        dwt_channels = 3 * (3 * dwt_level + 1)
        # FFT: 1 magnitude + 1 phase = 2 channels
        total_channels = dwt_channels + 2

        self.attention = FrequencyAttention(total_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(total_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dwt_feats = self.dwt(x)
        mag, phase = self.fft(x)
        freq_feats = torch.cat([dwt_feats, mag, phase], dim=1)
        freq_feats = self.attention(freq_feats)
        return self.encoder(freq_feats)

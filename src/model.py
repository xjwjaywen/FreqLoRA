"""FreqLoRA model: CLIP-ViT + Frequency Encoder + LoRA with adaptive gating."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from peft import LoraConfig, get_peft_model

from .freq_module import FrequencyEncoder


class AdaptiveGate(nn.Module):
    """Learns how much frequency information to incorporate based on spatial features."""

    def __init__(self, spatial_dim: int, freq_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(spatial_dim, freq_dim),
            nn.ReLU(),
            nn.Linear(freq_dim, freq_dim),
            nn.Sigmoid(),
        )

    def forward(self, spatial: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        g = self.gate(spatial)
        return spatial_dim_pad(spatial) + g * freq if False else torch.cat([spatial, g * freq], dim=-1)


class FreqLoRADetector(nn.Module):
    """
    AI-generated image detector with frequency-guided LoRA adaptation.

    Fusion modes:
      - "concat": [spatial; freq] (naive, baseline)
      - "gated":  [spatial; gate(spatial) * freq] (our method - AFSG)
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_target_modules: list = None,
        freq_dim: int = 256,
        fusion_dim: int = 256,
        use_freq: bool = True,
        use_lora: bool = True,
        fusion_mode: str = "gated",  # "concat" or "gated"
    ):
        super().__init__()
        self.use_freq = use_freq
        self.use_lora = use_lora
        self.fusion_mode = fusion_mode

        # CLIP visual encoder
        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.visual = clip.visual
        self.clip_dim = self.visual.output_dim
        del clip

        for param in self.visual.parameters():
            param.requires_grad = False

        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["out_proj", "c_fc", "c_proj"]
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
            )
            self.visual = get_peft_model(self.visual, lora_config)
            self.visual.print_trainable_parameters()

        if use_freq:
            self.freq_encoder = FrequencyEncoder(out_dim=freq_dim)
            if fusion_mode == "gated":
                self.gate = nn.Sequential(
                    nn.Linear(self.clip_dim, freq_dim),
                    nn.ReLU(),
                    nn.Linear(freq_dim, freq_dim),
                    nn.Sigmoid(),
                )
            total_dim = self.clip_dim + freq_dim
        else:
            self.freq_encoder = None
            total_dim = self.clip_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.visual(x)
        if isinstance(spatial, tuple):
            spatial = spatial[0]
        spatial = F.normalize(spatial, dim=-1)

        if self.use_freq and self.freq_encoder is not None:
            freq = self.freq_encoder(x)
            if self.fusion_mode == "gated":
                g = self.gate(spatial.detach())  # detach: gate doesn't affect CLIP gradients
                freq = g * freq
            features = torch.cat([spatial, freq], dim=-1)
        else:
            features = spatial

        return self.classifier(features)


class CLIPLinearProbe(nn.Module):
    """Baseline: frozen CLIP + linear classifier."""

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
    ):
        super().__init__()
        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.visual = clip.visual
        self.clip_dim = self.visual.output_dim
        del clip

        for param in self.visual.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.clip_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.visual(x)
            if isinstance(features, tuple):
                features = features[0]
            features = F.normalize(features, dim=-1)
        return self.classifier(features)

"""FreqLoRA model: CLIP-ViT + Frequency Encoder + LoRA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from peft import LoraConfig, get_peft_model

from .freq_module import FrequencyEncoder


class FreqLoRADetector(nn.Module):
    """
    AI-generated image detector with frequency-guided LoRA adaptation.

    Architecture:
      Image -> CLIP-ViT (frozen + LoRA) -> spatial features (512-d)
      Image -> FrequencyEncoder (DWT+FFT+Attention) -> freq features (256-d)
      [spatial; freq] -> Fusion MLP -> Real/Fake
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
    ):
        super().__init__()
        self.use_freq = use_freq
        self.use_lora = use_lora

        # CLIP visual encoder
        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.visual = clip.visual
        self.clip_dim = self.visual.output_dim
        del clip

        # Freeze CLIP
        for param in self.visual.parameters():
            param.requires_grad = False

        # LoRA on CLIP-ViT
        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "v_proj"]
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
            )
            self.visual = get_peft_model(self.visual, lora_config)
            self.visual.print_trainable_parameters()

        # Frequency encoder
        if use_freq:
            self.freq_encoder = FrequencyEncoder(out_dim=freq_dim)
            total_dim = self.clip_dim + freq_dim
        else:
            self.freq_encoder = None
            total_dim = self.clip_dim

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial features from CLIP-ViT
        spatial = self.visual(x)
        if isinstance(spatial, tuple):
            spatial = spatial[0]
        spatial = F.normalize(spatial, dim=-1)

        if self.use_freq and self.freq_encoder is not None:
            freq = self.freq_encoder(x)
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

"""Dual-View CLIP detector: original image + noise residual views."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from peft import LoraConfig, get_peft_model
import copy

from .freq_module import SRMConv


class DualViewDetector(nn.Module):
    """
    Dual-View CLIP detector (our method).

    View 1: Original image -> CLIP + LoRA-A -> semantic features
    View 2: SRM noise residual -> CLIP + LoRA-B -> forensic features
    Fusion: [semantic; forensic] -> MLP -> Real/Fake
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_target_modules: list = None,
        fusion_dim: int = 256,
    ):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["out_proj", "c_fc", "c_proj"]

        # Load CLIP once, then clone for two views
        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        clip_dim = clip.visual.output_dim
        self.clip_dim = clip_dim

        # View 1: semantic (original image)
        self.visual_semantic = clip.visual
        for param in self.visual_semantic.parameters():
            param.requires_grad = False
        lora_config_a = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            target_modules=lora_target_modules, lora_dropout=0.05,
        )
        self.visual_semantic = get_peft_model(self.visual_semantic, lora_config_a)
        print("Semantic view:")
        self.visual_semantic.print_trainable_parameters()

        # View 2: forensic (noise residual)
        self.visual_forensic = copy.deepcopy(clip.visual)
        for param in self.visual_forensic.parameters():
            param.requires_grad = False
        lora_config_b = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            target_modules=lora_target_modules, lora_dropout=0.05,
        )
        self.visual_forensic = get_peft_model(self.visual_forensic, lora_config_b)
        print("Forensic view:")
        self.visual_forensic.print_trainable_parameters()

        del clip

        # SRM noise extraction -> convert to 3-channel residual image
        self.srm = SRMConv(num_filters=9)
        self.residual_proj = nn.Conv2d(9, 3, kernel_size=1)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2),
        )

    def _extract_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Extract noise residual and convert to 3-channel image."""
        srm_out = self.srm(x)  # (B, 9, H, W)
        residual = self.residual_proj(srm_out)  # (B, 3, H, W)
        # Normalize to similar range as original image
        residual = torch.tanh(residual)
        return residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # View 1: semantic features from original image
        feat_semantic = self.visual_semantic(x)
        if isinstance(feat_semantic, tuple):
            feat_semantic = feat_semantic[0]
        feat_semantic = F.normalize(feat_semantic, dim=-1)

        # View 2: forensic features from noise residual
        residual = self._extract_residual(x)
        feat_forensic = self.visual_forensic(residual)
        if isinstance(feat_forensic, tuple):
            feat_forensic = feat_forensic[0]
        feat_forensic = F.normalize(feat_forensic, dim=-1)

        # Fusion
        features = torch.cat([feat_semantic, feat_forensic], dim=-1)
        return self.classifier(features)


class SingleViewLoRA(nn.Module):
    """Baseline: single CLIP + LoRA (no dual view)."""

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_target_modules: list = None,
        fusion_dim: int = 256,
    ):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["out_proj", "c_fc", "c_proj"]

        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.visual = clip.visual
        self.clip_dim = self.visual.output_dim
        del clip

        for param in self.visual.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            target_modules=lora_target_modules, lora_dropout=0.05,
        )
        self.visual = get_peft_model(self.visual, lora_config)
        self.visual.print_trainable_parameters()

        self.classifier = nn.Sequential(
            nn.Linear(self.clip_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.visual(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        feat = F.normalize(feat, dim=-1)
        return self.classifier(feat)


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

"""PatchLTD: Patch-Level Layer Transition Discrepancy for AI-Generated Image Detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from peft import LoraConfig, get_peft_model


class PatchLTDDetector(nn.Module):
    """
    Patch-Level Layer Transition Discrepancy detector.

    Key insight: Real images show smooth patch-level feature transitions
    across CLIP layers. AI-generated images show anomalous transitions
    at specific patches/layers.

    Architecture:
      Image -> CLIP-ViT + LoRA
        ├── CLS token (semantic feature)
        └── Patch tokens from layers [3,6,9,12]
              ├── Transition d1 = layer6 - layer3 (per patch)
              ├── Transition d2 = layer9 - layer6 (per patch)
              └── Transition d3 = layer12 - layer9 (per patch)
                    ↓
              Projection → Aggregation → transition feature
                    ↓
              [CLS; transition] → Classifier → Real/Fake
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_target_modules: list = None,
        selected_layers: list = None,
        transition_proj_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["out_proj", "c_fc", "c_proj"]
        if selected_layers is None:
            selected_layers = [2, 5, 8, 11]  # 0-indexed: layers 3, 6, 9, 12

        self.selected_layers = selected_layers
        self.num_transitions = len(selected_layers) - 1

        # Load CLIP
        clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.visual = clip.visual
        self.clip_dim = self.visual.output_dim
        self.hidden_dim = self.visual.transformer.width
        del clip

        # Register hooks BEFORE LoRA wrapping
        self.intermediate_features = {}
        for idx in self.selected_layers:
            block = self.visual.transformer.resblocks[idx]
            block.register_forward_hook(self._make_hook(idx))

        # Freeze + LoRA
        for param in self.visual.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            target_modules=lora_target_modules, lora_dropout=0.05,
        )
        self.visual = get_peft_model(self.visual, lora_config)
        print("CLIP + LoRA:")
        self.visual.print_trainable_parameters()

        # Transition projection: 768-d -> proj_dim
        self.transition_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, transition_proj_dim),
            nn.LayerNorm(transition_proj_dim),
            nn.GELU(),
        )

        # Transition aggregator: processes patch-level transitions
        self.transition_aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transition_proj_dim,
                nhead=4,
                dim_feedforward=transition_proj_dim * 2,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.transition_cls = nn.Parameter(torch.randn(1, 1, transition_proj_dim))

        transition_feat_dim = transition_proj_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_dim + transition_feat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2),
        )

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self.intermediate_features[layer_idx] = output
        return hook

    def _compute_patch_transitions(self) -> torch.Tensor:
        """Compute patch-level layer transition discrepancies."""
        layer_patches = []
        for idx in self.selected_layers:
            feat = self.intermediate_features[idx]
            # open_clip uses (seq_len, batch, dim) format
            if feat.dim() == 3 and feat.shape[0] != feat.shape[1]:
                feat = feat.permute(1, 0, 2)  # -> (batch, seq_len, dim)
            patches = feat[:, 1:, :]  # exclude CLS token: (B, num_patches, dim)
            layer_patches.append(patches)

        # Compute transitions between adjacent selected layers
        transitions = []
        for k in range(self.num_transitions):
            d = layer_patches[k + 1] - layer_patches[k]  # (B, num_patches, dim)
            d_proj = self.transition_proj(d)  # (B, num_patches, proj_dim)
            transitions.append(d_proj)

        # Stack all transitions: (B, num_transitions * num_patches, proj_dim)
        all_transitions = torch.cat(transitions, dim=1)
        return all_transitions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.intermediate_features.clear()

        # Forward through CLIP (hooks capture intermediate features)
        cls_feat = self.visual(x)
        if isinstance(cls_feat, tuple):
            cls_feat = cls_feat[0]
        cls_feat = F.normalize(cls_feat, dim=-1)

        # Compute patch-level transitions
        transitions = self._compute_patch_transitions()  # (B, N_trans*N_patches, proj_dim)

        # Prepend learnable CLS token for aggregation
        B = transitions.shape[0]
        cls_token = self.transition_cls.expand(B, -1, -1)
        tokens = torch.cat([cls_token, transitions], dim=1)

        # Aggregate via Transformer
        aggregated = self.transition_aggregator(tokens)
        transition_feat = aggregated[:, 0, :]  # take CLS output: (B, proj_dim)

        # Fuse CLS + transition features
        features = torch.cat([cls_feat, transition_feat], dim=-1)
        return self.classifier(features)


class SingleViewLoRA(nn.Module):
    """Baseline: CLIP + LoRA, no patch transition analysis."""

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

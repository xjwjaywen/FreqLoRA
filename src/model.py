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
        patch_mode: str = "transformer",  # "transformer", "meanpool", "cls_only"
    ):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["out_proj", "c_fc", "c_proj"]
        if selected_layers is None:
            selected_layers = [2, 5, 8, 11]  # 0-indexed: layers 3, 6, 9, 12

        self.selected_layers = selected_layers
        self.num_transitions = len(selected_layers) - 1
        self.patch_mode = patch_mode

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

    def _get_layer_tokens(self, batch_size: int):
        """Extract tokens from hooked layers in (batch, seq, dim) format."""
        layer_tokens = []
        for idx in self.selected_layers:
            feat = self.intermediate_features[idx]
            if feat.dim() == 3:
                if feat.shape[0] == batch_size:
                    pass
                elif feat.shape[1] == batch_size:
                    feat = feat.permute(1, 0, 2)
                else:
                    feat = feat.permute(1, 0, 2)
            layer_tokens.append(feat)
        return layer_tokens

    def _compute_transitions(self, batch_size: int) -> torch.Tensor:
        """Compute layer transition features based on patch_mode."""
        layer_tokens = self._get_layer_tokens(batch_size)

        if self.patch_mode == "cls_only":
            # LTD-style: only use CLS token (index 0)
            cls_tokens = [t[:, 0, :] for t in layer_tokens]  # list of (B, dim)
            transitions = []
            for k in range(self.num_transitions):
                d = cls_tokens[k + 1] - cls_tokens[k]  # (B, dim)
                d_proj = self.transition_proj(d)  # (B, proj_dim)
                transitions.append(d_proj)
            # (B, num_transitions, proj_dim)
            return torch.stack(transitions, dim=1)
        else:
            # Patch-level: use all patch tokens (exclude CLS)
            layer_patches = [t[:, 1:, :] for t in layer_tokens]
            transitions = []
            for k in range(self.num_transitions):
                d = layer_patches[k + 1] - layer_patches[k]  # (B, num_patches, dim)
                d_proj = self.transition_proj(d)  # (B, num_patches, proj_dim)
                transitions.append(d_proj)
            # (B, num_transitions * num_patches, proj_dim)
            return torch.cat(transitions, dim=1)

    def _aggregate_transitions(self, transitions: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Aggregate projected transition tokens into one forensic feature."""
        if self.patch_mode == "meanpool":
            return transitions.mean(dim=1)
        if self.patch_mode == "cls_only":
            return transitions.mean(dim=1)

        cls_token = self.transition_cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, transitions], dim=1)
        aggregated = self.transition_aggregator(tokens)
        return aggregated[:, 0, :]

    def _encode(self, x: torch.Tensor):
        self.intermediate_features.clear()

        # Forward through CLIP (hooks capture intermediate features)
        cls_feat = self.visual(x)
        if isinstance(cls_feat, tuple):
            cls_feat = cls_feat[0]
        cls_feat = F.normalize(cls_feat, dim=-1)

        # Compute transitions
        B = x.shape[0]
        transitions = self._compute_transitions(B)
        transition_feat = self._aggregate_transitions(transitions, B)

        # Fuse CLS + transition features
        fused_feat = torch.cat([cls_feat, transition_feat], dim=-1)
        return fused_feat, transition_feat

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        fused_feat, _ = self._encode(x)
        return fused_feat

    def forward_with_features(self, x: torch.Tensor):
        fused_feat, transition_feat = self._encode(x)
        return self.classifier(fused_feat), transition_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(features)


class SingleViewLoRA(nn.Module):
    """Baseline: CLIP + LoRA, no patch transition analysis.
    Uses a larger MLP head to match PatchLTD parameter count for fair comparison.
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

        # Larger MLP to roughly match PatchLTD's total parameter count
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.visual(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        return F.normalize(feat, dim=-1)

    def forward_with_features(self, x: torch.Tensor):
        feat = self.forward_features(x)
        return self.classifier(feat), feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.visual(x)
            if isinstance(features, tuple):
                features = features[0]
            features = F.normalize(features, dim=-1)
        return features

    def forward_with_features(self, x: torch.Tensor):
        features = self.forward_features(x)
        return self.classifier(features), features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(features)

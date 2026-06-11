"""Degradation-Consistent Paired Training (DCPT).

For each training batch, create a degraded copy and enforce prediction
consistency between clean and degraded views. This forces the model to
learn degradation-invariant forensic features.

Reference: arXiv 2604.10102
"""
import torch
import torch.nn.functional as F
import random
import io
from PIL import Image
from torchvision import transforms


def apply_random_degradation(images: torch.Tensor) -> torch.Tensor:
    """Apply random degradation to a batch of images (on GPU).

    Degradations: JPEG compression, Gaussian blur, resize.
    Applied in-batch for efficiency.
    """
    B, C, H, W = images.shape
    device = images.device

    degraded = images.clone()

    # Random degradation type per sample
    for i in range(B):
        deg_type = random.choice(["jpeg", "blur", "resize", "jpeg"])  # jpeg weighted 2x

        if deg_type == "jpeg":
            # JPEG compression: convert to PIL, compress, back to tensor
            img = images[i].cpu()
            img_pil = transforms.ToPILImage()(img * 0.5 + 0.5)  # denormalize
            quality = random.randint(20, 80)
            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img_pil = Image.open(buf).convert("RGB")
            img_t = transforms.ToTensor()(img_pil)
            degraded[i] = (img_t * 2 - 1).to(device)  # renormalize to [-1, 1]

        elif deg_type == "blur":
            # Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            degraded[i] = transforms.GaussianBlur(kernel_size, sigma)(images[i])

        elif deg_type == "resize":
            # Downscale then upscale
            scale = random.uniform(0.3, 0.8)
            small_h, small_w = int(H * scale), int(W * scale)
            small = F.interpolate(images[i:i+1], size=(small_h, small_w), mode="bilinear", align_corners=False)
            degraded[i] = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

    return degraded


def dcpt_consistency_loss(
    logits_clean: torch.Tensor,
    logits_degraded: torch.Tensor,
    lambda_feat: float = 0.5,
    lambda_pred: float = 0.5,
) -> torch.Tensor:
    """Compute DCPT consistency loss between clean and degraded predictions.

    L = lambda_pred * symmetric_KL(p_clean, p_degraded)
    """
    p_clean = F.softmax(logits_clean, dim=1)
    p_deg = F.softmax(logits_degraded, dim=1)

    # Symmetric KL divergence
    kl_1 = F.kl_div(p_deg.log(), p_clean, reduction="batchmean")
    kl_2 = F.kl_div(p_clean.log(), p_deg, reduction="batchmean")
    sym_kl = (kl_1 + kl_2) / 2

    return lambda_pred * sym_kl

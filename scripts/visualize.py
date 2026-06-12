"""Generate visualizations for the paper."""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PatchLTDDetector
from src.dataset import GenImageDataset, get_transforms


def load_model(checkpoint_dir, config_path="configs/default.yaml", device="cuda"):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]
    model = PatchLTDDetector(
        model_cfg["clip_model"], model_cfg["clip_pretrained"],
        lora_rank=model_cfg["lora_rank"], lora_alpha=model_cfg["lora_alpha"],
        lora_target_modules=model_cfg["lora_target_modules"],
    )
    state = torch.load(f"{checkpoint_dir}/best.pt", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def get_patch_transition_heatmap(model, image_tensor, device="cuda"):
    """Extract per-patch transition norms as a heatmap."""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        model.intermediate_features.clear()
        _ = model.visual(image_tensor)
        B = 1
        layer_tokens = model._get_layer_tokens(B)
        layer_patches = [t[:, 1:, :] for t in layer_tokens]
        all_norms = []
        for k in range(len(layer_patches) - 1):
            d = layer_patches[k + 1] - layer_patches[k]
            norms = d.norm(dim=-1).squeeze(0)
            all_norms.append(norms)
        avg_norms = torch.stack(all_norms).mean(dim=0)
        grid_size = int(avg_norms.shape[0] ** 0.5)
        heatmap = avg_norms.reshape(grid_size, grid_size).cpu().numpy()
    return heatmap


def get_fused_feature(model, image_tensor, device="cuda"):
    """Extract the full [CLS; transition] fused feature."""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        model.intermediate_features.clear()
        cls_feat = model.visual(image_tensor)
        if isinstance(cls_feat, tuple):
            cls_feat = cls_feat[0]
        cls_feat = F.normalize(cls_feat, dim=-1)

        B = 1
        transitions = model._compute_transitions(B)
        cls_token = model.transition_cls.expand(B, -1, -1)
        tokens = torch.cat([cls_token, transitions], dim=1)
        aggregated = model.transition_aggregator(tokens)
        transition_feat = aggregated[:, 0, :]

        fused = torch.cat([cls_feat, transition_feat], dim=-1)
    return fused.squeeze(0).cpu().numpy()


def viz_difference_heatmap(model, dataset, output_dir, num_samples=100, device="cuda"):
    """Difference heatmap: avg(fake) - avg(real) transition norms."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_heatmaps, fake_heatmaps = [], []
    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        heatmap = get_patch_transition_heatmap(model, img_tensor, device)
        if label == 0 and len(real_heatmaps) < num_samples:
            real_heatmaps.append(heatmap)
        elif label == 1 and len(fake_heatmaps) < num_samples:
            fake_heatmaps.append(heatmap)
        if len(real_heatmaps) >= num_samples and len(fake_heatmaps) >= num_samples:
            break

    print(f"  Collected {len(real_heatmaps)} real + {len(fake_heatmaps)} fake heatmaps")
    avg_real = np.mean(real_heatmaps, axis=0)
    avg_fake = np.mean(fake_heatmaps, axis=0)
    diff = avg_fake - avg_real

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(avg_real, cmap="hot", interpolation="bilinear")
    axes[0].set_title("Real (avg)", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(avg_fake, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Fake (avg)", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, cmap="RdBu_r", interpolation="bilinear", vmin=-diff.max(), vmax=diff.max())
    axes[2].set_title("Fake - Real (difference)", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_dir / "difference_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_dir / 'difference_heatmap.png'}")


def viz_transition_norm_histogram(model, dataset, output_dir, num_samples=200, device="cuda"):
    """Histogram of per-image mean transition norms: real vs fake."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_norms, fake_norms = [], []
    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        heatmap = get_patch_transition_heatmap(model, img_tensor, device)
        mean_norm = heatmap.mean()
        if label == 0 and len(real_norms) < num_samples:
            real_norms.append(mean_norm)
        elif label == 1 and len(fake_norms) < num_samples:
            fake_norms.append(mean_norm)
        if len(real_norms) >= num_samples and len(fake_norms) >= num_samples:
            break
    print(f"  Collected {len(real_norms)} real + {len(fake_norms)} fake norms")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_norms, bins=30, alpha=0.6, color="blue", label="Real", density=True)
    ax.hist(fake_norms, bins=30, alpha=0.6, color="red", label="Fake", density=True)
    ax.set_xlabel("Mean Patch Transition Norm", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Patch Transition Norms", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "transition_norm_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_dir / 'transition_norm_histogram.png'}")


def viz_jpeg_histogram(model, dataset_cls, data_dir, generator, output_dir,
                       num_samples=200, device="cuda"):
    """Compare transition norm distributions: clean vs JPEG compressed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_transform = get_transforms(224, is_train=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, q in enumerate([None, 50, 30]):
        ds = dataset_cls(data_dir, generator, split="val",
                         transform=val_transform, max_per_class=num_samples,
                         jpeg_quality=q)
        real_norms, fake_norms = [], []
        for i in range(len(ds)):
            img_tensor, label = ds[i]
            heatmap = get_patch_transition_heatmap(model, img_tensor, device)
            if label == 0 and len(real_norms) < num_samples:
                real_norms.append(heatmap.mean())
            elif label == 1 and len(fake_norms) < num_samples:
                fake_norms.append(heatmap.mean())
            if len(real_norms) >= num_samples and len(fake_norms) >= num_samples:
                break

        title = "Clean" if q is None else f"JPEG Q={q}"
        axes[col].hist(real_norms, bins=25, alpha=0.6, color="blue", label="Real", density=True)
        axes[col].hist(fake_norms, bins=25, alpha=0.6, color="red", label="Fake", density=True)
        axes[col].set_title(title, fontsize=13)
        axes[col].set_xlabel("Mean Transition Norm", fontsize=11)
        axes[col].legend(fontsize=10)

    fig.suptitle("Transition Norm Distributions Under JPEG Compression", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "jpeg_norm_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_dir / 'jpeg_norm_histogram.png'}")


def viz_tsne_fused(model, dataset, output_dir, num_samples=500, device="cuda"):
    """t-SNE of FUSED features [CLS; transition], not transition alone."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels = [], []
    model.eval()
    real_count, fake_count = 0, 0
    half = num_samples // 2
    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        if label == 0 and real_count >= half:
            continue
        if label == 1 and fake_count >= half:
            continue
        feat = get_fused_feature(model, img_tensor, device)
        features.append(feat)
        labels.append(label)
        if label == 0:
            real_count += 1
        else:
            fake_count += 1
        if real_count >= half and fake_count >= half:
            break
    print(f"  Collected {real_count} real + {fake_count} fake for t-SNE")

    features = np.array(features)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot fake first so real doesn't cover it
    fake_mask = labels == 1
    real_mask = labels == 0
    ax.scatter(embedded[fake_mask, 0], embedded[fake_mask, 1],
               c="red", alpha=0.4, s=10, label="Fake")
    ax.scatter(embedded[real_mask, 0], embedded[real_mask, 1],
               c="blue", alpha=0.4, s=10, label="Real")
    ax.legend(fontsize=12)
    ax.set_title("t-SNE of Fused Features [CLS + Transition]", fontsize=14)
    ax.axis("off")
    plt.savefig(output_dir / "tsne_fused.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_dir / 'tsne_fused.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/GenImage")
    parser.add_argument("--generator", type=str, default="sd14")
    parser.add_argument("--output_dir", type=str, default="./visualizations")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)
    val_transform = get_transforms(224, is_train=False)
    dataset = GenImageDataset(args.data_dir, args.generator, split="val",
                               transform=val_transform, max_per_class=500)

    print("1/4 Difference heatmap...")
    viz_difference_heatmap(model, dataset, args.output_dir, device=args.device)

    print("2/4 Transition norm histogram...")
    viz_transition_norm_histogram(model, dataset, args.output_dir, device=args.device)

    print("3/4 JPEG norm histogram...")
    viz_jpeg_histogram(model, GenImageDataset, args.data_dir,
                       args.generator, args.output_dir, device=args.device)

    print("4/4 t-SNE (fused features)...")
    viz_tsne_fused(model, dataset, args.output_dir, device=args.device)

    print("All visualizations done!")

"""Generate patch transition heatmaps and t-SNE visualizations."""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
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
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        model.intermediate_features.clear()
        _ = model.visual(image_tensor)

        B = 1
        layer_tokens = model._get_layer_tokens(B)
        # Get patch tokens (exclude CLS)
        layer_patches = [t[:, 1:, :] for t in layer_tokens]

        # Compute transition norms per patch
        all_norms = []
        for k in range(len(layer_patches) - 1):
            d = layer_patches[k + 1] - layer_patches[k]  # (1, 196, 768)
            norms = d.norm(dim=-1).squeeze(0)  # (196,)
            all_norms.append(norms)

        # Average across transitions
        avg_norms = torch.stack(all_norms).mean(dim=0)  # (196,)
        # Reshape to spatial grid (14x14 for ViT-B/16 with 224x224 input)
        grid_size = int(avg_norms.shape[0] ** 0.5)
        heatmap = avg_norms.reshape(grid_size, grid_size).cpu().numpy()

    return heatmap


def visualize_heatmaps(model, dataset, output_dir, num_samples=4, device="cuda"):
    """Generate side-by-side heatmaps for real vs fake images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect real and fake samples
    real_indices = [i for i in range(len(dataset)) if dataset.labels[i] == 0]
    fake_indices = [i for i in range(len(dataset)) if dataset.labels[i] == 1]

    fig, axes = plt.subplots(2, num_samples * 2, figsize=(num_samples * 8, 6))
    fig.suptitle("Patch Transition Heatmaps: Real vs Fake", fontsize=16)

    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for col, (idx, label) in enumerate(
        [(real_indices[i], "Real") for i in range(num_samples)] +
        [(fake_indices[i], "Fake") for i in range(num_samples)]
    ):
        img_tensor, _ = dataset[idx]
        heatmap = get_patch_transition_heatmap(model, img_tensor, device)

        # Show original image
        img_show = denorm(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, col].imshow(img_show)
        axes[0, col].set_title(f"{label}", fontsize=12)
        axes[0, col].axis("off")

        # Show heatmap
        axes[1, col].imshow(heatmap, cmap="hot", interpolation="bilinear")
        axes[1, col].set_title(f"Transition Norm", fontsize=10)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "patch_transition_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmaps to {output_dir / 'patch_transition_heatmaps.png'}")


def visualize_jpeg_robustness(model, dataset_class, data_dir, generator, output_dir,
                               num_samples=200, device="cuda"):
    """Show how transition features change under JPEG compression."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_transform = get_transforms(224, is_train=False)
    qualities = [None, 95, 75, 50, 30]
    labels_list = ["Clean", "Q=95", "Q=75", "Q=50", "Q=30"]

    all_heatmaps_real = {q: [] for q in qualities}
    all_heatmaps_fake = {q: [] for q in qualities}

    for q in qualities:
        ds = GenImageDataset(data_dir, generator, split="val",
                              transform=val_transform, max_per_class=num_samples,
                              jpeg_quality=q)
        for i in range(min(num_samples, len(ds))):
            img_tensor, label = ds[i]
            heatmap = get_patch_transition_heatmap(model, img_tensor, device)
            if label == 0:
                all_heatmaps_real[q].append(heatmap)
            else:
                all_heatmaps_fake[q].append(heatmap)

    # Plot average heatmaps
    fig, axes = plt.subplots(2, len(qualities), figsize=(len(qualities) * 4, 6))
    fig.suptitle("Average Transition Heatmaps Under JPEG Compression", fontsize=14)

    for col, (q, label) in enumerate(zip(qualities, labels_list)):
        if all_heatmaps_real[q]:
            avg_real = np.mean(all_heatmaps_real[q], axis=0)
            axes[0, col].imshow(avg_real, cmap="hot", interpolation="bilinear")
        axes[0, col].set_title(f"Real - {label}", fontsize=10)
        axes[0, col].axis("off")

        if all_heatmaps_fake[q]:
            avg_fake = np.mean(all_heatmaps_fake[q], axis=0)
            axes[1, col].imshow(avg_fake, cmap="hot", interpolation="bilinear")
        axes[1, col].set_title(f"Fake - {label}", fontsize=10)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "jpeg_transition_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved JPEG comparison to {output_dir / 'jpeg_transition_comparison.png'}")


def visualize_tsne(model, dataset, output_dir, num_samples=500, device="cuda"):
    """t-SNE of transition features: real vs fake."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    labels = []
    model.eval()

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img_tensor, label = dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)

            model.intermediate_features.clear()
            cls_feat = model.visual(img_tensor)
            if isinstance(cls_feat, tuple):
                cls_feat = cls_feat[0]

            B = 1
            transitions = model._compute_transitions(B)

            if model.patch_mode == "transformer":
                cls_token = model.transition_cls.expand(B, -1, -1)
                tokens = torch.cat([cls_token, transitions], dim=1)
                aggregated = model.transition_aggregator(tokens)
                feat = aggregated[:, 0, :].squeeze(0)
            else:
                feat = transitions.mean(dim=1).squeeze(0)

            features.append(feat.cpu().numpy())
            labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    real_mask = labels == 0
    fake_mask = labels == 1
    plt.scatter(embedded[real_mask, 0], embedded[real_mask, 1],
                c="blue", alpha=0.5, s=10, label="Real")
    plt.scatter(embedded[fake_mask, 0], embedded[fake_mask, 1],
                c="red", alpha=0.5, s=10, label="Fake")
    plt.legend(fontsize=12)
    plt.title("t-SNE of Patch Transition Features", fontsize=14)
    plt.axis("off")
    plt.savefig(output_dir / "tsne_transitions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE to {output_dir / 'tsne_transitions.png'}")


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

    print("Generating heatmaps...")
    visualize_heatmaps(model, dataset, args.output_dir, device=args.device)

    print("Generating JPEG comparison...")
    visualize_jpeg_robustness(model, GenImageDataset, args.data_dir,
                               args.generator, args.output_dir, device=args.device)

    print("Generating t-SNE...")
    visualize_tsne(model, dataset, args.output_dir, device=args.device)

    print("Done!")

"""Generate visualizations for the paper."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def _lazy_imports():
    """Import heavy deps only when needed (not for --only_curve)."""
    import torch
    import torch.nn.functional as F
    from sklearn.manifold import TSNE
    from src.model import PatchLTDDetector
    from src.dataset import GenImageDataset, get_transforms
    return torch, F, TSNE, PatchLTDDetector, GenImageDataset, get_transforms


def load_model(checkpoint_dir, config_path="configs/default.yaml", device="cuda"):
    torch, F, TSNE, PatchLTDDetector, GenImageDataset, get_transforms = _lazy_imports()
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


def _collect_balanced(dataset, num_per_class):
    """Collect balanced real/fake indices from dataset."""
    real_idx, fake_idx = [], []
    for i in range(len(dataset)):
        if dataset.labels[i] == 0 and len(real_idx) < num_per_class:
            real_idx.append(i)
        elif dataset.labels[i] == 1 and len(fake_idx) < num_per_class:
            fake_idx.append(i)
        if len(real_idx) >= num_per_class and len(fake_idx) >= num_per_class:
            break
    return real_idx, fake_idx


def get_patch_transition_heatmap(model, image_tensor, device="cuda"):
    """Extract per-patch transition norms as a heatmap."""
    import torch
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
    import torch
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        fused_feat = model.forward_features(image_tensor)
    return fused_feat.squeeze(0).cpu().numpy()


def viz_difference_heatmap(model, dataset, output_dir, num_samples=100, device="cuda"):
    """Figure 2: Difference heatmap -- avg(fake) - avg(real) transition norms."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_idx, fake_idx = _collect_balanced(dataset, num_samples)
    print(f"  Collected {len(real_idx)} real + {len(fake_idx)} fake heatmaps")

    real_heatmaps = [get_patch_transition_heatmap(model, dataset[i][0], device) for i in real_idx]
    fake_heatmaps = [get_patch_transition_heatmap(model, dataset[i][0], device) for i in fake_idx]

    avg_real = np.mean(real_heatmaps, axis=0)
    avg_fake = np.mean(fake_heatmaps, axis=0)
    diff = avg_fake - avg_real

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = min(avg_real.min(), avg_fake.min())
    vmax = max(avg_real.max(), avg_fake.max())

    im0 = axes[0].imshow(avg_real, cmap="hot", interpolation="bilinear", vmin=vmin, vmax=vmax)
    axes[0].set_title("Real (avg)", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(avg_fake, cmap="hot", interpolation="bilinear", vmin=vmin, vmax=vmax)
    axes[1].set_title("Fake (avg)", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    abs_max = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap="RdBu_r", interpolation="bilinear", vmin=-abs_max, vmax=abs_max)
    axes[2].set_title("Fake − Real", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_dir / "difference_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_dir / 'difference_heatmap.png'}")


def viz_tsne_fused(model, dataset, output_dir, num_per_class=250, device="cuda"):
    """t-SNE of fused features [CLS + transition]."""
    from sklearn.manifold import TSNE
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_idx, fake_idx = _collect_balanced(dataset, num_per_class)
    print(f"  Collected {len(real_idx)} real + {len(fake_idx)} fake for t-SNE")

    indices = real_idx + fake_idx
    features = np.array([get_fused_feature(model, dataset[i][0], device) for i in indices])
    labels = np.array([0] * len(real_idx) + [1] * len(fake_idx))

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    real_mask = labels == 0
    fake_mask = labels == 1
    ax.scatter(embedded[real_mask, 0], embedded[real_mask, 1],
               c="tab:blue", alpha=0.5, s=12, label="Real", edgecolors="none")
    ax.scatter(embedded[fake_mask, 0], embedded[fake_mask, 1],
               c="tab:red", alpha=0.5, s=12, label="Fake", edgecolors="none")
    ax.legend(fontsize=12, markerscale=3)
    ax.set_title("t-SNE of Fused Features [CLS + Transition]", fontsize=14)
    ax.axis("off")
    plt.savefig(output_dir / "tsne_fused.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_dir / 'tsne_fused.png'}")


def viz_jpeg_robustness_curve(results_dir, output_dir):
    """Figure 3: Accuracy vs JPEG quality factor curve (multi-gen, 5 seeds)."""
    import re
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)

    methods = {
        "CLIP+LoRA": "single_lora",
        "CLS-LTD": "cls_ltd",
        "PatchLTD-meanpool": "patchltd_meanpool",
        "PatchLTD (ours)": "patchltd",
    }
    colors = {
        "CLIP+LoRA": "tab:gray",
        "CLS-LTD": "tab:orange",
        "PatchLTD-meanpool": "tab:green",
        "PatchLTD (ours)": "tab:red",
    }
    markers = {
        "CLIP+LoRA": "s",
        "CLS-LTD": "^",
        "PatchLTD-meanpool": "D",
        "PatchLTD (ours)": "o",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    qualities = [100, 95, 75, 50, 30]

    for label, prefix in methods.items():
        seed_data = {q: [] for q in qualities}
        for seed in [42, 123, 456, 789, 1024]:
            fpath = results_dir / f"{prefix}_multi_seed{seed}.txt"
            if not fpath.exists():
                continue
            text = fpath.read_text()
            # Clean acc
            m = re.search(r"best_val_acc: ([0-9.]+)", text)
            if m:
                seed_data[100].append(float(m.group(1)))
            for line in text.split("\n"):
                mq = re.match(r"Q=(\d+): acc=([0-9.]+)", line)
                if mq:
                    seed_data[int(mq.group(1))].append(float(mq.group(2)))

        means, stds, qs = [], [], []
        for q in qualities:
            if seed_data[q]:
                means.append(np.mean(seed_data[q]) * 100)
                stds.append(np.std(seed_data[q]) * 100)
                qs.append(q)

        if means:
            ax.errorbar(qs, means, yerr=stds, label=label,
                        color=colors[label], marker=markers[label],
                        linewidth=2, capsize=4, markersize=7)

    ax.set_xlabel("JPEG Quality Factor", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("JPEG Robustness (Multi-Generator Training)", fontsize=14)
    ax.set_xticks(qualities)
    ax.set_xticklabels(["Clean", "95", "75", "50", "30"])
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim([75, 100])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "jpeg_robustness_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_dir / 'jpeg_robustness_curve.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint dir (for heatmap/tsne)")
    parser.add_argument("--data_dir", type=str, default="./data/GenImage")
    parser.add_argument("--generator", type=str, default="sd14")
    parser.add_argument("--output_dir", type=str, default="./visualizations")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--only_curve", action="store_true",
                        help="Only generate the JPEG robustness curve (no model needed)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.only_curve:
        print("Generating JPEG robustness curve...")
        viz_jpeg_robustness_curve(args.results_dir, args.output_dir)
        print("Done!")
        sys.exit(0)

    if args.checkpoint is None:
        print("ERROR: --checkpoint required for heatmap/tsne visualizations")
        sys.exit(1)

    torch, F, TSNE, PatchLTDDetector, GenImageDataset, get_transforms = _lazy_imports()
    model = load_model(args.checkpoint, device=args.device)
    val_transform = get_transforms(224, is_train=False)
    dataset = GenImageDataset(args.data_dir, args.generator, split="val",
                               transform=val_transform, max_per_class=500)

    print("1/3 Difference heatmap...")
    viz_difference_heatmap(model, dataset, args.output_dir, device=args.device)

    print("2/3 t-SNE (fused features)...")
    viz_tsne_fused(model, dataset, args.output_dir, device=args.device)

    print("3/3 JPEG robustness curve...")
    viz_jpeg_robustness_curve(args.results_dir, args.output_dir)

    print("All visualizations done!")

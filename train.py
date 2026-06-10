"""Training script for PatchLTD detector."""
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import numpy as np

from src.model import PatchLTDDetector, SingleViewLoRA, CLIPLinearProbe
from src.dataset import GenImageDataset, get_transforms, get_available_generators


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = map(np.array, [all_preds, all_labels, all_probs])
    acc = accuracy_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_probs)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    return {"acc": acc, "ap": ap, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_gen", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="patchltd",
                        choices=["patchltd", "single_lora", "clip_linear"])
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=2000)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.data_dir:
        config["dataset"]["data_dir"] = args.data_dir
    if args.train_gen:
        config["dataset"]["train_generator"] = args.train_gen

    data_dir = config["dataset"]["data_dir"]
    train_gen = config["dataset"]["train_generator"]
    method = args.method
    image_size = config["dataset"]["image_size"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    output_dir = Path(config["output"]["output_dir"]) / f"{method}_{train_gen}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PatchLTD | method={method} | train_gen={train_gen}")
    print(f"{'='*60}")

    # --- Dataset ---
    train_transform = get_transforms(image_size, is_train=True, jpeg_aug=True)
    val_transform = get_transforms(image_size, is_train=False)

    train_dataset = GenImageDataset(data_dir, train_gen, split="train",
                                     transform=train_transform, max_per_class=args.max_train)
    val_dataset = GenImageDataset(data_dir, train_gen, split="val",
                                   transform=val_transform, max_per_class=args.max_test)
    if len(train_dataset) == 0:
        print("ERROR: No training data"); return

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"],
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    if method == "clip_linear":
        model = CLIPLinearProbe(model_cfg["clip_model"], model_cfg["clip_pretrained"])
    elif method == "single_lora":
        model = SingleViewLoRA(
            model_cfg["clip_model"], model_cfg["clip_pretrained"],
            lora_rank=model_cfg["lora_rank"], lora_alpha=model_cfg["lora_alpha"],
            lora_target_modules=model_cfg["lora_target_modules"],
        )
    else:  # patchltd
        model = PatchLTDDetector(
            model_cfg["clip_model"], model_cfg["clip_pretrained"],
            lora_rank=model_cfg["lora_rank"], lora_alpha=model_cfg["lora_alpha"],
            lora_target_modules=model_cfg["lora_target_modules"],
        )

    model = model.to(args.device)

    # --- Training ---
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg["learning_rate"],
                                   weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    best_acc = 0.0
    for epoch in range(train_cfg["epochs"]):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")

        for images, labels in progress:
            images, labels = images.to(args.device), labels.to(args.device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{running_loss/(progress.n+1):.4f}", acc=f"{correct/total:.4f}")

        scheduler.step()

        if len(val_dataset) > 0:
            val_r = evaluate(model, val_loader, args.device)
            print(f"  Val: acc={val_r['acc']:.4f} ap={val_r['ap']:.4f} auc={val_r['auc']:.4f}")
            if val_r['acc'] > best_acc:
                best_acc = val_r['acc']
                torch.save(model.state_dict(), str(output_dir / "best.pt"))

    print(f"\nBest val acc: {best_acc:.4f}")

    # --- Cross-Generator Evaluation ---
    print(f"\n{'='*60}")
    print("Cross-Generator Evaluation")
    print(f"{'='*60}")

    if (output_dir / "best.pt").exists():
        model.load_state_dict(torch.load(str(output_dir / "best.pt"), weights_only=True))

    generators = get_available_generators(data_dir)
    all_results = {}
    for gen in generators:
        ds = GenImageDataset(data_dir, gen, split="val",
                              transform=val_transform, max_per_class=args.max_test)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
        r = evaluate(model, loader, args.device)
        all_results[gen] = r
        marker = " <-- train" if gen == train_gen else ""
        print(f"  {gen:<35} acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}{marker}")

    if all_results:
        unseen = {k: v for k, v in all_results.items() if k != train_gen}
        if unseen:
            print(f"\n  {'Mean (unseen)':<35} acc={np.mean([r['acc'] for r in unseen.values()]):.4f}")

    # --- JPEG Robustness ---
    print(f"\n{'='*60}")
    print("JPEG Robustness Evaluation")
    print(f"{'='*60}")

    jpeg_results = {}
    for q in [95, 75, 50, 30]:
        ds = GenImageDataset(data_dir, train_gen, split="val",
                              transform=val_transform, max_per_class=args.max_test, jpeg_quality=q)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
        r = evaluate(model, loader, args.device)
        jpeg_results[q] = r
        print(f"  JPEG Q={q:<3}  acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}")

    # --- Save ---
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"method: {method}\ntrain_generator: {train_gen}\nbest_val_acc: {best_acc:.4f}\n\n")
        f.write("=== Cross-Generator ===\n")
        for gen, r in all_results.items():
            f.write(f"{gen}: acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}\n")
        f.write("\n=== JPEG Robustness ===\n")
        for q, r in jpeg_results.items():
            f.write(f"Q={q}: acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}\n")

    print(f"\nResults saved to {output_dir / 'results.txt'}")


if __name__ == "__main__":
    main()

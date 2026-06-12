"""Training script for PatchLTD detector."""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import numpy as np

from src.model import PatchLTDDetector, SingleViewLoRA, CLIPLinearProbe
from src.dataset import (
    GenImageDataset,
    get_transforms,
    get_paired_degradation_transforms,
    get_available_generators,
)


def parse_degradations(value):
    if value is None:
        return ("jpeg", "blur", "downsample")
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return tuple(v.strip() for v in value.split(",") if v.strip())


def symmetric_kl(logits_a, logits_b):
    log_pa = F.log_softmax(logits_a, dim=1)
    log_pb = F.log_softmax(logits_b, dim=1)
    pa = F.softmax(logits_a, dim=1)
    pb = F.softmax(logits_b, dim=1)
    return 0.5 * (
        F.kl_div(log_pa, pb, reduction="batchmean")
        + F.kl_div(log_pb, pa, reduction="batchmean")
    )


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            if isinstance(images, (list, tuple)):
                images = images[0]
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
                        choices=["patchltd", "patchltd_meanpool",
                                 "cls_ltd", "single_lora", "clip_linear"])
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=2000)
    parser.add_argument("--multi_gen", type=str, default=None,
                        help="Comma-separated generators for multi-gen training, e.g. 'sd14,biggan,adm'")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--degradation_consistency", action="store_true",
                        help="Train with paired clean/degraded views and consistency losses")
    parser.add_argument("--dcpt", dest="degradation_consistency", action="store_true",
                        help="Alias for --degradation_consistency")
    parser.add_argument("--lambda_feat", type=float, default=None,
                        help="Feature consistency loss weight")
    parser.add_argument("--lambda_pred", type=float, default=None,
                        help="Prediction consistency loss weight")
    parser.add_argument("--degradations", type=str, default=None,
                        help="Comma-separated train degradations, e.g. jpeg,blur,downsample,webp")
    parser.add_argument("--extra_degradation_eval", action="store_true",
                        help="Evaluate blur/downsample/WebP robustness after training")
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
    consistency_enabled = bool(args.degradation_consistency or train_cfg.get("degradation_consistency", False))
    lambda_feat = args.lambda_feat if args.lambda_feat is not None else train_cfg.get("lambda_feat", 0.1)
    lambda_pred = args.lambda_pred if args.lambda_pred is not None else train_cfg.get("lambda_pred", 0.5)
    degradations = parse_degradations(
        args.degradations if args.degradations is not None else train_cfg.get("degradations")
    )
    method_tag = f"{method}_dcpt" if consistency_enabled else method

    # Set seed
    seed = args.seed if args.seed is not None else train_cfg["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    output_dir = Path(config["output"]["output_dir"]) / f"{method_tag}_{train_gen}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PatchLTD | method={method_tag} | train_gen={train_gen}")
    print(f"{'='*60}")
    if consistency_enabled:
        print(
            "Degradation consistency: "
            f"lambda_feat={lambda_feat} lambda_pred={lambda_pred} "
            f"degradations={','.join(degradations)}"
        )

    # --- Dataset ---
    if consistency_enabled:
        train_transform = get_paired_degradation_transforms(image_size, degradations=degradations)
    else:
        train_transform = get_transforms(image_size, is_train=True, jpeg_aug=True)
    val_transform = get_transforms(image_size, is_train=False)

    if args.multi_gen:
        # Multi-generator training: combine multiple generators
        from torch.utils.data import ConcatDataset
        train_gens = [g.strip() for g in args.multi_gen.split(",")]
        train_datasets = []
        for gen in train_gens:
            ds = GenImageDataset(data_dir, gen, split="train",
                                  transform=train_transform, max_per_class=args.max_train)
            if len(ds) > 0:
                train_datasets.append(ds)
        train_dataset = ConcatDataset(train_datasets)
        train_gen = "+".join(train_gens)
        output_dir = Path(config["output"]["output_dir"]) / f"{method_tag}_{train_gen}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Multi-gen training: {train_gen}, total {len(train_dataset)} images")
    else:
        train_dataset = GenImageDataset(data_dir, train_gen, split="train",
                                         transform=train_transform, max_per_class=args.max_train)

    val_dataset = GenImageDataset(data_dir, train_gen.split("+")[0] if "+" in train_gen else train_gen,
                                   split="val", transform=val_transform, max_per_class=args.max_test)
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
    elif method == "patchltd_meanpool":
        model = PatchLTDDetector(
            model_cfg["clip_model"], model_cfg["clip_pretrained"],
            lora_rank=model_cfg["lora_rank"], lora_alpha=model_cfg["lora_alpha"],
            lora_target_modules=model_cfg["lora_target_modules"],
            patch_mode="meanpool",
        )
    elif method == "cls_ltd":
        model = PatchLTDDetector(
            model_cfg["clip_model"], model_cfg["clip_pretrained"],
            lora_rank=model_cfg["lora_rank"], lora_alpha=model_cfg["lora_alpha"],
            lora_target_modules=model_cfg["lora_target_modules"],
            patch_mode="cls_only",
        )
    else:  # patchltd or patchltd_dcpt
        model = PatchLTDDetector(
            model_cfg["clip_model"], model_cfg["clip_pretrained"],
            lora_rank=model_cfg["lora_rank"], lora_alpha=model_cfg["lora_alpha"],
            lora_target_modules=model_cfg["lora_target_modules"],
            patch_mode="transformer",
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
            labels = labels.to(args.device)

            if consistency_enabled:
                if not isinstance(images, (list, tuple)) or len(images) != 2:
                    raise ValueError("Expected paired clean/degraded images for consistency training")
                clean_images = images[0].to(args.device)
                degraded_images = images[1].to(args.device)

                logits, feat = model.forward_with_features(clean_images)
                degraded_logits, degraded_feat = model.forward_with_features(degraded_images)

                ce_loss = 0.5 * (criterion(logits, labels) + criterion(degraded_logits, labels))
                feat_loss = (1.0 - F.cosine_similarity(feat, degraded_feat, dim=1)).mean()
                pred_loss = symmetric_kl(logits, degraded_logits)
                loss = ce_loss + lambda_feat * feat_loss + lambda_pred * pred_loss
            else:
                images = images.to(args.device)
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            postfix = {
                "loss": f"{running_loss/(progress.n+1):.4f}",
                "acc": f"{correct/total:.4f}",
            }
            if consistency_enabled:
                postfix["feat"] = f"{feat_loss.item():.4f}"
                postfix["pred"] = f"{pred_loss.item():.4f}"
            progress.set_postfix(**postfix)

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

    # JPEG test on training generator
    jpeg_test_gen = train_gen.split("+")[0] if "+" in train_gen else train_gen
    jpeg_results = {}
    print(f"  [On {jpeg_test_gen} (train)]")
    for q in [95, 75, 50, 30]:
        ds = GenImageDataset(data_dir, jpeg_test_gen, split="val",
                              transform=val_transform, max_per_class=args.max_test, jpeg_quality=q)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
        r = evaluate(model, loader, args.device)
        jpeg_results[q] = r
        print(f"  JPEG Q={q:<3}  acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}")

    # JPEG test on unseen generators
    jpeg_unseen_results = {}
    unseen_jpeg_gens = ["glide", "vqdm", "wukong"]
    for gen in unseen_jpeg_gens:
        if gen in [g.strip() for g in train_gen.split("+")]:
            continue
        gen_results = {}
        has_data = False
        for q in [95, 50, 30]:
            ds = GenImageDataset(data_dir, gen, split="val",
                                  transform=val_transform, max_per_class=args.max_test, jpeg_quality=q)
            if len(ds) == 0:
                continue
            has_data = True
            loader = DataLoader(ds, batch_size=train_cfg["batch_size"],
                                 shuffle=False, num_workers=4, pin_memory=True)
            r = evaluate(model, loader, args.device)
            gen_results[q] = r
        if has_data:
            jpeg_unseen_results[gen] = gen_results
            print(f"  [{gen} (unseen)] Q=95:{gen_results.get(95,{}).get('acc',0):.4f} Q=50:{gen_results.get(50,{}).get('acc',0):.4f} Q=30:{gen_results.get(30,{}).get('acc',0):.4f}")

    # --- Additional Degradation Robustness ---
    extra_degradation_results = {}
    if args.extra_degradation_eval:
        print(f"\n{'='*60}")
        print("Additional Degradation Robustness Evaluation")
        print(f"{'='*60}")
        eval_specs = [
            ("blur_r1", "blur", 1.0),
            ("blur_r2", "blur", 2.0),
            ("downsample_50", "downsample", 0.5),
            ("webp_Q50", "webp", 50),
        ]

        print(f"  [On {jpeg_test_gen} (train)]")
        for name, degradation, value in eval_specs:
            ds = GenImageDataset(
                data_dir, jpeg_test_gen, split="val",
                transform=val_transform, max_per_class=args.max_test,
                degradation=degradation, degradation_value=value,
            )
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=train_cfg["batch_size"],
                                shuffle=False, num_workers=4, pin_memory=True)
            r = evaluate(model, loader, args.device)
            extra_degradation_results[name] = r
            print(f"  {name:<15} acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}")

        for gen in unseen_jpeg_gens:
            if gen in [g.strip() for g in train_gen.split("+")]:
                continue
            gen_line = []
            for name, degradation, value in eval_specs:
                ds = GenImageDataset(
                    data_dir, gen, split="val",
                    transform=val_transform, max_per_class=args.max_test,
                    degradation=degradation, degradation_value=value,
                )
                if len(ds) == 0:
                    continue
                loader = DataLoader(ds, batch_size=train_cfg["batch_size"],
                                    shuffle=False, num_workers=4, pin_memory=True)
                r = evaluate(model, loader, args.device)
                key = f"{gen}_{name}"
                extra_degradation_results[key] = r
                gen_line.append(f"{name}:{r['acc']:.4f}")
            if gen_line:
                print(f"  [{gen} (unseen)] " + " ".join(gen_line))

    # --- Save ---
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"method: {method_tag}\ntrain_generator: {train_gen}\nbest_val_acc: {best_acc:.4f}\n")
        if consistency_enabled:
            f.write(
                "degradation_consistency: true\n"
                f"lambda_feat: {lambda_feat}\n"
                f"lambda_pred: {lambda_pred}\n"
                f"degradations: {','.join(degradations)}\n"
            )
        f.write("\n")
        f.write("=== Cross-Generator ===\n")
        for gen, r in all_results.items():
            f.write(f"{gen}: acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}\n")
        f.write("\n=== JPEG Robustness (train gen) ===\n")
        for q, r in jpeg_results.items():
            f.write(f"Q={q}: acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}\n")
        if jpeg_unseen_results:
            f.write("\n=== JPEG Robustness (unseen generators) ===\n")
            for gen, gen_r in jpeg_unseen_results.items():
                for q, r in gen_r.items():
                    f.write(f"{gen}_Q={q}: acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}\n")
        if extra_degradation_results:
            f.write("\n=== Additional Degradation Robustness ===\n")
            for name, r in extra_degradation_results.items():
                f.write(f"{name}: acc={r['acc']:.4f} ap={r['ap']:.4f} auc={r['auc']:.4f}\n")

    print(f"\nResults saved to {output_dir / 'results.txt'}")


if __name__ == "__main__":
    main()

"""Benchmark parameter counts and inference speed for all methods."""
import argparse
import time
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PatchLTDDetector, SingleViewLoRA, CLIPLinearProbe


LORA_KWARGS = dict(
    lora_rank=8, lora_alpha=8,
    lora_target_modules=["out_proj", "c_fc", "c_proj"],
)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_speed(model, device, num_warmup=20, num_runs=100):
    x = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / num_runs * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = args.device

    clip_model = "ViT-B-16"
    clip_pretrained = "laion2b_s34b_b88k"

    configs = [
        ("CLIP Linear", CLIPLinearProbe, dict(clip_model=clip_model, clip_pretrained=clip_pretrained)),
        ("CLIP+LoRA", SingleViewLoRA, dict(clip_model=clip_model, clip_pretrained=clip_pretrained, **LORA_KWARGS)),
        ("CLS-LTD", PatchLTDDetector, dict(clip_model=clip_model, clip_pretrained=clip_pretrained, **LORA_KWARGS, patch_mode="cls_only")),
        ("PatchLTD-mp", PatchLTDDetector, dict(clip_model=clip_model, clip_pretrained=clip_pretrained, **LORA_KWARGS, patch_mode="meanpool")),
        ("PatchLTD", PatchLTDDetector, dict(clip_model=clip_model, clip_pretrained=clip_pretrained, **LORA_KWARGS, patch_mode="transformer")),
    ]

    print(f"\n{'Method':<16} {'Total Params':>14} {'Trainable':>12} {'Overhead':>10} {'Speed (ms)':>12}")
    print("-" * 68)

    baseline_trainable = None
    for name, cls, kwargs in configs:
        model = cls(**kwargs).to(device)
        total, trainable = count_params(model)
        if baseline_trainable is None:
            baseline_trainable = trainable
        overhead = trainable - baseline_trainable
        ms = measure_speed(model, device)
        print(f"{name:<16} {total:>14,} {trainable:>12,} {'+' + str(overhead) + '' if overhead > 0 else '--':>10} {ms:>10.1f} ms")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print()


if __name__ == "__main__":
    main()

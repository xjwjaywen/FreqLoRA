"""Print CLIP-ViT module names to find LoRA targets."""
import open_clip

model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
for name, module in model.visual.named_modules():
    if "Linear" in type(module).__name__:
        print(f"{name}: {type(module).__name__}({module.in_features}, {module.out_features})")

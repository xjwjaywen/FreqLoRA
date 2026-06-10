"""Build real/fake dataset using CUB-200 (real) + SD 2.1 (fake)."""
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image


PROMPTS = [
    "a photo of a bird",
    "a bird perched on a branch",
    "a bird in flight",
    "a close-up photo of a bird",
    "a bird in its natural habitat",
    "a colorful bird",
    "a bird standing on the ground",
    "a small bird on a tree",
    "a bird near water",
    "a photo of a wild bird",
    "a bird in a forest",
    "a bird sitting on a fence",
    "a tropical bird",
    "a bird eating seeds",
    "a beautiful bird in nature",
    "a photo of a songbird",
]


def collect_real_images(cub_dir: str, output_dir: str, num_train: int = 2000, num_val: int = 500):
    """Collect real images from CUB-200."""
    cub_path = Path(cub_dir) / "CUB_200_2011" / "images"
    if not cub_path.exists():
        print(f"CUB-200 not found at {cub_path}")
        return

    all_images = list(cub_path.rglob("*.jpg"))
    random.seed(42)
    random.shuffle(all_images)

    train_dir = Path(output_dir) / "train" / "nature"
    val_dir = Path(output_dir) / "val" / "nature"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(tqdm(all_images[:num_train], desc="Copying train real")):
        dst = train_dir / f"real_{i:05d}.png"
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        img.save(dst)

    for i, img_path in enumerate(tqdm(all_images[num_train:num_train+num_val], desc="Copying val real")):
        dst = val_dir / f"real_{i:05d}.png"
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        img.save(dst)

    print(f"Real images: {num_train} train + {num_val} val")


def generate_fake_images(
    model_path: str,
    output_dir: str,
    num_train: int = 2000,
    num_val: int = 500,
    device: str = "cuda",
):
    """Generate fake images using SD 2.1."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device)

    for split, num in [("train", num_train), ("val", num_val)]:
        out_dir = Path(output_dir) / split / "ai"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(num), desc=f"Generating {split} fake"):
            prompt = PROMPTS[i % len(PROMPTS)]
            generator.manual_seed(42 + i + (10000 if split == "val" else 0))
            image = pipe(
                prompt, num_inference_steps=25, guidance_scale=7.5,
                height=512, width=512, generator=generator,
            ).images[0]
            image.save(out_dir / f"fake_{i:05d}.png")

    del pipe
    torch.cuda.empty_cache()
    print(f"Fake images: {num_train} train + {num_val} val")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_dir", type=str, default="../CA-LoRA/data")
    parser.add_argument("--model_path", type=str, default="../CA-LoRA/models/AI-ModelScope/stable-diffusion-2-1-base")
    parser.add_argument("--output_dir", type=str, default="./data/GenImage/sd21")
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_val", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Step 1: Collecting real images from CUB-200...")
    collect_real_images(args.cub_dir, args.output_dir, args.num_train, args.num_val)

    print("\nStep 2: Generating fake images with SD 2.1...")
    generate_fake_images(args.model_path, args.output_dir, args.num_train, args.num_val, args.device)

    print("\nDataset ready!")
    for split in ["train", "val"]:
        for cls in ["nature", "ai"]:
            p = Path(args.output_dir) / split / cls
            n = len(list(p.glob("*"))) if p.exists() else 0
            print(f"  {split}/{cls}: {n} images")


if __name__ == "__main__":
    main()

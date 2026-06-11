"""Dataset loading for GenImage benchmark."""
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GenImageDataset(Dataset):
    """
    GenImage dataset loader.

    Expected structure:
      data_dir/
        {generator_name}/
          train/
            ai/     (or 1_fake/)
            nature/ (or 0_real/)
          val/
            ai/
            nature/
    """

    def __init__(
        self,
        data_dir: str,
        generator: str,
        split: str = "train",
        transform=None,
        max_per_class: int = None,
        jpeg_quality: int = None,
    ):
        self.transform = transform
        self.jpeg_quality = jpeg_quality
        self.images = []
        self.labels = []

        gen_path = Path(data_dir) / generator / split

        # Try different directory naming conventions
        real_dirs = ["nature", "0_real", "real"]
        fake_dirs = ["ai", "1_fake", "fake"]

        real_path = None
        for d in real_dirs:
            p = gen_path / d
            if p.exists():
                real_path = p
                break

        fake_path = None
        for d in fake_dirs:
            p = gen_path / d
            if p.exists():
                fake_path = p
                break

        if real_path is None or fake_path is None:
            # Try flat structure: generator/train/ with subdirectories
            if not gen_path.exists():
                print(f"WARNING: {gen_path} not found")
                return
            subdirs = sorted([d for d in gen_path.iterdir() if d.is_dir()])
            if len(subdirs) >= 2:
                real_path = subdirs[0]
                fake_path = subdirs[1]
                print(f"Using subdirs: real={real_path.name}, fake={fake_path.name}")
            else:
                print(f"WARNING: Cannot find real/fake dirs in {gen_path}")
                return

        # Load real images (label=0)
        real_images = sorted(self._get_images(real_path))
        if max_per_class:
            real_images = real_images[:max_per_class]
        self.images.extend(real_images)
        self.labels.extend([0] * len(real_images))

        # Load fake images (label=1)
        fake_images = sorted(self._get_images(fake_path))
        if max_per_class:
            fake_images = fake_images[:max_per_class]
        self.images.extend(fake_images)
        self.labels.extend([1] * len(fake_images))

        print(f"Loaded {generator}/{split}: {len(real_images)} real + {len(fake_images)} fake = {len(self.images)} total")

    def _get_images(self, path: Path):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        images = []
        for f in path.rglob("*"):
            if f.suffix.lower() in exts:
                images.append(str(f))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        for offset in range(100):
            try:
                i = (idx + offset) % len(self.images)
                img = Image.open(self.images[i]).convert("RGB")
                label = self.labels[i]
                break
            except Exception:
                continue
        else:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            label = 0
        if self.jpeg_quality is not None:
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class RandomJPEGCompression:
    """Randomly apply JPEG compression during training."""
    def __init__(self, quality_range=(30, 100), prob=0.5):
        self.quality_range = quality_range
        self.prob = prob

    def __call__(self, img):
        import random, io
        if random.random() < self.prob:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


def get_transforms(image_size: int = 224, is_train: bool = True, jpeg_aug: bool = False):
    if is_train:
        t = [transforms.Resize((image_size, image_size))]
        if jpeg_aug:
            t.append(RandomJPEGCompression(quality_range=(30, 95), prob=0.5))
        t.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(t)
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_available_generators(data_dir: str) -> list:
    """List available generators in the dataset directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted([d.name for d in data_path.iterdir() if d.is_dir()])

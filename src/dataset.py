"""Dataset loading for GenImage benchmark."""
import os
import random
from pathlib import Path
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


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
        degradation: str = None,
        degradation_value: float = None,
    ):
        self.transform = transform
        self.jpeg_quality = jpeg_quality
        self.degradation = degradation
        self.degradation_value = degradation_value
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
        if self.degradation is not None:
            img = apply_fixed_degradation(img, self.degradation, self.degradation_value)
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


def _roundtrip_compression(img, fmt: str, quality: int):
    import io
    buf = io.BytesIO()
    try:
        img.save(buf, format=fmt, quality=quality)
    except Exception:
        return img
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_fixed_degradation(img, degradation: str, value: float = None):
    """Apply a deterministic degradation for evaluation."""
    degradation = degradation.lower()
    if degradation == "jpeg":
        quality = int(value if value is not None else 50)
        return _roundtrip_compression(img, "JPEG", quality)
    if degradation == "webp":
        quality = int(value if value is not None else 50)
        return _roundtrip_compression(img, "WEBP", quality)
    if degradation == "blur":
        radius = float(value if value is not None else 1.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    if degradation == "downsample":
        scale = float(value if value is not None else 0.5)
        scale = min(max(scale, 0.1), 1.0)
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    raise ValueError(f"Unknown degradation: {degradation}")


class RandomDegradation:
    """Apply one random real-world degradation to a PIL image."""

    def __init__(
        self,
        degradations=("jpeg", "blur", "downsample"),
        jpeg_quality_range=(30, 95),
        webp_quality_range=(30, 95),
        blur_radius_range=(0.5, 2.0),
        downsample_scale_range=(0.35, 0.75),
    ):
        self.degradations = tuple(degradations)
        self.jpeg_quality_range = jpeg_quality_range
        self.webp_quality_range = webp_quality_range
        self.blur_radius_range = blur_radius_range
        self.downsample_scale_range = downsample_scale_range

    def __call__(self, img):
        if not self.degradations:
            return img
        degradation = random.choice(self.degradations)
        if degradation == "jpeg":
            q = random.randint(*self.jpeg_quality_range)
            return apply_fixed_degradation(img, "jpeg", q)
        if degradation == "webp":
            q = random.randint(*self.webp_quality_range)
            return apply_fixed_degradation(img, "webp", q)
        if degradation == "blur":
            radius = random.uniform(*self.blur_radius_range)
            return apply_fixed_degradation(img, "blur", radius)
        if degradation == "downsample":
            scale = random.uniform(*self.downsample_scale_range)
            return apply_fixed_degradation(img, "downsample", scale)
        raise ValueError(f"Unknown degradation: {degradation}")


class PairedDegradationTransform:
    """
    Return clean/degraded tensor views with shared resize and flip.

    The shared spatial transform keeps patch positions aligned for transition
    feature consistency.
    """

    def __init__(self, image_size: int = 224, degradations=("jpeg", "blur", "downsample")):
        self.image_size = image_size
        self.degradation = RandomDegradation(degradations=degradations)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _to_tensor(self, img):
        return self.normalize(TF.to_tensor(img))

    def __call__(self, img):
        img = TF.resize(img, (self.image_size, self.image_size))
        if random.random() < 0.5:
            img = TF.hflip(img)
        clean = img
        degraded = self.degradation(img)
        return self._to_tensor(clean), self._to_tensor(degraded)


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


def get_paired_degradation_transforms(
    image_size: int = 224,
    degradations=("jpeg", "blur", "downsample"),
):
    return PairedDegradationTransform(image_size=image_size, degradations=degradations)


def get_available_generators(data_dir: str) -> list:
    """List available generators in the dataset directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted([d.name for d in data_path.iterdir() if d.is_dir()])

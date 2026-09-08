"""Image transforms for Pox Scanner (conservative, ImageNet-normalized).

Train-only augmentation is mild: no vertical flip, elastic deform, or heavy jitter.
"""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_EVAL_SPLITS = frozenset({"val", "test", "inference"})


def build_transforms(split: str, image_size: int = 224) -> T.Compose:
    """Return a torchvision transform pipeline for ``split``.

    Parameters
    ----------
    split:
        ``train``, ``val``, ``test``, or ``inference``.
    image_size:
        Square output size (default 224).
    """
    if image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {image_size}")

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    key = str(split).strip().lower()

    if key == "train":
        return T.Compose(
            [
                T.Resize(image_size),
                T.RandomHorizontalFlip(0.5),
                T.RandomRotation(10),
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                normalize,
            ]
        )

    if key in _EVAL_SPLITS:
        return T.Compose(
            [
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                normalize,
            ]
        )

    raise ValueError(
        f"Unknown split {split!r}. Expected one of: train, val, test, inference."
    )

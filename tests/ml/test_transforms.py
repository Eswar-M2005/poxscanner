"""Transforms produce ImageNet-normalized CHW tensors of the requested size."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.src.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402


def _rgb(size: tuple[int, int] = (48, 64)) -> Image.Image:
    return Image.new("RGB", size, (40, 80, 120))


def _type_names(compose) -> list[str]:
    return [type(step).__name__ for step in compose.transforms]


def test_val_transform_shape_224() -> None:
    tensor = build_transforms("val", image_size=224)(_rgb())
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_test_and_inference_match_val() -> None:
    names_val = _type_names(build_transforms("val"))
    names_test = _type_names(build_transforms("test"))
    names_inf = _type_names(build_transforms("inference"))
    assert names_val == names_test == names_inf
    assert names_val == ["Resize", "CenterCrop", "ToTensor", "Normalize"]


def test_train_transform_shape_and_ops() -> None:
    pipeline = build_transforms("train", image_size=224)
    names = _type_names(pipeline)
    assert names == [
        "Resize",
        "RandomHorizontalFlip",
        "RandomRotation",
        "RandomResizedCrop",
        "ColorJitter",
        "ToTensor",
        "Normalize",
    ]
    assert "RandomVerticalFlip" not in names
    tensor = pipeline(_rgb())
    assert tuple(tensor.shape) == (3, 224, 224)


def test_custom_image_size() -> None:
    image_size = 32
    tensor = build_transforms("val", image_size=image_size)(_rgb((16, 20)))
    assert tuple(tensor.shape) == (3, image_size, image_size)
    train = build_transforms("train", image_size=image_size)(_rgb((16, 20)))
    assert tuple(train.shape) == (3, image_size, image_size)


def test_no_forbidden_augmentation() -> None:
    blob = str(build_transforms("train")).lower()
    assert "vertical" not in blob
    assert "elastic" not in blob
    jitter = next(
        step
        for step in build_transforms("train").transforms
        if type(step).__name__ == "ColorJitter"
    )
    assert jitter.brightness not in (None, 0, 0.0)
    assert jitter.contrast not in (None, 0, 0.0)
    # saturation/hue stay at identity (None / 0 means unused)
    assert jitter.saturation in (None, 0, 0.0, (1.0, 1.0))
    assert jitter.hue in (None, 0, 0.0, (0.0, 0.0))


def test_normalize_uses_imagenet_stats() -> None:
    normalize = build_transforms("val").transforms[-1]
    assert tuple(round(x, 3) for x in normalize.mean) == IMAGENET_MEAN
    assert tuple(round(x, 3) for x in normalize.std) == IMAGENET_STD


def test_unknown_split_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown split"):
        build_transforms("holdout")

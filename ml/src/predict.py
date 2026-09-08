"""Checkpoint loading and single-image inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml.src.dataset import DISPLAY, INDEX_TO_LABEL, LABEL_TO_INDEX
from ml.src.models import build_model
from ml.src.utils import get_logger, repo_root


log = get_logger("ml.src.predict")


def get_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device) if not isinstance(device, torch.device) else device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _image_size_from_config(config: dict[str, Any]) -> int:
    if "image_size" in config and config["image_size"] is not None:
        return int(config["image_size"])
    train = config.get("train") or {}
    if isinstance(train, dict) and train.get("image_size"):
        return int(train["image_size"])
    return 224


def _build_inference_transform(image_size: int):
    try:
        from ml.src.transforms import build_transforms

        for split in ("inference", "val", "test"):
            try:
                return build_transforms(split, image_size=image_size)
            except Exception:
                continue
    except ImportError:
        pass
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_checkpoint(path: Path, device=None) -> tuple[nn.Module, dict]:
    """Load a training checkpoint payload and reconstruct the model."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    device = get_device(device)
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {ckpt_path}")

    cfg = payload.get("config") or {}
    model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    name = payload.get("model_name") or model_cfg.get("name")
    if not name:
        raise ValueError("Checkpoint is missing model_name")
    model = build_model(
        str(name),
        num_classes=int(model_cfg.get("num_classes", 2)),
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def predict_image(model: nn.Module, image, config: dict[str, Any] | None = None) -> dict:
    """Run binary inference on a PIL image.

    Returns display-label prediction, confidence of the predicted class, and
    probabilities keyed ``monkeypox`` / ``non_monkeypox``.
    """
    from PIL import Image

    config = config or {}
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    image = image.convert("RGB")
    image_size = _image_size_from_config(config)
    transform = _build_inference_transform(image_size)
    tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    p_non = float(probs[LABEL_TO_INDEX["non_monkeypox"]].item())
    p_mpx = float(probs[LABEL_TO_INDEX["monkeypox"]].item())
    pred_idx = int(probs.argmax().item())
    label = INDEX_TO_LABEL[pred_idx]
    return {
        "prediction": DISPLAY[label],
        "confidence": float(probs[pred_idx].item()),
        "probabilities": {
            "monkeypox": p_mpx,
            "non_monkeypox": p_non,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pox Scanner image prediction")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--checkpoint", required=True, help="Path to a *.pt checkpoint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from PIL import Image

    image_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root() / ckpt_path
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1
    try:
        image = Image.open(image_path)
        image.load()
    except Exception as exc:
        print(f"Could not decode image: {exc}", file=sys.stderr)
        return 1

    device = get_device()
    model, payload = load_checkpoint(ckpt_path, device=device)
    config = payload.get("config") or {}
    if isinstance(config, dict) and "image_size" not in config:
        config = {**config, "image_size": payload.get("image_size", 224)}
    result = predict_image(model, image, config)
    result["model"] = payload.get("model_name")
    result["version"] = payload.get("version", "0.1.0")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

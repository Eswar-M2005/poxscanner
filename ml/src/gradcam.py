"""Grad-CAM heatmaps for convolutional classifiers (Selvaraju et al.)."""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PNG_FORMAT = "PNG"


def _minmax_norm(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    cmin = float(cam.min())
    cmax = float(cam.max())
    denom = cmax - cmin
    if denom < 1e-8:
        return np.zeros(cam.shape, dtype=np.float32)
    return ((cam - cmin) / denom).astype(np.float32, copy=False)


def _jet_lut() -> np.ndarray:
    """256-entry jet-like RGB lookup table in [0, 1]."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


_JET_LUT = _jet_lut()


def _apply_jet(cam: np.ndarray) -> np.ndarray:
    """Map a 2D [0, 1] heatmap to an HxWx3 float RGB jet image."""
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    idx = np.clip((cam * 255.0).round().astype(np.int32), 0, 255)
    return _JET_LUT[idx]


def _as_hw_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim == 3 and cam.shape[0] == 1:
        cam = cam[0]
    if cam.ndim != 2:
        raise ValueError("cam must be a 2D heatmap of shape (H, W)")
    return cam


def _image_to_float_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if np.issubdtype(image.dtype, np.floating):
        max_val = float(np.nanmax(image)) if image.size else 0.0
        img_f = image.astype(np.float32, copy=False)
        if max_val > 1.0:
            img_f = img_f / 255.0
        return np.clip(img_f, 0.0, 1.0)
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def _resize_cam(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    if cam.shape == (height, width):
        return np.clip(cam, 0.0, 1.0)
    cam_u8 = np.clip(cam * 255.0, 0.0, 255.0).astype(np.uint8)
    resized = Image.fromarray(cam_u8).resize(
        (width, height), resample=Image.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def overlay_cam_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto an RGB image. Returns uint8 HxWx3."""
    img_f = _image_to_float_rgb(image_rgb)
    cam_hw = _minmax_norm(_as_hw_cam(cam))
    cam_hw = _resize_cam(cam_hw, img_f.shape[0], img_f.shape[1])
    heat = _apply_jet(cam_hw)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = (1.0 - alpha) * img_f + alpha * heat
    return np.clip(blended * 255.0, 0.0, 255.0).astype(np.uint8)


def cam_to_png_bytes(overlay_rgb: np.ndarray) -> bytes:
    """Encode an RGB overlay array as PNG bytes."""
    arr = np.asarray(overlay_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("overlay_rgb must have shape (H, W, 3)")
    if arr.dtype != np.uint8:
        arr = _image_to_float_rgb(arr)
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=_PNG_FORMAT)
    return buf.getvalue()


def heatmap_to_png_bytes(cam: np.ndarray) -> bytes:
    """Encode a 2D [0, 1] CAM as a jet-colored PNG."""
    heat = (_apply_jet(_minmax_norm(_as_hw_cam(cam))) * 255.0).astype(np.uint8)
    return cam_to_png_bytes(heat)


class GradCAM:
    """Class-discriminative localization map from a convolutional target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._on_forward)
        if hasattr(target_layer, "register_full_backward_hook"):
            self._backward_handle = target_layer.register_full_backward_hook(
                self._on_backward
            )
        else:
            self._backward_handle = target_layer.register_backward_hook(self._on_backward)

    def _on_forward(self, module, inputs, output) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Grad-CAM target_layer must return a tensor")
        # Clone so later in-place activations do not corrupt the saved map.
        self.activations = output.detach().clone()

    def _on_backward(self, module, grad_input, grad_output) -> None:
        if not grad_output:
            return
        grad = grad_output[0]
        if grad is None:
            return
        self.gradients = grad.detach().clone()

    def generate(
        self, input_tensor: torch.Tensor, class_idx: int | None = None
    ) -> np.ndarray:
        """
        Compute a spatial Grad-CAM heatmap in [0, 1].

        Parameters
        ----------
        input_tensor:
            Batch of size 1, NCHW.
        class_idx:
            Target class. If None, uses argmax of logits.

        Returns
        -------
        np.ndarray
            Float32 array of shape (H, W) matching the input spatial size.
        """
        if not torch.is_tensor(input_tensor):
            raise TypeError("input_tensor must be a torch.Tensor")
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("input_tensor must have shape (1, C, H, W)")

        was_training = self.model.training
        self.model.eval()
        self.activations = None
        self.gradients = None

        try:
            x = input_tensor.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(x)
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise ValueError("model must return logits of shape (1, num_classes)")
                if class_idx is None:
                    class_idx = int(torch.argmax(logits, dim=1).item())
                else:
                    class_idx = int(class_idx)
                n_classes = int(logits.shape[1])
                if class_idx < 0 or class_idx >= n_classes:
                    raise IndexError(
                        f"class_idx {class_idx} is out of range for {n_classes} classes"
                    )

                score = logits[0, class_idx]
                self.model.zero_grad(set_to_none=True)
                if x.grad is not None:
                    x.grad = None
                score.backward()

            acts = self.activations
            grads = self.gradients
            if acts is None or grads is None:
                raise RuntimeError(
                    "Grad-CAM hooks did not capture activations or gradients"
                )
            if acts.ndim != 4 or grads.ndim != 4:
                raise RuntimeError(
                    "target_layer must produce 4D feature maps (N, C, H, W)"
                )

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
            _, _, height, width = input_tensor.shape
            cam = F.interpolate(
                cam, size=(height, width), mode="bilinear", align_corners=False
            )
            cam_np = cam.squeeze(0).squeeze(0).detach().cpu().numpy()
            return _minmax_norm(cam_np)
        finally:
            self.model.train(was_training)

    def close(self) -> None:
        """Remove registered forward and backward hooks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None
        self.activations = None
        self.gradients = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

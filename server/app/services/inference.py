"""Inference service: real checkpoint when available, otherwise deterministic demo.

Torch / ml.src are imported only when a checkpoint is actually loaded so tests
that stay in demo mode pass without PyTorch.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.errors import APIError
from app.schemas.predict import DISCLAIMER
from app.utils.images import ALLOWED_MIMES
from app.services import model_registry

logger = logging.getLogger("poxscanner.inference")

DEFAULT_MODEL_NAME = "efficientnet_b0"
TASK_NAME = "binary_skin_lesion_classification"
CLASSES = ["non_monkeypox", "monkeypox"]
DISPLAY = {"non_monkeypox": "Non-Monkeypox", "monkeypox": "Monkeypox"}
DEFAULT_IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.5
DEMO_EXPLANATION_NOTE = (
    "Placeholder heatmap (demo mode). Not a trained Grad-CAM explanation."
)


def _server_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _select_runtime_device():
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _softmax_two(logit_a: float, logit_b: float) -> tuple[float, float]:
    logits = np.array([logit_a, logit_b], dtype=np.float64)
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    probs = exps / exps.sum()
    return float(probs[0]), float(probs[1])


def demo_probabilities(image_bytes: bytes) -> tuple[float, float]:
    """Deterministic (monkeypox, non_monkeypox) from SHA-256 logits."""
    digest = hashlib.sha256(image_bytes).digest()
    logit_mp = (int.from_bytes(digest[0:8], "big") / 2**64) * 8.0 - 4.0
    logit_nmp = (int.from_bytes(digest[8:16], "big") / 2**64) * 8.0 - 4.0
    return _softmax_two(logit_mp, logit_nmp)


def _radial_cam(height: int, width: int, digest: bytes) -> np.ndarray:
    cx = width * (0.30 + 0.40 * (digest[16] / 255.0))
    cy = height * (0.30 + 0.40 * (digest[17] / 255.0))
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = float(np.sqrt(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2) or 1.0)
    cam = 1.0 - (dist / max_dist)
    return np.clip(cam, 0.0, 1.0).astype(np.float64)


def overlay_cam_on_image(image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Simple jet-like heatmap overlay (used when ml.gradcam is unavailable)."""
    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")
    height, width = rgb.shape[:2]
    heat = np.asarray(cam, dtype=np.float64)
    if heat.shape[:2] != (height, width):
        cam_img = Image.fromarray(
            (np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
        )
        cam_img = cam_img.resize((width, height), Image.BILINEAR)
        heat = np.asarray(cam_img).astype(np.float64) / 255.0
    heat = np.clip(heat, 0.0, 1.0)
    colormap = np.zeros((height, width, 3), dtype=np.float64)
    colormap[..., 0] = np.clip(1.5 * heat, 0.0, 1.0)
    colormap[..., 1] = np.clip(1.5 * (1.0 - np.abs(heat - 0.5) * 2.0), 0.0, 1.0)
    colormap[..., 2] = np.clip(1.5 * (1.0 - heat), 0.0, 1.0)
    alpha = 0.45 * heat[..., None]
    blended = (1.0 - alpha) * rgb.astype(np.float64) + alpha * (colormap * 255.0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def overlay_to_png_b64(overlay_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def placeholder_overlay_b64(image: Image.Image, image_bytes: bytes) -> str:
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    digest = hashlib.sha256(image_bytes).digest()
    cam = _radial_cam(h, w, digest)
    overlay = overlay_cam_on_image(rgb, cam)
    return overlay_to_png_b64(overlay)


def maybe_store_upload(data: bytes, pil_format: str | None) -> None:
    if not settings.STORE_UPLOADS:
        return
    dest_dir = _server_root() / "var" / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(
        (pil_format or "").upper(), ".bin"
    )
    path = dest_dir / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    logger.info("Stored upload as %s", path.name)


class InferenceService:
    def __init__(self) -> None:
        self._load_attempted = False
        self._model: Any = None
        self._meta: dict[str, Any] = {}
        self._model_loaded = False
        self._checkpoint_path: Path | None = None
        self._device = None

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def using_demo(self) -> bool:
        mode = settings.demo_mode_normalized
        if mode == "true":
            return True
        if mode == "false":
            return False
        return not self._model_loaded

    def try_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if settings.demo_mode_normalized == "true":
            logger.info("DEMO_MODE=true; skipping checkpoint load")
            return
        path = model_registry.checkpoint_path()
        if path is None:
            logger.info("No checkpoint found")
            return
        try:
            self._load_checkpoint(path)
        except Exception:
            logger.exception("Failed to load checkpoint from %s", path)

    def _load_checkpoint(self, path: Path) -> None:
        _ensure_repo_on_path()
        from ml.src.predict import load_checkpoint  # noqa: PLC0415

        if self._device is None:
            self._device = _select_runtime_device()
        model, meta = load_checkpoint(path, device=self._device)
        self._model = model
        self._meta = meta if isinstance(meta, dict) else {}
        self._model_loaded = True
        self._checkpoint_path = path
        logger.info("Loaded checkpoint %s (%s)", path, self._meta.get("model_name"))

    def require_predict_ready(self) -> None:
        self.try_load()
        if settings.demo_mode_normalized == "false" and not self._model_loaded:
            raise APIError(
                503,
                "model_unavailable",
                "No model checkpoint is available and demo mode is disabled.",
            )

    def predict(
        self,
        image_bytes: bytes | None,
        image: Image.Image | None,
        *,
        explain: bool = True,
        clinical_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        has_image = image is not None
        has_clinical = clinical_data is not None and len(clinical_data) > 0

        if not has_image and not has_clinical:
            raise APIError(400, "missing_input", "Please provide a lesion image, clinical symptoms, or both.")

        if has_image:
            self.require_predict_ready()

        if self.using_demo or (has_image and not self._model_loaded):
            return self._predict_demo(image_bytes, image, explain=explain, clinical_data=clinical_data)
        return self._predict_real(image_bytes, image, explain=explain, clinical_data=clinical_data)

    def _predict_demo(
        self,
        image_bytes: bytes | None,
        image: Image.Image | None,
        *,
        explain: bool,
        clinical_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        has_image = image is not None
        has_clinical = clinical_data is not None and len(clinical_data) > 0

        if has_clinical and not has_image:
            modality = "clinical_only"
            model_name = "Model B: Clinical-Only MLP"
            is_multi = False
            n_sym = len(clinical_data)
            p_mp = min(0.95, max(0.55, 0.5 + n_sym * 0.08))
            p_nmp = 1.0 - p_mp
        elif has_image and has_clinical:
            modality = "multimodal"
            model_name = "Model C: Multimodal Fusion Network"
            is_multi = True
            p_mp, p_nmp = demo_probabilities(image_bytes or b"multimodal_demo")
        else:
            modality = "image_only"
            model_name = "Model A: ResNet-50 (Image-Only)"
            is_multi = False
            p_mp, p_nmp = demo_probabilities(image_bytes or b"image_demo")

        if p_mp >= p_nmp:
            label_key = "monkeypox"
            confidence = p_mp
        else:
            label_key = "non_monkeypox"
            confidence = p_nmp

        demo_clin_expl = None
        if has_clinical:
            top_supp = []
            for k, v in list(clinical_data.items())[:5]:
                top_supp.append({"feature": str(k), "importance": 0.05, "direction": "supports"})
            demo_clin_expl = {
                "attributions": top_supp,
                "top_supporting_symptoms": top_supp,
                "top_opposing_symptoms": [],
            }

        payload: dict[str, Any] = {
            "prediction": DISPLAY[label_key],
            "confidence": confidence,
            "probabilities": {
                "monkeypox": p_mp,
                "non_monkeypox": p_nmp,
            },
            "model": model_name,
            "version": settings.APP_VERSION,
            "demo_mode": True,
            "disclaimer": DISCLAIMER,
            "explanation": None,
            "clinical_explanation": demo_clin_expl,
            "multimodal": is_multi,
            "modality": modality,
        }
        if explain and has_image and image is not None:
            payload["explanation"] = {
                "method": "gradcam",
                "overlay_png_base64": placeholder_overlay_b64(image, image_bytes or b""),
                "note": DEMO_EXPLANATION_NOTE,
            }
        return payload

    def _predict_real(
        self,
        image_bytes: bytes | None,
        image: Image.Image | None,
        *,
        explain: bool,
        clinical_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ensure_repo_on_path()
        import torch

        has_image = image is not None
        has_clinical = clinical_data is not None and len(clinical_data) > 0

        device = _select_runtime_device()
        clinical_json = _repo_root() / "ml" / "artifacts" / "metrics" / "clinical_features.json"

        # ------------------------------------------------------------
        # 1. USER INPUTS ONLY CLINICAL SYMPTOMS -> USE MODEL B
        # ------------------------------------------------------------
        if has_clinical and not has_image:
            from ml.src.clinical import ClinicalPreprocessor
            from ml.src.architectures.multimodal import build_clinical_model
            from ml.src.explainability import explain_clinical

            preprocessor = ClinicalPreprocessor.load(clinical_json)
            clin_vec = preprocessor.transform(clinical_data)
            clin_tensor = torch.from_numpy(clin_vec).float().to(device)

            model_b_ckpt = _repo_root() / "ml" / "artifacts" / "checkpoints" / "clinical_only_best.pt"
            model_b = build_clinical_model(clinical_dim=len(preprocessor.selected_features_))
            if model_b_ckpt.is_file():
                ckpt = torch.load(model_b_ckpt, map_location=device)
                model_b.load_state_dict(ckpt.get("state_dict", ckpt))
            model_b.to(device)
            model_b.eval()

            with torch.no_grad():
                logits = model_b(clin_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                p_nmp = float(probs[0].item())
                p_mp = float(probs[1].item())

            class_idx = 1 if p_mp >= 0.5 else 0
            label_key = "monkeypox" if class_idx == 1 else "non_monkeypox"
            confidence = p_mp if class_idx == 1 else p_nmp

            clinical_explanation_data = explain_clinical(
                model_b,
                clin_tensor,
                preprocessor.selected_features_,
                target_class=class_idx,
            )

            return {
                "prediction": DISPLAY[label_key],
                "confidence": confidence,
                "probabilities": {
                    "monkeypox": p_mp,
                    "non_monkeypox": p_nmp,
                },
                "model": "Model B: Clinical-Only MLP",
                "version": settings.APP_VERSION,
                "demo_mode": False,
                "disclaimer": DISCLAIMER,
                "explanation": None,
                "clinical_explanation": clinical_explanation_data,
                "multimodal": False,
                "modality": "clinical_only",
            }

        # ------------------------------------------------------------
        # 2. USER INPUTS BOTH IMAGE & CLINICAL -> USE MODEL C (MULTIMODAL)
        # ------------------------------------------------------------
        if has_image and has_clinical:
            multi_ckpt_path = _repo_root() / "ml" / "artifacts" / "checkpoints" / "multimodal_best.pt"
            if multi_ckpt_path.is_file() and clinical_json.is_file():
                from ml.src.transforms import build_transforms
                from ml.src.clinical import ClinicalPreprocessor
                from ml.src.architectures.multimodal import build_multimodal_model
                from ml.src.explainability import explain_clinical

                preprocessor = ClinicalPreprocessor.load(clinical_json)
                clin_vec = preprocessor.transform(clinical_data)
                clin_tensor = torch.from_numpy(clin_vec).float().to(device)

                ckpt = torch.load(multi_ckpt_path, map_location=device)
                multi_model = build_multimodal_model(
                    backbone_name=ckpt.get("backbone_name", "resnet50"),
                    clinical_dim=ckpt.get("clinical_dim", 28),
                    num_classes=2,
                    pretrained=False,
                )
                multi_model.load_state_dict(ckpt["state_dict"])
                multi_model.to(device)
                multi_model.eval()

                tfm = build_transforms("val", 224)
                img_tensor = tfm(image)
                if getattr(img_tensor, "dim", lambda: 3)() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                img_tensor = img_tensor.to(device)

                with torch.no_grad():
                    logits = multi_model(img_tensor, clin_tensor)
                    probs_tensor = torch.softmax(logits, dim=1)[0]
                    p_nmp = float(probs_tensor[0].item())
                    p_mp = float(probs_tensor[1].item())

                threshold = self._deployment_threshold()
                class_idx = 1 if p_mp >= threshold else 0
                label_key = "monkeypox" if class_idx == 1 else "non_monkeypox"
                confidence = p_mp if class_idx == 1 else p_nmp

                clinical_explanation_data = explain_clinical(
                    multi_model,
                    clin_tensor,
                    preprocessor.selected_features_,
                    image_tensor=img_tensor,
                    target_class=class_idx,
                )

                explanation_payload = None
                if explain and image is not None:
                    explanation_payload = self._real_explanation(
                        image, image_bytes or b"", class_idx=class_idx, custom_model=multi_model
                    )

                return {
                    "prediction": DISPLAY[label_key],
                    "confidence": confidence,
                    "probabilities": {
                        "monkeypox": p_mp,
                        "non_monkeypox": p_nmp,
                    },
                    "model": "Model C: Multimodal Fusion Network",
                    "version": str(self._meta.get("version") or model_registry.version()),
                    "demo_mode": False,
                    "disclaimer": DISCLAIMER,
                    "explanation": explanation_payload,
                    "clinical_explanation": clinical_explanation_data,
                    "multimodal": True,
                    "modality": "multimodal",
                }

        # ------------------------------------------------------------
        # 3. USER INPUTS ONLY IMAGE -> USE MODEL A (IMAGE-ONLY)
        # ------------------------------------------------------------
        from ml.src.predict import predict_image

        raw = predict_image(self._model, image, self._meta)
        probs = raw.get("probabilities", raw) if isinstance(raw, dict) else {}
        try:
            p_mp = float(probs["monkeypox"])
            p_nmp = float(probs["non_monkeypox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise APIError(
                503,
                "model_unavailable",
                "Model returned an unexpected probability payload.",
            ) from exc

        threshold = self._deployment_threshold()
        if p_mp >= threshold:
            label_key = "monkeypox"
            confidence = p_mp
            class_idx = 1
        else:
            label_key = "non_monkeypox"
            confidence = p_nmp
            class_idx = 0

        payload: dict[str, Any] = {
            "prediction": DISPLAY[label_key],
            "confidence": confidence,
            "probabilities": {
                "monkeypox": p_mp,
                "non_monkeypox": p_nmp,
            },
            "model": "Model A: ResNet-50 (Image-Only)",
            "version": str(self._meta.get("version") or model_registry.version()),
            "demo_mode": False,
            "disclaimer": DISCLAIMER,
            "explanation": None,
            "clinical_explanation": None,
            "multimodal": False,
            "modality": "image_only",
        }
        if explain and image is not None:
            payload["explanation"] = self._real_explanation(
                image, image_bytes or b"", class_idx=class_idx
            )
        return payload

    def _real_explanation(
        self,
        image: Image.Image,
        image_bytes: bytes,
        *,
        class_idx: int,
        custom_model: Any | None = None,
    ) -> dict[str, Any]:
        try:
            _ensure_repo_on_path()
            from ml.src.gradcam import (  # noqa: PLC0415
                GradCAM,
                cam_to_png_bytes,
                overlay_cam_on_image as ml_overlay,
            )
            from ml.src.models import get_gradcam_target_layer  # noqa: PLC0415
            from ml.src.transforms import build_transforms  # noqa: PLC0415

            model_obj = custom_model if custom_model is not None else self._model
            if hasattr(model_obj, "get_gradcam_target_layer"):
                target = model_obj.get_gradcam_target_layer()
            else:
                name = str(self._meta.get("model_name") or DEFAULT_MODEL_NAME)
                target = get_gradcam_target_layer(model_obj, name)

            engine = GradCAM(model_obj, target)
            image_size = int(self._meta.get("image_size") or DEFAULT_IMAGE_SIZE)
            try:
                tfm = build_transforms("val", image_size)
                tensor = tfm(image)
                if getattr(tensor, "dim", lambda: 3)() == 3:
                    tensor = tensor.unsqueeze(0)
                heatmap = engine.generate(tensor, class_idx=class_idx)
            finally:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
            rgb = np.array(image.convert("RGB"))
            overlay = ml_overlay(rgb, heatmap)
            png = cam_to_png_bytes(overlay)
            return {
                "method": "gradcam",
                "overlay_png_base64": base64.b64encode(png).decode("ascii"),
                "note": None,
            }
        except Exception:
            logger.exception("Grad-CAM failed; using placeholder overlay")
            return {
                "method": "gradcam",
                "overlay_png_base64": placeholder_overlay_b64(image, image_bytes),
                "note": "Grad-CAM unavailable; placeholder overlay shown.",
            }
        except Exception:
            logger.exception("Grad-CAM failed; using placeholder overlay")
            return {
                "method": "gradcam",
                "overlay_png_base64": placeholder_overlay_b64(image, image_bytes),
                "note": "Grad-CAM unavailable; placeholder overlay shown.",
            }

    def _display_model_name(self) -> str:
        if self._model_loaded and self._meta.get("model_name"):
            return str(self._meta["model_name"])
        return model_registry.model_name() or DEFAULT_MODEL_NAME

    def _metrics_for_loaded_model(self) -> dict[str, Any] | None:
        if not self._model_loaded:
            return None
        name = str(self._meta.get("model_name") or self._display_model_name())
        metrics_dir = _repo_root() / "ml" / "artifacts" / "metrics"
        for split in ("test", "val"):
            path = metrics_dir / f"{name}_{split}.json"
            data = model_registry.read_json(path)
            if data is not None:
                return data
        return None

    def _deployment_threshold(self) -> float:
        threshold = model_registry.threshold()
        return threshold if 0.0 <= threshold <= 1.0 else DEFAULT_THRESHOLD

    def model_info(self) -> dict[str, Any]:
        self.try_load()
        demo = self.using_demo
        payload = model_registry.model_info_payload(demo_mode=demo)
        if self._model_loaded:
            payload["name"] = str(self._meta.get("model_name") or payload["name"])
            payload["version"] = str(self._meta.get("version") or payload["version"])
            try:
                payload["input"]["image_size"] = int(
                    self._meta.get("image_size") or payload["input"]["image_size"]
                )
            except (TypeError, ValueError):
                pass
            metrics = self._metrics_for_loaded_model()
            if metrics is not None:
                payload["metrics"] = metrics
            if isinstance(self._meta.get("class_to_idx"), dict):
                class_mapping = {
                    str(key): int(value)
                    for key, value in self._meta["class_to_idx"].items()
                    if isinstance(key, str)
                }
                if class_mapping:
                    payload["class_mapping"] = class_mapping
        return payload


_SERVICE: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = InferenceService()
    return _SERVICE

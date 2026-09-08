"""API v1 routes: /predict and /model-info."""

from __future__ import annotations

import json
from fastapi import APIRouter, File, Form, Query, UploadFile

from app.core.config import settings
from app.core.errors import APIError
from app.schemas.predict import ModelInfoResponse, PredictResponse
from app.services.inference import get_inference_service, maybe_store_upload
from app.utils.images import normalize_content_type, validate_and_decode

_MIME_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

router = APIRouter()


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    payload = get_inference_service().model_info()
    return ModelInfoResponse.model_validate(payload)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile | None = File(None, description="Lesion image"),
    explain: bool = Query(True, description="Include Grad-CAM overlay"),
    clinical_data: str | None = Form(None, description="JSON-encoded dictionary of clinical symptoms"),
) -> PredictResponse:
    parsed_clinical = None
    if clinical_data:
        try:
            parsed_clinical = json.loads(clinical_data)
            if not isinstance(parsed_clinical, dict) or len(parsed_clinical) == 0:
                parsed_clinical = None
        except Exception:
            parsed_clinical = None

    has_image = image is not None and bool(image.filename)
    if not has_image and not parsed_clinical:
        raise APIError(
            400,
            "missing_input",
            "Please provide a lesion image, clinical symptoms, or both.",
        )

    data = None
    pil = None
    if has_image and image is not None:
        if image.size is not None and image.size > settings.MAX_UPLOAD_BYTES:
            raise APIError(
                413,
                "file_too_large",
                f"File exceeds maximum size of {settings.MAX_UPLOAD_BYTES} bytes.",
            )
        data = await image.read()
        pil = validate_and_decode(data, image.content_type, settings.MAX_UPLOAD_BYTES)
        maybe_store_upload(
            data,
            _MIME_FORMAT.get(normalize_content_type(image.content_type), "PNG"),
        )

    payload = get_inference_service().predict(data, pil, explain=explain, clinical_data=parsed_clinical)
    return PredictResponse.model_validate(payload)

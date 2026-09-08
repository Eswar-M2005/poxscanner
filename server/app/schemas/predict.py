"""Pydantic models matching the HTTP JSON contracts exactly."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Research and educational use only. This tool does not provide a medical diagnosis."
)

PredictionLabel = Literal["Monkeypox", "Non-Monkeypox"]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    demo_mode: bool
    version: str


class ModelInputSpec(BaseModel):
    image_size: int
    formats: list[str]
    normalization: dict[str, list[float]] | None = None


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    task: str
    classes: list[str]
    class_mapping: dict[str, int] | None = None
    checkpoint: str | None = None
    input: ModelInputSpec
    threshold: float | None = None
    metrics: dict[str, Any] | None = None
    selection_status: str
    demo_mode: bool
    disclaimer: str


class Probabilities(BaseModel):
    monkeypox: float
    non_monkeypox: float


class Explanation(BaseModel):
    method: str
    overlay_png_base64: str
    note: str | None = None


class FeatureAttribution(BaseModel):
    feature: str
    importance: float
    direction: Literal["supports", "opposes"]


class ClinicalExplanation(BaseModel):
    attributions: list[FeatureAttribution] = Field(default_factory=list)
    top_supporting_symptoms: list[FeatureAttribution] = Field(default_factory=list)
    top_opposing_symptoms: list[FeatureAttribution] = Field(default_factory=list)


class PredictResponse(BaseModel):
    prediction: PredictionLabel
    confidence: float
    probabilities: Probabilities
    model: str
    version: str
    demo_mode: bool
    disclaimer: str = Field(default=DISCLAIMER)
    explanation: Explanation | None = None
    clinical_explanation: ClinicalExplanation | None = None
    multimodal: bool = False
    modality: Literal["image_only", "clinical_only", "multimodal"] = "image_only"

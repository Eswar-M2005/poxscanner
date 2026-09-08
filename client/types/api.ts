/** Shared API types — must match docs/contracts.md */

export type ClassKey = "monkeypox" | "non_monkeypox";

export interface HealthResponse {
  status: "ok";
  service: "poxscanner-api";
  model_loaded: boolean;
  demo_mode: boolean;
  version: string;
}

export interface ModelInputSpec {
  image_size: number;
  formats: string[];
  normalization?: {
    mean: number[];
    std: number[];
  } | null;
}

export interface ModelInfoResponse {
  name: string;
  version: string;
  task: string;
  classes: string[];
  class_mapping?: Record<string, number> | null;
  checkpoint?: string | null;
  input: ModelInputSpec;
  threshold?: number | null;
  metrics: Record<string, unknown> | null;
  selection_status: string;
  demo_mode: boolean;
  disclaimer: string;
}

export interface PredictExplanation {
  method: string;
  overlay_png_base64: string;
  note: string | null;
}

export interface FeatureAttribution {
  feature: string;
  importance: number;
  direction: "supports" | "opposes";
}

export interface ClinicalExplanation {
  attributions: FeatureAttribution[];
  top_supporting_symptoms: FeatureAttribution[];
  top_opposing_symptoms: FeatureAttribution[];
}

export interface PredictResponse {
  prediction: "Monkeypox" | "Non-Monkeypox";
  confidence: number;
  probabilities: {
    monkeypox: number;
    non_monkeypox: number;
  };
  model: string;
  version: string;
  demo_mode: boolean;
  disclaimer: string;
  explanation: PredictExplanation | null;
  clinical_explanation?: ClinicalExplanation | null;
  multimodal?: boolean;
  modality?: "image_only" | "clinical_only" | "multimodal";
}

export type PredictionResponse = PredictResponse;

export interface ApiErrorDetail {
  code: string;
  message: string;
}

export interface ApiErrorBody {
  error: ApiErrorDetail;
}

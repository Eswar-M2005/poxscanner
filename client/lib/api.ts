import type {
  ApiErrorBody,
  HealthResponse,
  ModelInfoResponse,
  PredictResponse,
} from "@/types/api";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000"
).replace(/\/$/, "");

export class ApiRequestError extends Error {
  readonly status?: number;
  readonly code?: string;

  constructor(message: string, status?: number, code?: string) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (!isObject(value) || !("error" in value)) {
    return false;
  }
  const error = value.error;
  return (
    isObject(error) &&
    typeof error.code === "string" &&
    typeof error.message === "string"
  );
}

function isHealthResponse(value: unknown): value is HealthResponse {
  return (
    isObject(value) &&
    value.status === "ok" &&
    value.service === "poxscanner-api" &&
    typeof value.model_loaded === "boolean" &&
    typeof value.demo_mode === "boolean" &&
    typeof value.version === "string"
  );
}

function isModelInfoResponse(value: unknown): value is ModelInfoResponse {
  const input = isObject(value) ? value.input : null;
  return (
    isObject(value) &&
    typeof value.name === "string" &&
    typeof value.version === "string" &&
    typeof value.task === "string" &&
    Array.isArray(value.classes) &&
    (value.class_mapping === undefined ||
      value.class_mapping === null ||
      isObject(value.class_mapping)) &&
    (value.checkpoint === undefined ||
      value.checkpoint === null ||
      typeof value.checkpoint === "string") &&
    isObject(input) &&
    typeof input.image_size === "number" &&
    Array.isArray(input.formats) &&
    (input.normalization === undefined ||
      input.normalization === null ||
      (isObject(input.normalization) &&
        Array.isArray(input.normalization.mean) &&
        Array.isArray(input.normalization.std))) &&
    (value.threshold === undefined ||
      value.threshold === null ||
      typeof value.threshold === "number") &&
    typeof value.selection_status === "string" &&
    typeof value.demo_mode === "boolean" &&
    typeof value.disclaimer === "string"
  );
}

function isPredictResponse(value: unknown): value is PredictResponse {
  const probabilities = isObject(value) ? value.probabilities : null;
  const explanation = isObject(value) ? value.explanation : null;
  return (
    isObject(value) &&
    (value.prediction === "Monkeypox" || value.prediction === "Non-Monkeypox") &&
    typeof value.confidence === "number" &&
    isObject(probabilities) &&
    typeof probabilities.monkeypox === "number" &&
    typeof probabilities.non_monkeypox === "number" &&
    typeof value.model === "string" &&
    typeof value.version === "string" &&
    typeof value.demo_mode === "boolean" &&
    typeof value.disclaimer === "string" &&
    (explanation === null ||
      (isObject(explanation) &&
        typeof explanation.method === "string" &&
        typeof explanation.overlay_png_base64 === "string" &&
        (explanation.note === null || typeof explanation.note === "string")))
  );
}

async function readError(response: Response): Promise<ApiRequestError> {
  try {
    const body: unknown = await response.json();
    if (isApiErrorBody(body)) {
      return new ApiRequestError(
        body.error.message,
        response.status,
        body.error.code,
      );
    }
  } catch {
    /* response was not JSON */
  }
  return new ApiRequestError(
    `Request failed (${response.status})`,
    response.status,
  );
}

function throwUnreachable(): never {
  throw new ApiRequestError(
    `Cannot reach the API. Check that the server is running at ${API_BASE}.`,
  );
}

export async function health(): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  } catch {
    throwUnreachable();
  }
  if (!response.ok) {
    throw await readError(response);
  }
  const payload: unknown = await response.json();
  if (!isHealthResponse(payload)) {
    throw new ApiRequestError("Unexpected health response shape.");
  }
  return payload;
}

export async function modelInfo(): Promise<ModelInfoResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/model-info`, {
      cache: "no-store",
    });
  } catch {
    throwUnreachable();
  }
  if (!response.ok) {
    throw await readError(response);
  }
  const payload: unknown = await response.json();
  if (!isModelInfoResponse(payload)) {
    throw new ApiRequestError("Unexpected model-info response shape.");
  }
  return payload;
}

export async function predict(
  image?: File | null,
  explain = true,
  clinicalData?: Record<string, number> | null,
): Promise<PredictResponse> {
  const form = new FormData();
  if (image) {
    form.append("image", image);
  }
  if (clinicalData && Object.keys(clinicalData).length > 0) {
    form.append("clinical_data", JSON.stringify(clinicalData));
  }
  const url = `${API_BASE}/api/v1/predict?explain=${explain ? "true" : "false"}`;

  let response: Response;
  try {
    response = await fetch(url, { method: "POST", body: form });
  } catch {
    throwUnreachable();
  }
  if (!response.ok) {
    throw await readError(response);
  }
  const payload: unknown = await response.json();
  if (!isPredictResponse(payload)) {
    throw new ApiRequestError("Unexpected prediction response shape.");
  }
  return payload;
}

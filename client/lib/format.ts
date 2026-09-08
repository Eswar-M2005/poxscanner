export const DISCLAIMER =
  "Research and educational use only. This tool does not provide a medical diagnosis.";

export const DEMO_BANNER =
  "Demo mode — no trained checkpoint loaded. Results are placeholders, not model output.";

export const DISPLAY_LABELS: Record<string, string> = {
  monkeypox: "Monkeypox",
  non_monkeypox: "Non-Monkeypox",
};

export const ALLOWED_MIME_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

export const MAX_UPLOAD_BYTES = 8_000_000;

/**
 * Format a probability in [0, 1] as a one-decimal percentage string.
 */
export function formatConfidencePercent(value: number): string {
  if (!Number.isFinite(value)) {
    return "—";
  }
  const clamped = Math.min(1, Math.max(0, value));
  return `${(clamped * 100).toFixed(1)}%`;
}

export function overlayDataUrl(base64: string): string {
  const trimmed = base64.trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.startsWith("data:")) {
    return trimmed;
  }
  return `data:image/png;base64,${trimmed}`;
}

export function validateImageFile(file: File): string | null {
  if (file.size === 0) {
    return "The selected file is empty.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "Image is too large. Maximum size is 8 MB.";
  }
  if (file.type && !ALLOWED_MIME_TYPES.has(file.type)) {
    return "Please choose a JPEG, PNG, or WebP image.";
  }
  if (!file.type) {
    return "Please choose a JPEG, PNG, or WebP image.";
  }
  return null;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

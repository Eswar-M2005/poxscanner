import { describe, expect, it } from "vitest";
import {
  formatConfidencePercent,
  overlayDataUrl,
  validateImageFile,
} from "@/lib/format";

describe("formatConfidencePercent", () => {
  it("formats a mid-range probability with one decimal place", () => {
    expect(formatConfidencePercent(0.873)).toBe("87.3%");
  });

  it("formats boundaries and clamps out-of-range values", () => {
    expect(formatConfidencePercent(0)).toBe("0.0%");
    expect(formatConfidencePercent(1)).toBe("100.0%");
    expect(formatConfidencePercent(1.4)).toBe("100.0%");
    expect(formatConfidencePercent(-0.2)).toBe("0.0%");
  });

  it("returns an em dash for non-finite input", () => {
    expect(formatConfidencePercent(Number.NaN)).toBe("—");
    expect(formatConfidencePercent(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

describe("overlayDataUrl", () => {
  it("prefixes raw base64 as a PNG data URL", () => {
    expect(overlayDataUrl("abc123")).toBe("data:image/png;base64,abc123");
  });

  it("leaves an existing data URL unchanged", () => {
    const existing = "data:image/png;base64,abc123";
    expect(overlayDataUrl(existing)).toBe(existing);
  });
});

describe("validateImageFile", () => {
  it("rejects unsupported types and oversized files", () => {
    const gif = new File(["x"], "lesion.gif", { type: "image/gif" });
    expect(validateImageFile(gif)).toMatch(/JPEG, PNG, or WebP/i);

    const huge = new File([new Uint8Array(8_000_001)], "lesion.jpg", {
      type: "image/jpeg",
    });
    expect(validateImageFile(huge)).toMatch(/8 MB/i);
  });

  it("accepts a small JPEG", () => {
    const jpeg = new File([new Uint8Array(32)], "lesion.jpg", {
      type: "image/jpeg",
    });
    expect(validateImageFile(jpeg)).toBeNull();
  });
});

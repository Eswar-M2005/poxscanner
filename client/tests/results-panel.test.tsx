import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { ResultsPanel } from "@/components/ResultsPanel";

describe("ResultsPanel smoke test", () => {
  it("renders a prediction summary", () => {
    const markup = renderToStaticMarkup(
      <ResultsPanel
        result={{
          prediction: "Monkeypox",
          confidence: 0.9342,
          probabilities: { monkeypox: 0.9342, non_monkeypox: 0.0658 },
          model: "efficientnet_b0",
          version: "1.0.0",
          demo_mode: true,
          disclaimer: "Research and educational use only. This tool does not provide a medical diagnosis.",
          explanation: {
            method: "gradcam",
            overlay_png_base64: "AAA",
            note: "Demo",
          },
        }}
        originalSrc="data:image/png;base64,AAA"
      />,
    );

    expect(markup).toContain("Monkeypox");
    expect(markup).toContain("Grad-CAM overlay");
  });
});

"use client";

import React from "react";

import { ProbabilityBars } from "@/components/ProbabilityBars";
import { formatConfidencePercent, overlayDataUrl } from "@/lib/format";
import type { PredictResponse } from "@/types/api";

type ResultsPanelProps = {
  result: PredictResponse;
  originalSrc: string | null;
};

export function ResultsPanel({ result, originalSrc }: ResultsPanelProps) {
  const overlay = overlayDataUrl(result.explanation?.overlay_png_base64 ?? "");

  return (
    <section aria-labelledby="results-heading" className="mt-12">
      <h2
        id="results-heading"
        className="font-heading text-2xl font-medium tracking-tight"
      >
        Results
      </h2>

      <div className="mt-6 grid gap-8 lg:grid-cols-2">
        <div className="rounded-md border border-rule bg-cream p-5 sm:p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink-muted">
            Predicted class
          </p>
          <p className="mt-2 font-heading text-3xl font-medium text-ink">
            {result.prediction}
          </p>
          <p className="mt-3 text-ink-muted">
            Confidence{" "}
            <span className="font-semibold text-ink tabular-nums">
              {formatConfidencePercent(result.confidence)}
            </span>
          </p>

          <div className="mt-6">
            <ProbabilityBars probabilities={result.probabilities} />
          </div>

          <div className="mt-6 border-t border-rule pt-4 flex flex-wrap items-center justify-between gap-2 text-sm text-ink-muted">
            <p>
              Model{" "}
              <span className="text-ink font-medium">
                {result.model} v{result.version}
              </span>
            </p>
            {result.modality === "multimodal" || result.multimodal ? (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-300">
                Model C: Multimodal
              </span>
            ) : result.modality === "clinical_only" ? (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-purple-100 text-purple-800 border border-purple-300">
                Model B: Clinical-Only
              </span>
            ) : (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-800 border border-blue-200">
                Model A: Image-Only
              </span>
            )}
          </div>
        </div>

        {result.modality !== "clinical_only" ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            <figure>
              {originalSrc ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={originalSrc}
                  alt="Original uploaded lesion photograph"
                  className="aspect-square w-full rounded-md border border-rule bg-paper-dark object-cover"
                />
              ) : (
                <div className="flex aspect-square items-center justify-center rounded-md border border-rule bg-paper-dark text-sm text-ink-muted">
                  Original image unavailable
                </div>
              )}
              <figcaption className="mt-2 text-sm text-ink-muted">
                Original image
              </figcaption>
            </figure>
            <figure>
              {overlay ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={overlay}
                  alt="Grad-CAM overlay highlighting attended regions"
                  className="aspect-square w-full rounded-md border border-rule bg-paper-dark object-cover"
                />
              ) : (
                <div className="flex aspect-square items-center justify-center rounded-md border border-rule bg-paper-dark text-sm text-ink-muted">
                  No overlay returned
                </div>
              )}
              <figcaption className="mt-2 text-sm text-ink-muted">
                Grad-CAM overlay
                {result.explanation?.method
                  ? ` (${result.explanation.method})`
                  : ""}
              </figcaption>
            </figure>
          </div>
        ) : (
          <div className="rounded-md border border-rule bg-cream p-5 sm:p-6 flex flex-col justify-center">
            <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-ink-muted">
              Clinical Modality Input
            </h3>
            <p className="mt-2 text-sm text-ink leading-relaxed">
              No lesion photograph was provided. The prediction was computed exclusively by <span className="font-semibold text-purple-900">Model B (Clinical-Only MLP)</span> from the patient&apos;s reported symptoms.
            </p>
            <p className="mt-3 text-xs text-ink-muted">
              Upload a lesion photograph below to run joint multimodal reasoning (Model C) with visual Grad-CAM overlays.
            </p>
          </div>
        )}
      </div>

      {result.clinical_explanation && (
        <div className="mt-6 rounded-md border border-rule bg-cream p-5 sm:p-6">
          <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-ink-muted">
            Symptom Attribution
          </h3>
          <p className="mt-1 text-xs text-stone-500">
            Symptom importance derived from sensitivity analysis of the clinical MLP. Demonstrates which features influenced the multimodal prediction.
          </p>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <h4 className="text-xs font-semibold text-emerald-800 uppercase tracking-wider mb-2">
                Top Supporting Symptoms (Pushed toward {result.prediction})
              </h4>
              <ul className="space-y-1.5">
                {result.clinical_explanation.top_supporting_symptoms.length > 0 ? (
                  result.clinical_explanation.top_supporting_symptoms.map((item, i) => (
                    <li
                      key={i}
                      className="flex items-center justify-between text-xs bg-emerald-50/90 border border-emerald-200 px-2.5 py-1.5 rounded text-emerald-950 font-medium"
                    >
                      <span className="capitalize">{item.feature}</span>
                      <span className="tabular-nums font-mono text-emerald-700">
                        +{item.importance.toFixed(4)}
                      </span>
                    </li>
                  ))
                ) : (
                  <li className="text-xs text-stone-400 italic">No strong positive symptoms</li>
                )}
              </ul>
            </div>

            <div>
              <h4 className="text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2">
                Top Opposing Symptoms (Pushed away)
              </h4>
              <ul className="space-y-1.5">
                {result.clinical_explanation.top_opposing_symptoms.length > 0 ? (
                  result.clinical_explanation.top_opposing_symptoms.map((item, i) => (
                    <li
                      key={i}
                      className="flex items-center justify-between text-xs bg-stone-50 border border-stone-200 px-2.5 py-1.5 rounded text-stone-700"
                    >
                      <span className="capitalize">{item.feature}</span>
                      <span className="tabular-nums font-mono text-stone-500">
                        {item.importance.toFixed(4)}
                      </span>
                    </li>
                  ))
                ) : (
                  <li className="text-xs text-stone-400 italic">No strong opposing symptoms</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      )}

      {result.explanation?.note ? (
        <p className="mt-4 text-sm text-ink-muted">{result.explanation.note}</p>
      ) : null}
    </section>
  );
}

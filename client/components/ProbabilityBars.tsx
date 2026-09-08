"use client";

import React from "react";

import { DISPLAY_LABELS, formatConfidencePercent } from "@/lib/format";

type ProbabilityBarsProps = {
  probabilities: {
    monkeypox: number;
    non_monkeypox: number;
  };
};

const ORDER = ["monkeypox", "non_monkeypox"] as const;

export function ProbabilityBars({ probabilities }: ProbabilityBarsProps) {
  return (
    <ul className="space-y-4" aria-label="Class probabilities">
      {ORDER.map((key) => {
        const value = probabilities[key];
        const percent = Number.isFinite(value)
          ? Math.min(100, Math.max(0, value * 100))
          : 0;
        return (
          <li key={key}>
            <div className="mb-1 flex items-baseline justify-between gap-3 text-sm">
              <span className="font-medium text-ink">
                {DISPLAY_LABELS[key] ?? key}
              </span>
              <span className="tabular-nums text-ink-muted">
                {formatConfidencePercent(value)}
              </span>
            </div>
            <div
              className="h-2.5 overflow-hidden rounded-sm bg-paper-dark"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(percent)}
              aria-label={`${DISPLAY_LABELS[key]} probability`}
            >
              <div
                className="h-full rounded-sm bg-serum"
                style={{ width: `${percent}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

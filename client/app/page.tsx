"use client";

import { useEffect, useState } from "react";
import { ClinicalForm } from "@/components/ClinicalForm";
import { ResultsPanel } from "@/components/ResultsPanel";
import { Uploader } from "@/components/Uploader";
import { ApiRequestError, health, predict } from "@/lib/api";
import { DEMO_BANNER } from "@/lib/format";
import type { PredictResponse } from "@/types/api";

export default function HomePage() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [originalSrc, setOriginalSrc] = useState<string | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [symptoms, setSymptoms] = useState<Record<string, number>>({});

  useEffect(() => {
    let cancelled = false;
    health()
      .then((res) => {
        if (!cancelled) {
          setDemoMode(res.demo_mode);
        }
      })
      .catch(() => {
        /* Header already reports API reachability */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function onAnalyze(file: File) {
    setBusy(true);
    setError(null);
    setResult(null);
    const url = URL.createObjectURL(file);
    setOriginalSrc((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return url;
    });

    try {
      const hasClinical = Object.keys(symptoms).length > 0;
      const response = await predict(file, true, hasClinical ? symptoms : null);
      setResult(response);
      setDemoMode(response.demo_mode);
    } catch (caught) {
      const message =
        caught instanceof ApiRequestError
          ? caught.message
          : caught instanceof Error
            ? caught.message
            : "Analysis failed.";
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  async function onAnalyzeClinicalOnly() {
    setBusy(true);
    setError(null);
    setResult(null);
    setOriginalSrc(null);

    try {
      const response = await predict(null, false, symptoms);
      setResult(response);
      setDemoMode(response.demo_mode);
    } catch (caught) {
      const message =
        caught instanceof ApiRequestError
          ? caught.message
          : caught instanceof Error
            ? caught.message
            : "Clinical analysis failed.";
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  const hasSymptoms = Object.keys(symptoms).length > 0;

  return (
    <div>
      <p className="max-w-2xl text-lg leading-relaxed text-ink">
        Educational tool for comparing monkeypox and non-monkeypox lesion
        photographs. A classifier returns a class label, confidence, and a
        Grad-CAM overlay of attended image regions for research and teaching.
      </p>

      {demoMode ? (
        <p
          role="status"
          className="mt-6 border border-rust/40 bg-rust-soft px-4 py-3 text-sm font-medium text-rust"
        >
          {DEMO_BANNER}
        </p>
      ) : null}

      <div className="mt-8">
        <ClinicalForm
          symptoms={symptoms}
          onChange={setSymptoms}
          onAnalyzeClinicalOnly={onAnalyzeClinicalOnly}
          disabled={busy}
        />
      </div>

      <div className="mt-8">
        <Uploader onAnalyze={onAnalyze} busy={busy} hasSymptoms={hasSymptoms} />
      </div>

      {error ? (
        <p role="alert" className="mt-6 text-sm text-rust">
          {error}
        </p>
      ) : null}

      {busy ? (
        <p className="mt-6 text-sm text-ink-muted" aria-live="polite">
          Running inference…
        </p>
      ) : null}

      {result ? (
        <ResultsPanel result={result} originalSrc={originalSrc} />
      ) : null}
    </div>
  );
}

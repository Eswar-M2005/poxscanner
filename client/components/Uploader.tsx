"use client";

import { useEffect, useId, useRef, useState } from "react";
import { formatBytes, validateImageFile } from "@/lib/format";

type UploaderProps = {
  onAnalyze: (file: File) => void;
  busy: boolean;
  hasSymptoms?: boolean;
};

export function Uploader({ onAnalyze, busy, hasSymptoms }: UploaderProps) {
  const inputId = useId();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    return () => {
      if (previewRef.current) {
        URL.revokeObjectURL(previewRef.current);
      }
    };
  }, []);

  function setPreviewUrl(url: string | null) {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
    }
    previewRef.current = url;
    setPreview(url);
  }

  function applyFile(next: File | null) {
    setLocalError(null);
    if (!next) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    const problem = validateImageFile(next);
    if (problem) {
      setFile(null);
      setPreviewUrl(null);
      setLocalError(problem);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      return;
    }
    setFile(next);
    setPreviewUrl(URL.createObjectURL(next));
  }

  return (
    <section aria-labelledby={`${inputId}-heading`}>
      <h2
        id={`${inputId}-heading`}
        className="font-heading text-2xl font-medium tracking-tight"
      >
        Analyze an image
      </h2>
      <p className="mt-2 max-w-2xl text-ink-muted">
        JPEG, PNG, or WebP, up to 8 MB. The file is sent to the local API for
        classification and is not stored by this page.
      </p>

      <div
        role="button"
        tabIndex={0}
        aria-disabled={busy}
        aria-describedby={localError ? `${inputId}-error` : undefined}
        onKeyDown={(event) => {
          if (busy) {
            return;
          }
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (!busy) {
            setDragOver(true);
          }
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragOver(false);
          if (!busy) {
            applyFile(event.dataTransfer.files?.[0] ?? null);
          }
        }}
        onClick={() => {
          if (!busy) {
            fileInputRef.current?.click();
          }
        }}
        className={`mt-6 cursor-pointer rounded-md border-2 border-dashed bg-cream px-4 py-8 text-center transition-colors ${
          dragOver ? "border-serum bg-serum-soft/40" : "border-rule"
        } ${busy ? "pointer-events-none opacity-70" : ""}`}
      >
        <input
          ref={fileInputRef}
          id={inputId}
          type="file"
          accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
          className="sr-only"
          disabled={busy}
          onChange={(event) => applyFile(event.target.files?.[0] ?? null)}
        />
        <p className="text-base text-ink">
          Drop a lesion photograph here, or click to browse
        </p>
        <p className="mt-1 text-sm text-ink-muted">
          Accepted types: JPEG, PNG, WebP
        </p>
      </div>

      {localError ? (
        <p
          id={`${inputId}-error`}
          role="alert"
          className="mt-3 text-sm text-rust"
        >
          {localError}
        </p>
      ) : null}

      {file && preview ? (
        <div className="mt-6 flex flex-col gap-5 sm:flex-row sm:items-end">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="Selected lesion photograph preview"
            className="h-40 w-40 rounded-md border border-rule bg-paper-dark object-cover"
          />
          <div className="flex-1">
            <p className="text-sm font-medium text-ink">{file.name}</p>
            <p className="text-sm text-ink-muted">{formatBytes(file.size)}</p>
            <div className="mt-4 flex flex-wrap gap-3">
              <button
                type="button"
                disabled={busy}
                onClick={() => onAnalyze(file)}
                className="rounded-md bg-serum px-5 py-2.5 text-sm font-semibold text-white hover:bg-serum-hover disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busy
                  ? "Analyzing…"
                  : hasSymptoms
                  ? "Analyze Multimodal (Model C)"
                  : "Analyze Image Only (Model A)"}
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  applyFile(null);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = "";
                  }
                }}
                className="rounded-md border border-rule bg-cream px-5 py-2.5 text-sm font-medium text-ink hover:bg-paper-dark disabled:opacity-60"
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

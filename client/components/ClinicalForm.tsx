"use client";

import { useState } from "react";

const COMMON_SYMPTOMS: { id: string; label: string; category: string }[] = [
  { id: "fever", label: "Fever", category: "Systemic" },
  { id: "chills", label: "Chills", category: "Systemic" },
  { id: "headache", label: "Headache", category: "Systemic" },
  { id: "fatigue", label: "Fatigue / Exhaustion", category: "Systemic" },
  { id: "malaise", label: "General Malaise", category: "Systemic" },
  { id: "muscle pain", label: "Muscle Pain / Myalgia", category: "Pain" },
  { id: "sore throat", label: "Sore Throat", category: "Pain" },
  { id: "swollen lymph nodes", label: "Swollen Lymph Nodes", category: "Lymphatic" },
  { id: "adenomegaly", label: "Inguinal / Groin Adenopathy", category: "Lymphatic" },
  { id: "rash", label: "Generalized Skin Rash", category: "Lesions" },
  { id: "skin lesions", label: "Distinct Skin Lesions", category: "Lesions" },
  { id: "blisters", label: "Fluid-filled Blisters / Vesicles", category: "Lesions" },
  { id: "pustules", label: "Pustules (Pus-filled Lesions)", category: "Lesions" },
  { id: "ulcerative lesions", label: "Ulcerative Lesions", category: "Lesions" },
  { id: "oral and genital ulcers", label: "Oral or Genital Mucosal Ulcers", category: "Lesions" },
  { id: "itch", label: "Pruritus / Itching", category: "Lesions" },
];

interface ClinicalFormProps {
  symptoms: Record<string, number>;
  onChange: (symptoms: Record<string, number>) => void;
  onAnalyzeClinicalOnly?: () => void;
  disabled?: boolean;
}

export function ClinicalForm({ symptoms, onChange, onAnalyzeClinicalOnly, disabled }: ClinicalFormProps) {
  const [expanded, setExpanded] = useState(false);

  const toggleSymptom = (id: string) => {
    const next = { ...symptoms };
    if (next[id]) {
      delete next[id];
    } else {
      next[id] = 1;
    }
    onChange(next);
  };

  const clearAll = () => {
    onChange({});
  };

  const activeCount = Object.keys(symptoms).length;

  return (
    <div className="border border-stone-300 bg-white/60 p-5 rounded-lg shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-stone-800">
            Patient Clinical Symptoms (Optional Multimodal Input)
          </h3>
          <p className="text-xs text-stone-500 mt-0.5">
            Integrate clinical presentation from <span className="font-mono">Data.csv</span> for combined multimodal reasoning.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="text-xs font-medium text-emerald-700 hover:text-emerald-800 bg-emerald-50 hover:bg-emerald-100 border border-emerald-300 px-3 py-1.5 rounded transition"
        >
          {expanded ? "Collapse Symptoms" : `Select Symptoms (${activeCount} active)`}
        </button>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-stone-200">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5">
            {COMMON_SYMPTOMS.map((s) => {
              const checked = Boolean(symptoms[s.id]);
              return (
                <label
                  key={s.id}
                  className={`flex items-start gap-2 p-2 rounded cursor-pointer border text-xs transition select-none ${
                    checked
                      ? "bg-emerald-50/80 border-emerald-400 text-emerald-950 font-medium"
                      : "bg-white border-stone-200 text-stone-700 hover:border-stone-300"
                  } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={disabled}
                    onChange={() => toggleSymptom(s.id)}
                    className="mt-0.5 rounded text-emerald-600 focus:ring-emerald-500"
                  />
                  <span>{s.label}</span>
                </label>
              );
            })}
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-xs text-stone-500">
            <span>
              {activeCount === 0
                ? "No symptoms selected (Image-only mode will be used)"
                : `${activeCount} symptom${activeCount === 1 ? "" : "s"} selected`}
            </span>
            {activeCount > 0 && (
              <div className="flex items-center gap-3">
                {onAnalyzeClinicalOnly && (
                  <button
                    type="button"
                    onClick={onAnalyzeClinicalOnly}
                    disabled={disabled}
                    className="bg-purple-700 hover:bg-purple-800 text-white font-medium px-3.5 py-1.5 rounded transition shadow-sm"
                  >
                    Analyze Symptoms Only (Model B)
                  </button>
                )}
                <button
                  type="button"
                  onClick={clearAll}
                  disabled={disabled}
                  className="text-stone-500 hover:text-rose-600 underline"
                >
                  Clear all
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

import { DISCLAIMER } from "@/lib/format";

export function Disclaimer() {
  return (
    <aside
      role="note"
      aria-label="Research disclaimer"
      className="mt-6 border-l-4 border-rust bg-rust-soft px-4 py-3 text-sm leading-relaxed text-rust sm:text-[0.9375rem]"
    >
      {DISCLAIMER}
    </aside>
  );
}

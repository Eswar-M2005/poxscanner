import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Fraunces, Source_Sans_3 } from "next/font/google";
import { Header } from "@/components/Header";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
});

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-source-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Pox Scanner",
  description:
    "Educational research tool for binary classification of monkeypox versus non-monkeypox lesion images. Not a medical diagnosis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en" className={`${fraunces.variable} ${sourceSans.variable}`}>
      <body className="paper-surface paper-grain min-h-screen font-sans text-ink antialiased">
        <div className="relative z-10 mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-6 sm:px-6 sm:py-10">
          <Header />
          <main className="flex-1 pt-8">{children}</main>
          <footer className="mt-16 border-t border-rule pt-6 text-sm text-ink-muted">
            <p>
              Image-based educational classifier. Literature context: Setegn
              &amp; Dejene, BMC Infectious Diseases 25:419 (2025).
            </p>
          </footer>
        </div>
      </body>
    </html>
  );
}

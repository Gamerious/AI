import type { Metadata } from "next";
import { RiskCheckWizard } from "@/components/risk-check/wizard";
import { SiteHeader } from "@/components/marketing/site-header";
import { SiteFooter } from "@/components/marketing/site-footer";

export const metadata: Metadata = {
  title: "Kostenloser EU AI Act Risiko-Check",
  description:
    "Klassifizieren Sie ein KI-System Ihres Unternehmens in unter 5 Minuten nach dem EU AI Act – kostenlos, ohne Anmeldung.",
  alternates: { canonical: "/risiko-check" },
};

export default function RiskCheckPage() {
  return (
    <>
      <SiteHeader />
      <main className="bg-paper py-16">
        <div className="container-narrow">
          <div className="mx-auto max-w-2xl text-center">
            <p className="text-sm font-semibold uppercase tracking-wider text-primary">
              Kostenloses Tool · Keine Anmeldung
            </p>
            <h1 className="mt-3 text-4xl font-bold tracking-tight text-ink sm:text-5xl">
              EU AI Act Risiko-Check
            </h1>
            <p className="mt-4 text-lg text-gray-700">
              Beantworten Sie 4 Sektionen mit Ja/Nein-Fragen und finden Sie heraus, in welche
              Risikoklasse Ihr KI-System fällt – inklusive konkreter Pflichten und Gesetzesverweise.
            </p>
          </div>

          <div className="mx-auto mt-10 max-w-2xl">
            <RiskCheckWizard />
          </div>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}

import Link from "next/link";
import { ArrowRight, ShieldCheck } from "lucide-react";

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-gray-200 py-20 sm:py-28">
      <div className="container-narrow text-center">
        <div className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-primary-100 bg-primary-50 px-4 py-1.5 text-xs font-medium text-primary-700">
          <ShieldCheck className="h-3.5 w-3.5" />
          Bereit für 2. August 2026 – dem AI-Act-Stichtag
        </div>

        <h1 className="mx-auto max-w-3xl text-4xl font-bold leading-[1.1] tracking-tight text-ink sm:text-6xl">
          Der EU AI Act –{" "}
          <span className="text-primary">erledigt bis Mittag.</span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-gray-700 sm:text-xl">
          ComplAI ist die erste deutsche Self-Service-Plattform für EU-AI-Act-Compliance.
          Bauen Sie Ihre KI-Inventur, klassifizieren Sie Risiken und generieren Sie auditfähige
          Dokumente – ohne Big4-Beratung, ohne Monate-Projekte, ab €49 im Monat.
        </p>

        <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <Link href="#waitlist" className="button-primary">
            Kostenlos auf Warteliste
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
          <Link href="#problem" className="button-ghost">
            Wie es funktioniert
          </Link>
        </div>

        <p className="mt-6 text-xs text-gray-500">
          Keine Kreditkarte nötig · DSGVO-konform · Hosting in Frankfurt
        </p>

        <div className="mt-16 grid grid-cols-2 gap-6 border-t border-gray-200 pt-10 text-left sm:grid-cols-4">
          <Stat label="Bußgeld-Risiko" value="bis €35M" sub="oder 7 % des Umsatzes" />
          <Stat label="Big4-Projekt" value="€50k+" sub="3–6 Monate Dauer" />
          <Stat label="ComplAI" value="€199/Mo" sub="15 Min Onboarding" />
          <Stat label="DACH-Mittelstand" value="95.000+" sub="potentielle Kunden" />
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-gray-500">{label}</div>
      <div className="mt-1 text-2xl font-bold text-ink">{value}</div>
      <div className="text-xs text-gray-500">{sub}</div>
    </div>
  );
}

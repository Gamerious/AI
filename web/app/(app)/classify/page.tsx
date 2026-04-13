import Link from "next/link";
import { Compass, ArrowRight } from "lucide-react";
import { RiskBadge } from "@/components/app/risk-badge";

// Demo-Daten. In Produktion: aus DB laden.
const unclassified = [
  { id: "11", name: "Neues Tool (noch nicht klassifiziert)", department: "Produktion" },
  { id: "12", name: "Predictive Maintenance KI", department: "Produktion" },
];

const recentlyClassified = [
  { id: "2", name: "Personio HR-Matching", risk: "high" as const, when: "vor 2 Tagen" },
  { id: "1", name: "ChatGPT Enterprise", risk: "limited" as const, when: "vor 3 Tagen" },
  { id: "6", name: "Call-Qualitäts-Bewertung", risk: "high" as const, when: "vor 5 Tagen" },
];

export default function ClassifyPage() {
  return (
    <div className="mx-auto max-w-6xl">
      <div>
        <p className="text-sm text-gray-500">Modul 2</p>
        <h1 className="mt-1 text-3xl font-bold text-ink">Risk Classifier</h1>
        <p className="mt-2 text-sm text-gray-600">
          Geführte Einstufung nach EU AI Act Art. 5, 6 (+ Anhang III), 50 und 51.
        </p>
      </div>

      {/* Hero Card */}
      <div className="mt-8 rounded-2xl border border-primary-100 bg-gradient-to-br from-primary-50 to-white p-8">
        <div className="flex items-start gap-6">
          <div className="flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-xl bg-primary text-white">
            <Compass className="h-7 w-7" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-ink">Neue Klassifizierung starten</h2>
            <p className="mt-2 text-sm text-gray-700">
              Beantworten Sie in ca. 5 Minuten 15–25 Ja/Nein-Fragen und erhalten Sie eine
              rechtlich fundierte Risiko-Einstufung mit Begründung, Artikel-Verweisen und konkreten
              Pflichten.
            </p>
            <Link
              href="/classify/new"
              className="mt-4 inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary-700"
            >
              Wizard öffnen
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* Unclassified */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold text-ink">
          Noch nicht klassifiziert ({unclassified.length})
        </h2>
        <p className="mt-1 text-sm text-gray-600">
          Diese Systeme aus Ihrer Inventur brauchen noch eine Einstufung.
        </p>
        <ul className="mt-4 space-y-2">
          {unclassified.map((s) => (
            <li
              key={s.id}
              className="flex items-center justify-between rounded-lg border border-gray-200 bg-white p-4"
            >
              <div>
                <p className="text-sm font-medium text-ink">{s.name}</p>
                <p className="text-xs text-gray-500">{s.department}</p>
              </div>
              <Link
                href={`/classify/new?systemId=${s.id}`}
                className="rounded-md bg-primary-50 px-3 py-1 text-xs font-medium text-primary-700 hover:bg-primary-100"
              >
                Jetzt klassifizieren
              </Link>
            </li>
          ))}
        </ul>
      </div>

      {/* Recently classified */}
      <div className="mt-10">
        <h2 className="text-lg font-semibold text-ink">Zuletzt klassifiziert</h2>
        <ul className="mt-4 space-y-2">
          {recentlyClassified.map((s) => (
            <li
              key={s.id}
              className="flex items-center justify-between rounded-lg border border-gray-200 bg-white p-4"
            >
              <div className="flex items-center gap-3">
                <RiskBadge level={s.risk} />
                <span className="text-sm font-medium text-ink">{s.name}</span>
              </div>
              <span className="text-xs text-gray-500">{s.when}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

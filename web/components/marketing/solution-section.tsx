import { CheckCircle2 } from "lucide-react";

export function SolutionSection() {
  return (
    <section className="border-b border-gray-200 bg-white py-20">
      <div className="container-narrow">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wider text-primary">
              Die Lösung
            </p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink sm:text-4xl">
              Compliance, die sich selbst erklärt.
            </h2>
            <p className="mt-4 text-lg text-gray-700">
              ComplAI führt Sie Schritt für Schritt durch jede Pflicht des EU AI Act.
              Sie tragen ein, was Sie haben. Wir zeigen Ihnen, was zu tun ist – und liefern die
              Dokumente.
            </p>

            <ul className="mt-8 space-y-4">
              {[
                "15-Minuten-Onboarding mit Vorlagen für gängige KI-Tools",
                "Geführter Wizard für die Risiko-Klassifizierung",
                "Auditfähige PDFs auf Knopfdruck",
                "AI-Literacy-Training mit Zertifikat für Mitarbeiter",
                "Versionierte Templates – wir aktualisieren bei Gesetzesänderungen",
                "DSGVO-konform, Hosting in Frankfurt",
              ].map((item) => (
                <li key={item} className="flex items-start gap-3">
                  <CheckCircle2 className="mt-0.5 h-5 w-5 flex-shrink-0 text-accent-leaf" />
                  <span className="text-sm text-gray-800">{item}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="relative">
            <div className="rounded-2xl border border-gray-200 bg-paper p-6 shadow-lg">
              <div className="flex items-center gap-2 border-b border-gray-200 pb-4">
                <div className="h-3 w-3 rounded-full bg-red-300" />
                <div className="h-3 w-3 rounded-full bg-yellow-300" />
                <div className="h-3 w-3 rounded-full bg-green-300" />
                <div className="ml-3 text-xs text-gray-500">app.complai.de/inventory</div>
              </div>

              <div className="mt-4 space-y-3">
                <div className="flex items-center justify-between rounded-lg bg-white p-3 ring-1 ring-gray-200">
                  <div>
                    <div className="text-sm font-medium text-ink">ChatGPT Enterprise</div>
                    <div className="text-xs text-gray-500">Marketing · OpenAI</div>
                  </div>
                  <span className="rounded-full bg-yellow-50 px-2.5 py-1 text-xs font-medium text-yellow-700">
                    Limited Risk
                  </span>
                </div>

                <div className="flex items-center justify-between rounded-lg bg-white p-3 ring-1 ring-gray-200">
                  <div>
                    <div className="text-sm font-medium text-ink">HR Bewerber-Scoring</div>
                    <div className="text-xs text-gray-500">HR · Eigenmodell</div>
                  </div>
                  <span className="rounded-full bg-red-50 px-2.5 py-1 text-xs font-medium text-red-700">
                    High Risk
                  </span>
                </div>

                <div className="flex items-center justify-between rounded-lg bg-white p-3 ring-1 ring-gray-200">
                  <div>
                    <div className="text-sm font-medium text-ink">DeepL Pro</div>
                    <div className="text-xs text-gray-500">Allgemein · DeepL SE</div>
                  </div>
                  <span className="rounded-full bg-green-50 px-2.5 py-1 text-xs font-medium text-green-700">
                    Minimal Risk
                  </span>
                </div>

                <div className="flex items-center justify-between rounded-lg bg-white p-3 ring-1 ring-gray-200">
                  <div>
                    <div className="text-sm font-medium text-ink">GitHub Copilot</div>
                    <div className="text-xs text-gray-500">Engineering · Microsoft</div>
                  </div>
                  <span className="rounded-full bg-yellow-50 px-2.5 py-1 text-xs font-medium text-yellow-700">
                    Limited Risk
                  </span>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-between border-t border-gray-200 pt-4">
                <span className="text-xs text-gray-500">Compliance-Score</span>
                <span className="text-sm font-bold text-accent-leaf">82 %</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

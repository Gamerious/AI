import { History } from "lucide-react";

const events = [
  { id: "e1", when: "2026-04-10 14:32", user: "Sabine Weiß", action: "classified", entity: "Personio HR-Matching", detail: "Ergebnis: High-Risk" },
  { id: "e2", when: "2026-04-10 14:18", user: "Sabine Weiß", action: "created", entity: "Personio HR-Matching", detail: "Neues System angelegt" },
  { id: "e3", when: "2026-04-09 11:05", user: "Markus Hoffmann", action: "generated", entity: "Transparency Notice", detail: "ChatGPT Enterprise" },
  { id: "e4", when: "2026-04-08 09:47", user: "Eva Richter", action: "completed_training", entity: "AI Literacy Basics", detail: "Zertifikat ausgestellt" },
  { id: "e5", when: "2026-04-05 16:22", user: "System", action: "reminder_sent", entity: "Training-Reminder", detail: "3 Mitarbeiter erinnert" },
];

const actionLabels: Record<string, string> = {
  classified: "Klassifiziert",
  created: "Angelegt",
  generated: "Generiert",
  completed_training: "Training abgeschlossen",
  reminder_sent: "Reminder gesendet",
};

export default function AuditPage() {
  return (
    <div className="mx-auto max-w-6xl">
      <div>
        <p className="text-sm text-gray-500">Modul 5</p>
        <h1 className="mt-1 text-3xl font-bold text-ink">Audit Trail</h1>
        <p className="mt-2 text-sm text-gray-600">
          Revisionssichere Historie aller Aktivitäten. Nicht editierbar, exportierbar für Auditoren.
        </p>
      </div>

      <div className="mt-8 rounded-2xl border border-gray-200 bg-white">
        <div className="flex items-center justify-between border-b border-gray-200 p-5">
          <div className="flex items-center gap-2">
            <History className="h-5 w-5 text-primary" />
            <h2 className="text-sm font-semibold text-ink">Letzte Ereignisse</h2>
          </div>
          <button
            type="button"
            className="rounded-md border border-gray-200 bg-white px-3 py-1 text-xs font-medium text-gray-700 hover:bg-gray-50"
          >
            Als PDF exportieren
          </button>
        </div>

        <ol className="divide-y divide-gray-100">
          {events.map((e) => (
            <li key={e.id} className="flex items-start gap-4 p-5">
              <div className="flex-shrink-0">
                <div className="mt-1 h-2 w-2 rounded-full bg-primary" />
              </div>
              <div className="flex-1">
                <div className="flex items-baseline justify-between">
                  <p className="text-sm font-medium text-ink">
                    {actionLabels[e.action] ?? e.action}: <span className="font-normal">{e.entity}</span>
                  </p>
                  <span className="text-xs text-gray-500">{e.when}</span>
                </div>
                <p className="mt-1 text-xs text-gray-600">
                  {e.user} · {e.detail}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>

      <p className="mt-4 text-center text-xs text-gray-500">
        Append-only. Einträge können nicht verändert oder gelöscht werden.
      </p>
    </div>
  );
}

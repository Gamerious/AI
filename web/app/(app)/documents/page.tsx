import { FileText, Download, Plus } from "lucide-react";
import Link from "next/link";

const templates = [
  { key: "technical-documentation", title: "Technical Documentation", reference: "Art. 11 + Anhang IV", scope: "High-Risk" },
  { key: "risk-management", title: "Risk Management System", reference: "Art. 9", scope: "High-Risk" },
  { key: "dpia-addendum", title: "DPIA Addendum (AI-spezifisch)", reference: "DSGVO Art. 35 + AI Act", scope: "Personenbezogen" },
  { key: "transparency-notice", title: "Transparenz-Hinweis (Chatbot)", reference: "Art. 50", scope: "Limited" },
  { key: "human-oversight", title: "Human Oversight Plan", reference: "Art. 14", scope: "High-Risk" },
  { key: "post-market-monitoring", title: "Post-Market Monitoring Plan", reference: "Art. 72", scope: "High-Risk" },
  { key: "literacy-record", title: "AI Literacy Training Record", reference: "Art. 4", scope: "Alle" },
  { key: "conformity", title: "Declaration of Conformity", reference: "Art. 47", scope: "High-Risk Provider" },
];

const recent = [
  { id: "d1", title: "Personio HR-Matching — Technical Documentation", created: "2026-04-10" },
  { id: "d2", title: "ChatGPT Enterprise — Transparenz-Hinweis", created: "2026-04-09" },
  { id: "d3", title: "AI Literacy Training Record — Q1/2026", created: "2026-04-05" },
];

export default function DocumentsPage() {
  return (
    <div className="mx-auto max-w-6xl">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500">Modul 3</p>
          <h1 className="mt-1 text-3xl font-bold text-ink">Dokumente</h1>
          <p className="mt-2 text-sm text-gray-600">
            Generieren Sie auditfähige Compliance-Dokumente auf Basis Ihrer Inventur.
          </p>
        </div>
      </div>

      {/* Templates */}
      <h2 className="mt-8 text-lg font-semibold text-ink">Vorlagen</h2>
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {templates.map((t) => (
          <div key={t.key} className="rounded-2xl border border-gray-200 bg-white p-5">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-50 text-primary">
              <FileText className="h-5 w-5" />
            </div>
            <h3 className="mt-4 text-sm font-semibold text-ink">{t.title}</h3>
            <p className="mt-1 text-xs text-gray-500">{t.reference}</p>
            <p className="mt-2 inline-block rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-gray-600">
              {t.scope}
            </p>
            <button
              type="button"
              className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
            >
              <Plus className="h-3 w-3" />
              Neu erstellen
            </button>
          </div>
        ))}
      </div>

      {/* Recent */}
      <h2 className="mt-10 text-lg font-semibold text-ink">Kürzlich generiert</h2>
      <ul className="mt-4 space-y-2">
        {recent.map((r) => (
          <li
            key={r.id}
            className="flex items-center justify-between rounded-lg border border-gray-200 bg-white p-4"
          >
            <div>
              <p className="text-sm font-medium text-ink">{r.title}</p>
              <p className="text-xs text-gray-500">erstellt am {r.created}</p>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded-md border border-gray-200 bg-white px-3 py-1 text-xs font-medium text-gray-700 hover:bg-gray-50"
            >
              <Download className="h-3 w-3" />
              PDF
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

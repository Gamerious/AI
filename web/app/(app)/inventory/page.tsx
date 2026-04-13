import Link from "next/link";
import { Plus, Filter, Search } from "lucide-react";
import { RiskBadge } from "@/components/app/risk-badge";

// Demo-Daten für das Scaffold. In Produktion: aus DB via Drizzle laden.
const demoSystems = [
  {
    id: "1",
    name: "ChatGPT Enterprise",
    provider: "OpenAI",
    department: "Marketing",
    risk: "limited" as const,
    updatedAt: "2026-04-10",
  },
  {
    id: "2",
    name: "Personio HR-Matching",
    provider: "Personio SE",
    department: "HR",
    risk: "high" as const,
    updatedAt: "2026-04-09",
  },
  {
    id: "3",
    name: "GitHub Copilot",
    provider: "Microsoft",
    department: "Engineering",
    risk: "limited" as const,
    updatedAt: "2026-04-08",
  },
  {
    id: "4",
    name: "DeepL Pro",
    provider: "DeepL SE",
    department: "Allgemein",
    risk: "minimal" as const,
    updatedAt: "2026-04-07",
  },
  {
    id: "5",
    name: "Salesforce Einstein",
    provider: "Salesforce",
    department: "Sales",
    risk: "minimal" as const,
    updatedAt: "2026-04-06",
  },
  {
    id: "6",
    name: "Call-Qualitäts-Bewertung",
    provider: "Eigenentwicklung",
    department: "Customer Service",
    risk: "high" as const,
    updatedAt: "2026-04-05",
  },
  {
    id: "7",
    name: "Zendesk Answer Bot",
    provider: "Zendesk",
    department: "Customer Service",
    risk: "limited" as const,
    updatedAt: "2026-04-04",
  },
  {
    id: "8",
    name: "Microsoft Copilot (M365)",
    provider: "Microsoft",
    department: "Allgemein",
    risk: "limited" as const,
    updatedAt: "2026-04-03",
  },
  {
    id: "9",
    name: "Midjourney",
    provider: "Midjourney Inc.",
    department: "Marketing",
    risk: "limited" as const,
    updatedAt: "2026-04-02",
  },
  {
    id: "10",
    name: "Outlook Spam-Filter",
    provider: "Microsoft",
    department: "IT",
    risk: "minimal" as const,
    updatedAt: "2026-04-01",
  },
  {
    id: "11",
    name: "Neues Tool (noch nicht klassifiziert)",
    provider: "—",
    department: "Produktion",
    risk: "unclassified" as const,
    updatedAt: "2026-03-30",
  },
  {
    id: "12",
    name: "Predictive Maintenance KI",
    provider: "Eigenentwicklung",
    department: "Produktion",
    risk: "unclassified" as const,
    updatedAt: "2026-03-29",
  },
];

export default function InventoryPage() {
  return (
    <div className="mx-auto max-w-6xl">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500">Modul 1</p>
          <h1 className="mt-1 text-3xl font-bold text-ink">AI Inventory</h1>
          <p className="mt-2 text-sm text-gray-600">
            Zentrales Register aller KI-Systeme in Ihrem Unternehmen.
          </p>
        </div>
        <Link
          href="/inventory/new"
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary-700"
        >
          <Plus className="h-4 w-4" />
          System hinzufügen
        </Link>
      </div>

      {/* Toolbar */}
      <div className="mt-6 flex flex-wrap items-center gap-3 rounded-xl border border-gray-200 bg-white p-3">
        <div className="flex flex-1 items-center gap-2 rounded-lg bg-gray-50 px-3 py-2">
          <Search className="h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="System suchen..."
            className="flex-1 border-none bg-transparent text-sm text-ink outline-none placeholder:text-gray-400"
          />
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
        >
          <Filter className="h-4 w-4" />
          Filter
        </button>
      </div>

      {/* Table */}
      <div className="mt-4 overflow-hidden rounded-2xl border border-gray-200 bg-white">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                Name
              </th>
              <th className="hidden px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500 md:table-cell">
                Anbieter
              </th>
              <th className="hidden px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500 md:table-cell">
                Abteilung
              </th>
              <th className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                Risikoklasse
              </th>
              <th className="hidden px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500 lg:table-cell">
                Zuletzt aktualisiert
              </th>
              <th className="px-5 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {demoSystems.map((s) => (
              <tr key={s.id} className="transition-colors hover:bg-gray-50">
                <td className="px-5 py-4">
                  <div className="text-sm font-medium text-ink">{s.name}</div>
                  <div className="text-xs text-gray-500 md:hidden">
                    {s.provider} · {s.department}
                  </div>
                </td>
                <td className="hidden px-5 py-4 text-sm text-gray-600 md:table-cell">
                  {s.provider}
                </td>
                <td className="hidden px-5 py-4 text-sm text-gray-600 md:table-cell">
                  {s.department}
                </td>
                <td className="px-5 py-4">
                  <RiskBadge level={s.risk} />
                </td>
                <td className="hidden px-5 py-4 text-xs text-gray-500 lg:table-cell">
                  {s.updatedAt}
                </td>
                <td className="px-5 py-4 text-right">
                  <Link
                    href={`/inventory/${s.id}`}
                    className="text-xs font-medium text-primary hover:underline"
                  >
                    Öffnen
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-center text-xs text-gray-500">
        {demoSystems.length} Systeme · Aktualisieren Sie die Inventur regelmäßig, insbesondere bei
        neuen Tools oder Use-Cases.
      </p>
    </div>
  );
}

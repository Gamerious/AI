import { GraduationCap, CheckCircle2, Clock, Mail } from "lucide-react";

const employees = [
  { name: "Sabine Weiß", email: "s.weiss@firma.de", status: "completed" as const, date: "2026-04-08" },
  { name: "Markus Hoffmann", email: "m.hoffmann@firma.de", status: "completed" as const, date: "2026-04-07" },
  { name: "Eva Richter", email: "e.richter@firma.de", status: "completed" as const, date: "2026-04-05" },
  { name: "Tobias Klein", email: "t.klein@firma.de", status: "in_progress" as const, date: "2026-04-10" },
  { name: "Anna Schmidt", email: "a.schmidt@firma.de", status: "pending" as const, date: "—" },
];

const completedCount = employees.filter((e) => e.status === "completed").length;
const totalCount = employees.length;
const completionRate = Math.round((completedCount / totalCount) * 100);

export default function LiteracyPage() {
  return (
    <div className="mx-auto max-w-6xl">
      <div>
        <p className="text-sm text-gray-500">Modul 4</p>
        <h1 className="mt-1 text-3xl font-bold text-ink">AI Literacy Training</h1>
        <p className="mt-2 text-sm text-gray-600">
          Art. 4 EU AI Act verpflichtet Sie, ausreichende KI-Kompetenz aller Mitarbeiter zu
          gewährleisten. Gilt seit 2. Februar 2025.
        </p>
      </div>

      {/* Progress */}
      <div className="mt-8 rounded-2xl border border-gray-200 bg-white p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-gray-500">Abschluss-Quote</p>
            <p className="mt-1 text-4xl font-bold text-ink">{completionRate}%</p>
            <p className="mt-1 text-xs text-gray-500">
              {completedCount} von {totalCount} Mitarbeitern abgeschlossen
            </p>
          </div>
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-accent-leaf/10">
            <GraduationCap className="h-8 w-8 text-accent-leaf" />
          </div>
        </div>
        <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-gray-100">
          <div
            className="h-full bg-accent-leaf transition-all"
            style={{ width: `${completionRate}%` }}
          />
        </div>
      </div>

      {/* Course Info */}
      <div className="mt-6 rounded-2xl border border-gray-200 bg-white p-6">
        <h2 className="text-lg font-semibold text-ink">Standard-Kurs: "EU AI Act Basics"</h2>
        <p className="mt-2 text-sm text-gray-600">
          20-Minuten-Training, Wissens-Check und automatisches Zertifikat. Deckt die 4 Kern-Themen
          ab, die Art. 4 fordert.
        </p>
        <ul className="mt-4 space-y-2 text-sm text-gray-700">
          <li className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-accent-leaf" />
            Was ist KI? Technisches Grundverständnis
          </li>
          <li className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-accent-leaf" />
            Rechtlicher Rahmen (EU AI Act, DSGVO)
          </li>
          <li className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-accent-leaf" />
            Risiken und verantwortungsvoller Einsatz
          </li>
          <li className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-accent-leaf" />
            Firmenspezifische Regeln und Ansprechpartner
          </li>
        </ul>
      </div>

      {/* Employee List */}
      <h2 className="mt-8 text-lg font-semibold text-ink">Mitarbeiter</h2>
      <div className="mt-4 overflow-hidden rounded-2xl border border-gray-200 bg-white">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                Name
              </th>
              <th className="hidden px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500 sm:table-cell">
                E-Mail
              </th>
              <th className="px-5 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                Status
              </th>
              <th className="px-5 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {employees.map((e) => (
              <tr key={e.email}>
                <td className="px-5 py-4 text-sm font-medium text-ink">{e.name}</td>
                <td className="hidden px-5 py-4 text-sm text-gray-600 sm:table-cell">{e.email}</td>
                <td className="px-5 py-4">
                  <StatusPill status={e.status} date={e.date} />
                </td>
                <td className="px-5 py-4 text-right">
                  {e.status !== "completed" && (
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 rounded-md bg-primary-50 px-2.5 py-1 text-xs font-medium text-primary-700 hover:bg-primary-100"
                    >
                      <Mail className="h-3 w-3" /> Reminder
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatusPill({
  status,
  date,
}: {
  status: "completed" | "in_progress" | "pending";
  date: string;
}) {
  if (status === "completed") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-medium text-green-700 ring-1 ring-inset ring-green-200">
        <CheckCircle2 className="h-3 w-3" />
        Abgeschlossen · {date}
      </span>
    );
  }
  if (status === "in_progress") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-yellow-50 px-2.5 py-0.5 text-xs font-medium text-yellow-700 ring-1 ring-inset ring-yellow-200">
        <Clock className="h-3 w-3" />
        In Bearbeitung
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-600 ring-1 ring-inset ring-gray-200">
      Offen
    </span>
  );
}

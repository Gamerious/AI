import Link from "next/link";
import {
  Boxes,
  Compass,
  FileText,
  GraduationCap,
  ShieldCheck,
  AlertTriangle,
} from "lucide-react";

export default function DashboardPage() {
  // In Produktion: aus DB laden. Für MVP-Scaffold: Demo-Werte.
  const stats = {
    totalSystems: 12,
    classified: 10,
    highRisk: 2,
    limitedRisk: 5,
    minimalRisk: 3,
    documentsGenerated: 8,
    literacyCompleted: 42,
    literacyTotal: 58,
  };
  const complianceScore = Math.round((stats.classified / stats.totalSystems) * 100);

  return (
    <div className="mx-auto max-w-6xl">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500">Willkommen zurück</p>
          <h1 className="mt-1 text-3xl font-bold text-ink">Compliance-Dashboard</h1>
        </div>
        <Link
          href="/classify/new"
          className="hidden items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary-700 sm:inline-flex"
        >
          <Compass className="h-4 w-4" />
          Neue Klassifizierung
        </Link>
      </div>

      {/* Compliance Score */}
      <div className="mt-8 rounded-2xl border border-gray-200 bg-white p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-gray-500">Compliance-Score</p>
            <p className="mt-1 text-4xl font-bold text-ink">{complianceScore}%</p>
            <p className="mt-2 text-xs text-gray-500">
              {stats.classified} von {stats.totalSystems} Systemen klassifiziert
            </p>
          </div>
          <div className="h-16 w-16 flex items-center justify-center rounded-full bg-accent-leaf/10">
            <ShieldCheck className="h-8 w-8 text-accent-leaf" />
          </div>
        </div>
        <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-gray-100">
          <div
            className="h-full bg-accent-leaf transition-all"
            style={{ width: `${complianceScore}%` }}
          />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={<Boxes className="h-5 w-5 text-primary" />}
          label="KI-Systeme"
          value={stats.totalSystems}
          href="/inventory"
        />
        <StatCard
          icon={<AlertTriangle className="h-5 w-5 text-accent-citrus" />}
          label="High-Risk"
          value={stats.highRisk}
          href="/inventory?filter=high"
          tone="warn"
        />
        <StatCard
          icon={<FileText className="h-5 w-5 text-primary" />}
          label="Dokumente"
          value={stats.documentsGenerated}
          href="/documents"
        />
        <StatCard
          icon={<GraduationCap className="h-5 w-5 text-accent-leaf" />}
          label="Training"
          value={`${stats.literacyCompleted}/${stats.literacyTotal}`}
          href="/literacy"
        />
      </div>

      {/* Next Actions */}
      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-gray-200 bg-white p-6">
          <h2 className="text-lg font-semibold text-ink">Nächste Schritte</h2>
          <ul className="mt-4 space-y-3">
            <TodoRow
              title="2 Systeme noch nicht klassifiziert"
              href="/inventory?filter=unclassified"
              cta="Jetzt klassifizieren"
            />
            <TodoRow
              title="16 Mitarbeiter müssen noch AI-Literacy-Training abschließen"
              href="/literacy"
              cta="Training zeigen"
            />
            <TodoRow
              title="Technische Dokumentation für High-Risk-System #3 fehlt"
              href="/documents/new"
              cta="Dokument erstellen"
            />
          </ul>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-6">
          <h2 className="text-lg font-semibold text-ink">Risiko-Übersicht</h2>
          <dl className="mt-4 space-y-3">
            <RiskRow label="Verboten" count={0} color="bg-red-500" />
            <RiskRow label="High-Risk" count={stats.highRisk} color="bg-orange-500" />
            <RiskRow label="Limited Risk" count={stats.limitedRisk} color="bg-yellow-500" />
            <RiskRow label="Minimal Risk" count={stats.minimalRisk} color="bg-green-500" />
            <RiskRow
              label="Unklassifiziert"
              count={stats.totalSystems - stats.classified}
              color="bg-gray-300"
            />
          </dl>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  href,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  href: string;
  tone?: "warn";
}) {
  return (
    <Link
      href={href}
      className="group rounded-2xl border border-gray-200 bg-white p-5 transition-shadow hover:shadow-md"
    >
      <div className="flex items-center justify-between">
        <div>{icon}</div>
        {tone === "warn" && (
          <span className="rounded-full bg-accent-citrus/10 px-2 py-0.5 text-xs font-medium text-accent-citrus">
            Achtung
          </span>
        )}
      </div>
      <p className="mt-4 text-xs font-medium uppercase tracking-wider text-gray-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-ink">{value}</p>
    </Link>
  );
}

function TodoRow({
  title,
  href,
  cta,
}: {
  title: string;
  href: string;
  cta: string;
}) {
  return (
    <li className="flex items-center justify-between gap-4 rounded-lg border border-gray-200 p-3">
      <span className="text-sm text-gray-700">{title}</span>
      <Link
        href={href}
        className="flex-shrink-0 rounded-md bg-primary-50 px-3 py-1 text-xs font-medium text-primary-700 hover:bg-primary-100"
      >
        {cta}
      </Link>
    </li>
  );
}

function RiskRow({ label, count, color }: { label: string; count: number; color: string }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <div className="flex items-center gap-2">
        <span className={`inline-block h-2 w-2 rounded-full ${color}`} aria-hidden />
        <span className="text-gray-700">{label}</span>
      </div>
      <span className="font-semibold text-ink">{count}</span>
    </div>
  );
}

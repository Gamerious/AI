export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="text-3xl font-bold text-ink">Einstellungen</h1>
      <p className="mt-2 text-sm text-gray-600">
        Organisations-Profil, Nutzer, Abrechnung und API-Keys.
      </p>

      <div className="mt-8 space-y-6">
        <Section title="Organisation">
          <p className="text-sm text-gray-600">
            Unternehmensdaten, Branche, Größe, HQ-Land. Werden in generierte Dokumente übernommen.
          </p>
        </Section>

        <Section title="Team">
          <p className="text-sm text-gray-600">
            Benutzer einladen, Rollen vergeben (Admin / Editor / Viewer).
          </p>
        </Section>

        <Section title="Abrechnung">
          <p className="text-sm text-gray-600">
            Aktueller Plan, Zahlungsmethode, Rechnungen. Integration über Stripe Customer Portal.
          </p>
        </Section>

        <Section title="API">
          <p className="text-sm text-gray-600">
            API-Keys für Automatisierung. Verfügbar ab Business-Plan.
          </p>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-6">
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      <div className="mt-3">{children}</div>
    </div>
  );
}

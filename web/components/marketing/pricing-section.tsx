import Link from "next/link";
import { Check } from "lucide-react";

const tiers = [
  {
    name: "Free",
    price: "€0",
    cadence: "/Monat",
    description: "Zum Reinschnuppern. Ideal für sehr kleine Teams.",
    features: [
      "Bis zu 3 KI-Systeme",
      "Basis-Risikoklassifizierung",
      "1 PDF-Export pro Monat",
      "1 Benutzer",
    ],
    cta: "Kostenlos starten",
    highlighted: false,
  },
  {
    name: "Pro",
    price: "€199",
    cadence: "/Monat",
    description: "Der Sweet Spot für den Mittelstand.",
    features: [
      "Unbegrenzte KI-Systeme",
      "Vollständige Klassifizierung & Dokumente",
      "AI Literacy Training (50 Mitarbeiter)",
      "Audit Trail & Dashboard",
      "Bis 5 Benutzer",
      "E-Mail-Support",
    ],
    cta: "Pro testen",
    highlighted: true,
  },
  {
    name: "Business",
    price: "€499",
    cadence: "/Monat",
    description: "Für Compliance-Teams in größeren Häusern.",
    features: [
      "Alles aus Pro",
      "SSO (Google + Microsoft)",
      "AVV auf Anfrage",
      "Unbegrenzte Mitarbeiter im Training",
      "Dedizierter CSM",
      "Quartalsweise Compliance-Reviews",
    ],
    cta: "Demo anfragen",
    highlighted: false,
  },
];

export function PricingSection() {
  return (
    <section id="preise" className="border-b border-gray-200 bg-white py-20">
      <div className="container-narrow">
        <div className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-primary">Preise</p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink sm:text-4xl">
            Bezahlen Sie weniger als für eine Beratungsstunde.
          </h2>
          <p className="mt-4 text-lg text-gray-700">
            Transparente Preise, jederzeit kündbar, keine versteckten Kosten. Bei Jahreszahlung
            sparen Sie zwei Monate.
          </p>
        </div>

        <div className="mt-14 grid gap-6 lg:grid-cols-3">
          {tiers.map((tier) => (
            <div
              key={tier.name}
              className={`rounded-2xl border p-8 ${
                tier.highlighted
                  ? "border-primary bg-primary text-white shadow-xl"
                  : "border-gray-200 bg-white"
              }`}
            >
              <h3
                className={`text-sm font-semibold uppercase tracking-wider ${
                  tier.highlighted ? "text-primary-100" : "text-primary"
                }`}
              >
                {tier.name}
              </h3>
              <div className="mt-4 flex items-baseline gap-1">
                <span className="text-4xl font-bold">{tier.price}</span>
                <span
                  className={`text-sm ${
                    tier.highlighted ? "text-primary-100" : "text-gray-500"
                  }`}
                >
                  {tier.cadence}
                </span>
              </div>
              <p
                className={`mt-2 text-sm ${
                  tier.highlighted ? "text-primary-100" : "text-gray-700"
                }`}
              >
                {tier.description}
              </p>

              <ul className="mt-6 space-y-3">
                {tier.features.map((f) => (
                  <li key={f} className="flex items-start gap-2 text-sm">
                    <Check
                      className={`mt-0.5 h-4 w-4 flex-shrink-0 ${
                        tier.highlighted ? "text-white" : "text-accent-leaf"
                      }`}
                    />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>

              <Link
                href="#waitlist"
                className={`mt-8 inline-flex w-full items-center justify-center rounded-lg px-5 py-3 text-sm font-semibold transition-colors ${
                  tier.highlighted
                    ? "bg-white text-primary hover:bg-primary-50"
                    : "border border-gray-300 bg-white text-ink hover:bg-gray-50"
                }`}
              >
                {tier.cta}
              </Link>
            </div>
          ))}
        </div>

        <p className="mt-8 text-center text-sm text-gray-500">
          Brauchen Sie Enterprise-Features (On-Prem, SLA, eigene Templates)?{" "}
          <Link href="#waitlist" className="font-medium text-primary hover:underline">
            Sprechen Sie mit uns.
          </Link>
        </p>
      </div>
    </section>
  );
}

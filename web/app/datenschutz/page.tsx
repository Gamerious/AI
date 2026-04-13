import type { Metadata } from "next";
import { SiteHeader } from "@/components/marketing/site-header";
import { SiteFooter } from "@/components/marketing/site-footer";

export const metadata: Metadata = {
  title: "Datenschutzerklärung",
  description: "Information zur Verarbeitung personenbezogener Daten gemäß Art. 13 DSGVO.",
};

export default function DatenschutzPage() {
  return (
    <>
      <SiteHeader />
      <main className="bg-paper py-16">
        <article className="container-narrow prose prose-slate max-w-content">
          <h1 className="text-3xl font-bold">Datenschutzerklärung</h1>

          <p className="text-sm text-gray-500">
            Stand: April 2026. Diese Datenschutzerklärung ist eine vorläufige Version. Vor öffentlichem
            Launch wird sie durch eine vollständige, anwaltlich geprüfte Fassung ersetzt.
          </p>

          <h2 className="mt-8 text-lg font-semibold">1. Verantwortlicher</h2>
          <p className="text-sm">
            ComplAI UG (haftungsbeschränkt) i.G., Musterstraße 1, 10115 Berlin,
            hello@complai.de.
          </p>

          <h2 className="mt-6 text-lg font-semibold">2. Welche Daten wir verarbeiten</h2>
          <ul className="list-disc pl-5 text-sm">
            <li>
              <strong>Server-Logfiles</strong>: IP-Adresse, Browser-Typ, aufgerufene Seite, Zeitstempel.
              Rechtsgrundlage: Art. 6 Abs. 1 lit. f DSGVO (berechtigtes Interesse an sicherem Betrieb).
              Speicherdauer: 14 Tage.
            </li>
            <li>
              <strong>Waitlist-Daten</strong>: Geschäfts-E-Mail, Firmenname, Firmengröße. Rechtsgrundlage:
              Art. 6 Abs. 1 lit. b DSGVO (vorvertragliche Maßnahme). Speicherdauer: bis Launch oder Widerruf.
            </li>
            <li>
              <strong>Kundendaten</strong> (nach Launch): Account-Daten, Inhalte der KI-Inventur, Dokumente.
              Rechtsgrundlage: Art. 6 Abs. 1 lit. b DSGVO (Vertragserfüllung).
            </li>
          </ul>

          <h2 className="mt-6 text-lg font-semibold">3. Hosting und Sub-Auftragsverarbeiter</h2>
          <p className="text-sm">
            Wir hosten ausschließlich in der Europäischen Union. Aktuelle Sub-Processors:
          </p>
          <ul className="list-disc pl-5 text-sm">
            <li>Vercel Inc. – Hosting (EU-Region)</li>
            <li>Supabase Inc. – Datenbank, Storage (Frankfurt)</li>
            <li>Resend Inc. – Transaktionale E-Mails</li>
            <li>Stripe Payments Europe Ltd. – Zahlungsabwicklung (nach Launch)</li>
          </ul>

          <h2 className="mt-6 text-lg font-semibold">4. Cookies & Tracking</h2>
          <p className="text-sm">
            Wir setzen <strong>keine Tracking-Cookies</strong>. Wir nutzen ein cookieloses
            Webanalyse-Tool (Plausible) und benötigen daher keinen Cookie-Banner.
          </p>

          <h2 className="mt-6 text-lg font-semibold">5. Ihre Rechte</h2>
          <p className="text-sm">
            Sie haben jederzeit das Recht auf Auskunft (Art. 15), Berichtigung (Art. 16),
            Löschung (Art. 17), Einschränkung (Art. 18), Datenübertragbarkeit (Art. 20) und
            Widerspruch (Art. 21). Anfragen richten Sie bitte an hello@complai.de.
          </p>

          <h2 className="mt-6 text-lg font-semibold">6. Beschwerderecht</h2>
          <p className="text-sm">
            Sie können sich jederzeit bei einer Aufsichtsbehörde beschweren. Zuständig ist die
            Berliner Beauftragte für Datenschutz und Informationsfreiheit.
          </p>
        </article>
      </main>
      <SiteFooter />
    </>
  );
}

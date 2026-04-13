import type { Metadata } from "next";
import { SiteHeader } from "@/components/marketing/site-header";
import { SiteFooter } from "@/components/marketing/site-footer";

export const metadata: Metadata = {
  title: "Impressum",
  description: "Anbieterkennzeichnung gemäß § 5 DDG.",
};

export default function ImpressumPage() {
  return (
    <>
      <SiteHeader />
      <main className="bg-paper py-16">
        <article className="container-narrow prose prose-slate max-w-content">
          <h1 className="text-3xl font-bold">Impressum</h1>

          <p className="text-sm text-gray-500">
            Angaben gemäß § 5 Digitale-Dienste-Gesetz (DDG).
          </p>

          <h2 className="mt-8 text-lg font-semibold">Anbieter</h2>
          <p className="text-sm">
            ComplAI UG (haftungsbeschränkt) <em>(in Gründung)</em>
            <br />
            Musterstraße 1<br />
            10115 Berlin
          </p>

          <h2 className="mt-6 text-lg font-semibold">Kontakt</h2>
          <p className="text-sm">
            E-Mail: hello@complai.de
            <br />
            Web: https://complai.de
          </p>

          <h2 className="mt-6 text-lg font-semibold">Vertretungsberechtigte</h2>
          <p className="text-sm">Geschäftsführung: [wird ergänzt]</p>

          <h2 className="mt-6 text-lg font-semibold">Registereintrag</h2>
          <p className="text-sm">
            Eintragung im Handelsregister.
            <br />
            Registergericht: Amtsgericht Berlin-Charlottenburg
            <br />
            Registernummer: HRB [in Gründung]
          </p>

          <h2 className="mt-6 text-lg font-semibold">
            Verantwortlich für den Inhalt nach § 18 Abs. 2 MStV
          </h2>
          <p className="text-sm">[wird ergänzt]</p>

          <p className="mt-10 text-xs text-gray-500">
            Dieses Impressum wird vor öffentlichem Launch durch endgültige Daten ersetzt
            (UG-Gründung, Geschäftsführung, HR-Eintrag).
          </p>
        </article>
      </main>
      <SiteFooter />
    </>
  );
}

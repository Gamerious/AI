const faqs = [
  {
    q: "Sind Sie Anwälte? Ist das Rechtsberatung?",
    a: "Nein. ComplAI ist ein Software-Werkzeug, das Ihnen hilft, die Pflichten des EU AI Act zu strukturieren, zu dokumentieren und nachzuweisen. Für rechtsverbindliche Beratung sollten Sie eine Kanzlei einschalten – aber 90 % der Compliance-Arbeit ist Dokumentation und Prozesse, und genau das automatisieren wir.",
  },
  {
    q: "Was passiert, wenn sich der Gesetzestext ändert?",
    a: "Wir versionieren alle Templates und Entscheidungsbäume. Bei Änderungen aktualisieren wir die Vorlagen, benachrichtigen Sie und triggern wo nötig eine Re-Klassifizierung. Sie zahlen nicht extra dafür.",
  },
  {
    q: "Wo werden meine Daten gespeichert?",
    a: "Ausschließlich in der EU (Frankfurt). Wir nutzen Supabase mit AWS-Rechenzentren in Deutschland. Verschlüsselung in Transit (TLS 1.3) und at Rest (AES-256). DSGVO-konform, AVV auf Anfrage.",
  },
  {
    q: "Können Sie meine bestehenden Dokumente importieren?",
    a: "Ja. CSV-Import für Inventuren steht ab MVP zur Verfügung. Im weiteren Verlauf folgen Word-Templates und Excel-Importer für bestehende Risk Registers.",
  },
  {
    q: "Wir nutzen kaum KI – brauchen wir das überhaupt?",
    a: "Wahrscheinlich ja. Wenn Sie Microsoft 365 nutzen, haben Sie Copilot. Wenn Sie Salesforce nutzen, haben Sie Einstein. Fast jede moderne Software hat AI-Features. Art. 4 (AI Literacy) gilt für alle Mitarbeiter, die mit solchen Systemen arbeiten – das betrifft praktisch jedes Unternehmen.",
  },
  {
    q: "Wie sieht ein Onboarding aus?",
    a: "Sie registrieren sich, beantworten 5 kurze Fragen zu Ihrem Unternehmen, und wir erstellen automatisch eine erste Liste von wahrscheinlichen KI-Systemen in Ihrem Tech-Stack. Innerhalb von 15 Minuten haben Sie Ihre erste Klassifizierung und einen ersten PDF-Report.",
  },
  {
    q: "Was ist mit der DSGVO? Brauchen wir noch eine DSGVO-Lösung?",
    a: "Ja, der EU AI Act ergänzt die DSGVO, ersetzt sie nicht. Wir bieten DPIA-Addenda (Data Protection Impact Assessment) für KI-Systeme, die Sie zu Ihren bestehenden DSGVO-Prozessen ergänzen können.",
  },
  {
    q: "Bekomme ich den Code als On-Prem-Version?",
    a: "Im Standard nein – wir sind eine Cloud-SaaS. Für Enterprise-Kunden mit besonderen Anforderungen (Banken, kritische Infrastruktur) prüfen wir On-Prem-Optionen ab Q4/2026 auf Anfrage.",
  },
];

export function FAQSection() {
  return (
    <section id="faq" className="border-b border-gray-200 py-20">
      <div className="container-narrow">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-ink sm:text-4xl">
            Häufige Fragen
          </h2>
          <p className="mt-4 text-lg text-gray-700">
            Was uns am häufigsten gefragt wird – ehrliche Antworten.
          </p>
        </div>

        <div className="mx-auto mt-12 max-w-3xl">
          {faqs.map((item) => (
            <details
              key={item.q}
              className="group border-b border-gray-200 py-5 last:border-b-0"
            >
              <summary className="flex cursor-pointer list-none items-start justify-between gap-6 text-left text-base font-semibold text-ink">
                {item.q}
                <span className="mt-1 inline-block flex-shrink-0 text-primary transition-transform group-open:rotate-45">
                  +
                </span>
              </summary>
              <p className="mt-3 text-sm leading-relaxed text-gray-700">{item.a}</p>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}

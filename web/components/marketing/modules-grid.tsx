import {
  Boxes,
  Compass,
  FileText,
  GraduationCap,
  History,
} from "lucide-react";

const modules = [
  {
    icon: Boxes,
    title: "AI Inventory",
    description:
      "Zentrales Register aller eingesetzten KI-Systeme. Manuell oder per CSV-Import. Inklusive Vorlagen für ChatGPT, Copilot, DeepL, Salesforce Einstein und 30+ weitere.",
  },
  {
    icon: Compass,
    title: "Risk Classifier",
    description:
      "Geführter Wizard, der jedes System nach Art. 5, 6 und 50 EU AI Act klassifiziert. Mit Erklärung der Pflichten und Verweisen auf den Gesetzestext.",
  },
  {
    icon: FileText,
    title: "Document Generator",
    description:
      "Automatisch generierte Risk Assessments, technische Dokumentationen, DPIAs, Transparenzhinweise und Conformity-Erklärungen – als PDF und DOCX.",
  },
  {
    icon: GraduationCap,
    title: "AI Literacy Tracker",
    description:
      "Erfüllen Sie Art. 4: Schulung, Wissens-Check und Zertifikat für jeden Mitarbeiter. Mit Reminder-Mails und Reporting für die Aufsicht.",
  },
  {
    icon: History,
    title: "Audit Trail & Dashboard",
    description:
      "Revisionssichere Historie aller Compliance-Aktivitäten, Compliance-Score, Vorfallsregister nach Art. 73 und Vorstandsreport per Export.",
  },
];

export function ModulesGrid() {
  return (
    <section id="module" className="border-b border-gray-200 py-20">
      <div className="container-narrow">
        <div className="mx-auto max-w-2xl text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-primary">
            Fünf Module, ein System
          </p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink sm:text-4xl">
            Alles, was Sie für den AI Act brauchen.
          </h2>
          <p className="mt-4 text-lg text-gray-700">
            Jede Pflicht aus dem Gesetz hat ein passendes Modul. Sie können einzeln einsteigen oder
            alles auf einmal nutzen.
          </p>
        </div>

        <div className="mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {modules.map((m) => (
            <div
              key={m.title}
              className="rounded-2xl border border-gray-200 bg-white p-6 transition-shadow hover:shadow-md"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary-50 text-primary">
                <m.icon className="h-6 w-6" />
              </div>
              <h3 className="mt-5 text-lg font-semibold text-ink">{m.title}</h3>
              <p className="mt-2 text-sm text-gray-700">{m.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

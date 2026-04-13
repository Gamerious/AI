import { AlertTriangle, FileWarning, Clock4 } from "lucide-react";

export function ProblemSection() {
  return (
    <section id="problem" className="border-b border-gray-200 py-20">
      <div className="container-narrow">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-ink sm:text-4xl">
            Ihr Unternehmen ist bereits jetzt vermutlich nicht konform.
          </h2>
          <p className="mt-4 text-lg text-gray-700">
            Die meisten Mittelständler in DACH wissen nicht einmal, welche KI-Systeme sie einsetzen
            – geschweige denn, welche Pflichten daraus folgen.
          </p>
        </div>

        <div className="mt-14 grid gap-6 sm:grid-cols-3">
          <ProblemCard
            icon={<AlertTriangle className="h-6 w-6 text-accent-citrus" />}
            title="Schatten-KI überall"
            text="Marketing nutzt ChatGPT, HR scort Bewerber mit AI, Engineering trainiert Modelle. Niemand hat einen Überblick."
          />
          <ProblemCard
            icon={<FileWarning className="h-6 w-6 text-accent-crimson" />}
            title="Bußgelder bis €35 Mio."
            text="Verstöße gegen Art. 5 (verbotene KI) bringen bis zu 7 % des globalen Jahresumsatzes als Strafe."
          />
          <ProblemCard
            icon={<Clock4 className="h-6 w-6 text-primary" />}
            title="Stichtag 2. August 2026"
            text="Der Großteil der Pflichten greift in wenigen Monaten. Big4 hat keine Kapazität mehr für den Mittelstand."
          />
        </div>

        <div className="mt-14 rounded-2xl border border-gray-200 bg-white p-8">
          <h3 className="text-xl font-semibold text-ink">Was der EU AI Act konkret verlangt</h3>
          <ul className="mt-6 space-y-3 text-sm text-gray-700">
            <ListItem strong="Art. 4 – AI Literacy">
              Alle Mitarbeiter, die mit KI arbeiten, müssen geschult sein. Gilt seit Februar 2025.
            </ListItem>
            <ListItem strong="Art. 5 – Verbotene KI">
              Social Scoring, Emotion-Detection im Job, biometrische Echtzeit-Identifikation.
            </ListItem>
            <ListItem strong="Art. 6 + Anhang III – High-Risk-Systeme">
              Bewerber-Screening, Kreditscoring, Bildungs-KI, Sicherheits-Komponenten.
            </ListItem>
            <ListItem strong="Art. 11 + Anhang IV – Technische Dokumentation">
              Vollständige technische Beschreibung jedes High-Risk-Systems.
            </ListItem>
            <ListItem strong="Art. 50 – Transparenzpflicht">
              Chatbots, Deepfakes und KI-generierte Inhalte müssen kennzeichnet sein.
            </ListItem>
            <ListItem strong="Art. 72 – Post-Market Monitoring">
              Laufende Überwachung von High-Risk-Systemen plus Vorfallsmeldung.
            </ListItem>
          </ul>
        </div>
      </div>
    </section>
  );
}

function ProblemCard({
  icon,
  title,
  text,
}: {
  icon: React.ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-6">
      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gray-50">
        {icon}
      </div>
      <h3 className="mt-4 text-lg font-semibold text-ink">{title}</h3>
      <p className="mt-2 text-sm text-gray-700">{text}</p>
    </div>
  );
}

function ListItem({ strong, children }: { strong: string; children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="mt-2 inline-block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
      <span>
        <span className="font-semibold text-ink">{strong}: </span>
        {children}
      </span>
    </li>
  );
}

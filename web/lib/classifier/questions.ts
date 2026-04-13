/**
 * Fragebogen für den Risk Classifier.
 *
 * Die Fragen sind nach Sektionen gruppiert, die in der UI als Schritte
 * angezeigt werden. Reihenfolge entspricht der Prüfreihenfolge in
 * `engine.ts` (Verbote zuerst, dann GPAI, dann High-Risk, dann Limited).
 */

import type { ClassifierAnswers } from "./engine";

export interface Question {
  id: keyof ClassifierAnswers;
  text: string;
  helper?: string;
  reference: string;
}

export interface Section {
  id: string;
  title: string;
  description: string;
  questions: Question[];
}

export const SECTIONS: Section[] = [
  {
    id: "prohibited",
    title: "Verbotene Praktiken",
    description:
      "Zuerst prüfen wir, ob das System unter eines der Verbote nach Art. 5 EU AI Act fällt. Bereits eine 'Ja'-Antwort führt zu einer verbotenen Klassifizierung.",
    questions: [
      {
        id: "manipulativeTechniques",
        text: "Verwendet das System unterschwellige oder manipulative Techniken, die das Verhalten der Nutzer wesentlich verzerren?",
        helper: "Beispiele: subliminale Werbe-KI, Dark Patterns, gezieltes Triggering vulnerabler Gruppen.",
        reference: "Art. 5 Abs. 1 lit. a",
      },
      {
        id: "exploitsVulnerabilities",
        text: "Nutzt das System gezielt Schwächen aufgrund von Alter, Behinderung oder sozio-ökonomischer Lage aus?",
        reference: "Art. 5 Abs. 1 lit. b",
      },
      {
        id: "socialScoring",
        text: "Bewertet das System Personen anhand ihres allgemeinen Sozialverhaltens (Social Scoring)?",
        helper: "Nicht gemeint ist sektorspezifisches Bonitäts-Scoring im Finanzbereich.",
        reference: "Art. 5 Abs. 1 lit. c",
      },
      {
        id: "emotionDetectionWorkplace",
        text: "Wird das System zur Emotionserkennung am Arbeitsplatz oder in Bildungseinrichtungen eingesetzt?",
        helper: "Ausnahmen gelten für medizinische und sicherheitstechnische Zwecke.",
        reference: "Art. 5 Abs. 1 lit. f",
      },
      {
        id: "biometricCategorization",
        text: "Kategorisiert das System Personen biometrisch nach sensiblen Merkmalen (politische Meinung, Religion, sexuelle Orientierung)?",
        reference: "Art. 5 Abs. 1 lit. g",
      },
      {
        id: "realtimeBiometricRemote",
        text: "Wird das System für die Echtzeit-Fernidentifizierung im öffentlichen Raum durch Strafverfolgung eingesetzt?",
        reference: "Art. 5 Abs. 1 lit. h",
      },
    ],
  },
  {
    id: "gpai",
    title: "General-Purpose AI",
    description:
      "Falls Ihr System ein General-Purpose-AI-Modell ist (z.B. ein eigener LLM oder ein Foundation-Modell), gelten gesonderte Regeln nach Art. 51 ff.",
    questions: [
      {
        id: "isGeneralPurposeModel",
        text: "Handelt es sich um ein General-Purpose-AI-Modell, das selbst trainiert oder bereitgestellt wird?",
        helper:
          "Reine Nutzung von ChatGPT, Claude, Gemini etc. zählt nicht – nur, wenn Sie selbst ein Foundation-Modell veröffentlichen oder hosten.",
        reference: "Art. 51",
      },
      {
        id: "generalPurposeWithSystemicRisk",
        text: "Hat das Modell systemisches Risiko (Trainingsaufwand > 10^25 FLOPs, oder von der Kommission als systemisch eingestuft)?",
        reference: "Art. 51 Abs. 1 + Art. 55",
      },
    ],
  },
  {
    id: "high-risk",
    title: "High-Risk Anwendungsfälle",
    description:
      "High-Risk-Systeme nach Anhang III unterliegen umfassenden Pflichten (Risikomanagement, Dokumentation, menschliche Aufsicht). Wenn auch nur eine Frage mit Ja beantwortet wird, ist das System High-Risk.",
    questions: [
      {
        id: "safetyComponentRegulated",
        text: "Ist das System eine Sicherheitskomponente eines Produkts, das unter EU-Harmonisierungsrechtsvorschriften fällt (Maschinen, Spielzeug, Medizinprodukte etc.)?",
        reference: "Art. 6 Abs. 1 + Anhang I",
      },
      {
        id: "biometricsForId",
        text: "Wird das System zur biometrischen Identifizierung oder Kategorisierung eingesetzt (außerhalb der Verbote von Art. 5)?",
        reference: "Anhang III Nr. 1",
      },
      {
        id: "criticalInfrastructure",
        text: "Wird das System zur Verwaltung oder zum Betrieb kritischer Infrastruktur eingesetzt (Strom, Wasser, Verkehr, digitale Infrastruktur)?",
        reference: "Anhang III Nr. 2",
      },
      {
        id: "educationAccess",
        text: "Beeinflusst das System den Zugang zu oder die Bewertung in Bildung und beruflicher Bildung?",
        helper: "Beispiele: automatische Bewertung von Prüfungen, Zulassungsentscheidungen.",
        reference: "Anhang III Nr. 3",
      },
      {
        id: "employmentDecisions",
        text: "Wird das System für Personalrekrutierung, Bewerberauswahl, Beförderung, Kündigung oder Aufgabenverteilung eingesetzt?",
        helper: "Auch Bewerber-Screening durch CV-Parser ist betroffen.",
        reference: "Anhang III Nr. 4",
      },
      {
        id: "essentialServices",
        text: "Beeinflusst das System den Zugang zu wesentlichen privaten Diensten (Kreditscoring, Versicherungs-Underwriting) oder öffentlichen Leistungen (Sozialleistungen)?",
        reference: "Anhang III Nr. 5",
      },
      {
        id: "lawEnforcement",
        text: "Wird das System für Strafverfolgung eingesetzt (Risk-Profiling, Beweisanalyse, Predictive Policing)?",
        reference: "Anhang III Nr. 6",
      },
      {
        id: "migrationAsylum",
        text: "Wird das System für Migrations-, Asyl- oder Grenzkontroll-Zwecke eingesetzt?",
        reference: "Anhang III Nr. 7",
      },
      {
        id: "justiceAdministration",
        text: "Wird das System in der Rechtspflege oder zur Beeinflussung demokratischer Prozesse eingesetzt?",
        reference: "Anhang III Nr. 8",
      },
    ],
  },
  {
    id: "limited",
    title: "Transparenz-Pflichten (Art. 50)",
    description:
      "Auch wenn das System nicht High-Risk ist, können Transparenzpflichten greifen – z.B. wenn Nutzer mit einem Chatbot interagieren oder synthetische Inhalte erzeugt werden.",
    questions: [
      {
        id: "interactsWithHumans",
        text: "Interagiert das System direkt mit natürlichen Personen (z.B. Chatbot, Voice-Agent)?",
        reference: "Art. 50 Abs. 1",
      },
      {
        id: "generatesSyntheticContent",
        text: "Erzeugt das System synthetische Audio-, Bild-, Video- oder Textinhalte?",
        reference: "Art. 50 Abs. 2",
      },
      {
        id: "emotionRecognitionGeneral",
        text: "Verwendet das System Emotionserkennung (außerhalb der verbotenen Bereiche aus Art. 5)?",
        reference: "Art. 50 Abs. 3",
      },
      {
        id: "deepfake",
        text: "Erzeugt das System Bild-, Audio- oder Videoinhalte, die echten Personen, Objekten oder Ereignissen ähneln (Deepfake)?",
        reference: "Art. 50 Abs. 4",
      },
    ],
  },
];

export const ALL_QUESTIONS: Question[] = SECTIONS.flatMap((s) => s.questions);

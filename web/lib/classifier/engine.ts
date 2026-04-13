/**
 * EU AI Act Risk Classifier Engine
 *
 * Implementiert eine vereinfachte, regelbasierte Klassifizierung nach
 * Verordnung (EU) 2024/1689 ("EU AI Act"). Die Klassifizierung ersetzt
 * keine juristische Beratung, gibt aber für die häufigsten Fälle das
 * korrekte Ergebnis und liefert Erklärungen + Gesetzes-Referenzen.
 *
 * Versions-Hinweis: Die Logik ist versioniert über `ENGINE_VERSION`.
 * Bei Änderungen muss das Versions-Feld inkrementiert werden, damit
 * bestehende Klassifizierungen erkennbar zu einer alten Version gehören.
 */

export const ENGINE_VERSION = "0.1.0" as const;

export type RiskLevel = "prohibited" | "high" | "limited" | "minimal" | "gpai";

export type Answer = "yes" | "no" | "unknown";

export interface ClassifierAnswers {
  // Art. 5 – Verbotene Praktiken
  socialScoring?: Answer;
  emotionDetectionWorkplace?: Answer;
  biometricCategorization?: Answer;
  realtimeBiometricRemote?: Answer;
  exploitsVulnerabilities?: Answer;
  manipulativeTechniques?: Answer;

  // GPAI
  isGeneralPurposeModel?: Answer;
  generalPurposeWithSystemicRisk?: Answer;

  // Art. 6 / Anhang III – High-Risk
  safetyComponentRegulated?: Answer;
  biometricsForId?: Answer;
  criticalInfrastructure?: Answer;
  educationAccess?: Answer;
  employmentDecisions?: Answer;
  essentialServices?: Answer;
  lawEnforcement?: Answer;
  migrationAsylum?: Answer;
  justiceAdministration?: Answer;

  // Art. 50 – Limited Risk / Transparenz
  interactsWithHumans?: Answer; // Chatbots
  generatesSyntheticContent?: Answer; // Bilder/Videos/Audio
  emotionRecognitionGeneral?: Answer;
  deepfake?: Answer;
}

export interface ClassificationResult {
  level: RiskLevel;
  rationale: string;
  obligations: Obligation[];
  legalReferences: string[];
  engineVersion: typeof ENGINE_VERSION;
}

export interface Obligation {
  id: string;
  title: string;
  description: string;
  reference: string;
}

const yes = (a: Answer | undefined) => a === "yes";

export function classify(answers: ClassifierAnswers): ClassificationResult {
  // 1. Verbotene Praktiken (Art. 5) – höchste Priorität
  if (yes(answers.socialScoring)) {
    return prohibited(
      "Allgemeines Social Scoring durch öffentliche oder private Stellen ist nach Art. 5 Abs. 1 lit. c verboten.",
      ["AI Act Art. 5 Abs. 1 lit. c"],
    );
  }
  if (yes(answers.emotionDetectionWorkplace)) {
    return prohibited(
      "Emotionserkennung am Arbeitsplatz oder in Bildungseinrichtungen ist nach Art. 5 Abs. 1 lit. f verboten (Ausnahmen: medizinische und sicherheitstechnische Zwecke).",
      ["AI Act Art. 5 Abs. 1 lit. f"],
    );
  }
  if (yes(answers.biometricCategorization)) {
    return prohibited(
      "Biometrische Kategorisierung zur Ableitung sensibler Merkmale (politische Meinung, sexuelle Orientierung, religiöse Überzeugung etc.) ist nach Art. 5 Abs. 1 lit. g verboten.",
      ["AI Act Art. 5 Abs. 1 lit. g"],
    );
  }
  if (yes(answers.realtimeBiometricRemote)) {
    return prohibited(
      "Echtzeit-Fernidentifizierung im öffentlichen Raum durch Strafverfolgung ist nach Art. 5 Abs. 1 lit. h grundsätzlich verboten (sehr enge Ausnahmen).",
      ["AI Act Art. 5 Abs. 1 lit. h"],
    );
  }
  if (yes(answers.exploitsVulnerabilities)) {
    return prohibited(
      "KI, die Schwächen aufgrund von Alter, Behinderung oder sozio-ökonomischer Lage ausnutzt, ist nach Art. 5 Abs. 1 lit. b verboten.",
      ["AI Act Art. 5 Abs. 1 lit. b"],
    );
  }
  if (yes(answers.manipulativeTechniques)) {
    return prohibited(
      "Unterschwellige oder manipulative Techniken, die das Verhalten wesentlich verzerren und Schaden verursachen können, sind nach Art. 5 Abs. 1 lit. a verboten.",
      ["AI Act Art. 5 Abs. 1 lit. a"],
    );
  }

  // 2. GPAI (Art. 51 ff.)
  if (yes(answers.isGeneralPurposeModel)) {
    if (yes(answers.generalPurposeWithSystemicRisk)) {
      return {
        level: "gpai",
        rationale:
          "Es handelt sich um ein General-Purpose-AI-Modell mit systemischem Risiko (z.B. > 10^25 FLOPs Trainingsaufwand). Verschärfte Pflichten gelten nach Art. 55.",
        legalReferences: ["AI Act Art. 51", "AI Act Art. 55"],
        obligations: [
          obl("gpai-eval", "Modell-Evaluierung", "Adversariale Tests und Risikobewertung.", "Art. 55 Abs. 1 lit. a"),
          obl("gpai-incident", "Vorfallsmeldung", "Schwere Vorfälle an die Kommission melden.", "Art. 55 Abs. 1 lit. c"),
          obl("gpai-cybersec", "Cybersicherheit", "Adäquates Niveau an Cybersicherheit.", "Art. 55 Abs. 1 lit. d"),
        ],
        engineVersion: ENGINE_VERSION,
      };
    }
    return {
      level: "gpai",
      rationale:
        "Es handelt sich um ein General-Purpose-AI-Modell. Pflichten zur technischen Dokumentation, Trainingsdaten-Zusammenfassung und Urheberrechts-Compliance.",
      legalReferences: ["AI Act Art. 51", "AI Act Art. 53"],
      obligations: [
        obl("gpai-techdoc", "Technische Dokumentation", "Erstellen und Aufbewahren der technischen Dokumentation gemäß Anhang XI.", "Art. 53 Abs. 1 lit. a"),
        obl("gpai-copyright", "Urheberrechts-Policy", "Policy zur Einhaltung des Urheberrechts beim Training.", "Art. 53 Abs. 1 lit. c"),
        obl("gpai-training-summary", "Trainingsdaten-Zusammenfassung", "Öffentliche Zusammenfassung der Trainingsdaten.", "Art. 53 Abs. 1 lit. d"),
      ],
      engineVersion: ENGINE_VERSION,
    };
  }

  // 3. High-Risk (Art. 6 + Anhang III)
  const highRiskTriggers: { key: keyof ClassifierAnswers; description: string; ref: string }[] = [
    { key: "safetyComponentRegulated", description: "Sicherheitskomponente eines Produkts, das unter Harmonisierungsrechtsvorschriften (Anhang I) fällt", ref: "Art. 6 Abs. 1" },
    { key: "biometricsForId", description: "Biometrische Identifizierung und Kategorisierung (Anhang III Nr. 1)", ref: "Anhang III Nr. 1" },
    { key: "criticalInfrastructure", description: "Verwaltung und Betrieb kritischer Infrastruktur (Anhang III Nr. 2)", ref: "Anhang III Nr. 2" },
    { key: "educationAccess", description: "Zugang zu Bildung und beruflicher Bildung (Anhang III Nr. 3)", ref: "Anhang III Nr. 3" },
    { key: "employmentDecisions", description: "Beschäftigung, Personalmanagement (Anhang III Nr. 4)", ref: "Anhang III Nr. 4" },
    { key: "essentialServices", description: "Zugang zu wesentlichen privaten und öffentlichen Diensten (Anhang III Nr. 5)", ref: "Anhang III Nr. 5" },
    { key: "lawEnforcement", description: "Strafverfolgung (Anhang III Nr. 6)", ref: "Anhang III Nr. 6" },
    { key: "migrationAsylum", description: "Migration, Asyl, Grenzkontrolle (Anhang III Nr. 7)", ref: "Anhang III Nr. 7" },
    { key: "justiceAdministration", description: "Rechtspflege und demokratische Prozesse (Anhang III Nr. 8)", ref: "Anhang III Nr. 8" },
  ];

  const highRiskHits = highRiskTriggers.filter((t) => yes(answers[t.key]));
  if (highRiskHits.length > 0) {
    return {
      level: "high",
      rationale: `High-Risk-System nach EU AI Act. Auslöser: ${highRiskHits.map((h) => h.description).join("; ")}.`,
      legalReferences: highRiskHits.map((h) => `AI Act ${h.ref}`).concat(["AI Act Art. 6"]),
      obligations: highRiskObligations(),
      engineVersion: ENGINE_VERSION,
    };
  }

  // 4. Limited Risk (Art. 50 – Transparenzpflichten)
  const limitedRiskHits: string[] = [];
  if (yes(answers.interactsWithHumans)) limitedRiskHits.push("Interaktion mit natürlichen Personen (Chatbot)");
  if (yes(answers.generatesSyntheticContent)) limitedRiskHits.push("Erzeugung synthetischer Inhalte");
  if (yes(answers.emotionRecognitionGeneral)) limitedRiskHits.push("Emotionserkennungssystem");
  if (yes(answers.deepfake)) limitedRiskHits.push("Deepfake-Erzeugung");

  if (limitedRiskHits.length > 0) {
    return {
      level: "limited",
      rationale: `Limited-Risk-System mit Transparenzpflichten nach Art. 50. Auslöser: ${limitedRiskHits.join("; ")}.`,
      legalReferences: ["AI Act Art. 50"],
      obligations: [
        obl(
          "transparency-notice",
          "Transparenzhinweis",
          "Nutzer müssen erkennen, dass sie mit einem KI-System interagieren bzw. dass Inhalte KI-generiert sind.",
          "Art. 50 Abs. 1, 2, 4",
        ),
        obl(
          "deepfake-marking",
          "Deepfake-Kennzeichnung",
          "Bild-, Audio- oder Videoinhalte, die echten Personen ähneln, müssen als KI-generiert ausgewiesen sein.",
          "Art. 50 Abs. 4",
        ),
      ],
      engineVersion: ENGINE_VERSION,
    };
  }

  // 5. Minimal Risk
  return {
    level: "minimal",
    rationale:
      "Das System fällt nach aktueller Bewertung in die Kategorie 'Minimal Risk'. Es bestehen keine spezifischen Pflichten aus dem EU AI Act, jedoch bleiben Art. 4 (AI Literacy) und allgemeine DSGVO-Anforderungen in Kraft.",
    legalReferences: ["AI Act Art. 4"],
    obligations: [
      obl(
        "ai-literacy",
        "AI Literacy",
        "Mitarbeiter, die das System bedienen, müssen eine ausreichende KI-Kompetenz nachweisen.",
        "Art. 4",
      ),
    ],
    engineVersion: ENGINE_VERSION,
  };
}

function obl(id: string, title: string, description: string, reference: string): Obligation {
  return { id, title, description, reference };
}

function prohibited(rationale: string, refs: string[]): ClassificationResult {
  return {
    level: "prohibited",
    rationale,
    legalReferences: refs,
    obligations: [
      obl(
        "stop-deployment",
        "Einsatz beenden",
        "Verbotene KI-Praktiken müssen unverzüglich eingestellt werden.",
        "Art. 5",
      ),
    ],
    engineVersion: ENGINE_VERSION,
  };
}

function highRiskObligations(): Obligation[] {
  return [
    obl(
      "risk-management",
      "Risikomanagementsystem",
      "Iteratives, dokumentiertes Risikomanagement über den gesamten Lebenszyklus.",
      "Art. 9",
    ),
    obl(
      "data-governance",
      "Daten-Governance",
      "Trainings-, Validierungs- und Testdatensätze müssen relevant, repräsentativ, fehlerfrei und vollständig sein.",
      "Art. 10",
    ),
    obl(
      "technical-documentation",
      "Technische Dokumentation",
      "Vor Inverkehrbringen vollständige technische Dokumentation gemäß Anhang IV.",
      "Art. 11 + Anhang IV",
    ),
    obl(
      "record-keeping",
      "Protokollierung",
      "Automatische Aufzeichnung von Vorgängen über die Lebensdauer.",
      "Art. 12",
    ),
    obl(
      "transparency",
      "Transparenz und Information",
      "Bereitstellung verständlicher Informationen für Betreiber.",
      "Art. 13",
    ),
    obl(
      "human-oversight",
      "Menschliche Aufsicht",
      "Geeignete Maßnahmen für die menschliche Kontrolle des Systems.",
      "Art. 14",
    ),
    obl(
      "accuracy-robustness",
      "Genauigkeit, Robustheit, Cybersicherheit",
      "Adäquates Maß an Genauigkeit, Robustheit und Cybersicherheit.",
      "Art. 15",
    ),
    obl(
      "post-market-monitoring",
      "Post-Market-Monitoring",
      "Aktive Überwachung der Performance nach Inverkehrbringen.",
      "Art. 72",
    ),
    obl(
      "incident-reporting",
      "Vorfallsmeldung",
      "Schwere Vorfälle an die zuständige Marktaufsichtsbehörde melden.",
      "Art. 73",
    ),
  ];
}

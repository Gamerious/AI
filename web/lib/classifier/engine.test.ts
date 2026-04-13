/**
 * Schnelle Sanity-Tests für die Classifier-Engine.
 * Kann mit Vitest oder als plain TS-Skript ausgeführt werden.
 */

import { classify, type ClassifierAnswers } from "./engine";

type Case = {
  name: string;
  answers: ClassifierAnswers;
  expectLevel: ReturnType<typeof classify>["level"];
};

const cases: Case[] = [
  {
    name: "Social Scoring → prohibited",
    answers: { socialScoring: "yes" },
    expectLevel: "prohibited",
  },
  {
    name: "Emotion Detection im Job → prohibited",
    answers: { emotionDetectionWorkplace: "yes" },
    expectLevel: "prohibited",
  },
  {
    name: "GPAI ohne systemisches Risiko → gpai",
    answers: { isGeneralPurposeModel: "yes" },
    expectLevel: "gpai",
  },
  {
    name: "GPAI mit systemischem Risiko → gpai mit verschärften Pflichten",
    answers: {
      isGeneralPurposeModel: "yes",
      generalPurposeWithSystemicRisk: "yes",
    },
    expectLevel: "gpai",
  },
  {
    name: "HR Bewerber-Screening → high",
    answers: { employmentDecisions: "yes" },
    expectLevel: "high",
  },
  {
    name: "Kritische Infrastruktur → high",
    answers: { criticalInfrastructure: "yes" },
    expectLevel: "high",
  },
  {
    name: "Chatbot, der mit Nutzern spricht → limited",
    answers: { interactsWithHumans: "yes" },
    expectLevel: "limited",
  },
  {
    name: "Deepfake-Generator → limited",
    answers: { deepfake: "yes" },
    expectLevel: "limited",
  },
  {
    name: "Standard DeepL Übersetzung im Marketing → minimal",
    answers: {},
    expectLevel: "minimal",
  },
  {
    name: "GitHub Copilot (interaktiv mit Mensch) → limited",
    answers: { interactsWithHumans: "yes" },
    expectLevel: "limited",
  },
  {
    name: "Verbot hat Vorrang vor High-Risk",
    answers: {
      socialScoring: "yes",
      employmentDecisions: "yes",
    },
    expectLevel: "prohibited",
  },
];

let failed = 0;
for (const c of cases) {
  const result = classify(c.answers);
  const ok = result.level === c.expectLevel;
  if (!ok) {
    failed++;
    console.error(
      `FAIL: ${c.name}\n  expected: ${c.expectLevel}\n  got: ${result.level}\n  rationale: ${result.rationale}`,
    );
  } else {
    console.log(`PASS: ${c.name} → ${result.level}`);
  }
}

if (failed > 0) {
  console.error(`\n${failed}/${cases.length} tests failed.`);
  process.exit(1);
} else {
  console.log(`\nAll ${cases.length} tests passed.`);
}

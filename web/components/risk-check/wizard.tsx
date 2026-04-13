"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { ArrowLeft, ArrowRight, AlertTriangle, ShieldCheck, Info, Ban, Sparkles } from "lucide-react";
import { classify, type Answer, type ClassifierAnswers, type RiskLevel } from "@/lib/classifier/engine";
import { SECTIONS } from "@/lib/classifier/questions";

const initialAnswers: ClassifierAnswers = {};

const levelMeta: Record<
  RiskLevel,
  { title: string; tone: string; icon: typeof ShieldCheck }
> = {
  prohibited: {
    title: "Verboten (Prohibited)",
    tone: "bg-red-50 text-red-800 border-red-200",
    icon: Ban,
  },
  high: {
    title: "High-Risk",
    tone: "bg-orange-50 text-orange-800 border-orange-200",
    icon: AlertTriangle,
  },
  limited: {
    title: "Limited Risk",
    tone: "bg-yellow-50 text-yellow-800 border-yellow-200",
    icon: Info,
  },
  minimal: {
    title: "Minimal Risk",
    tone: "bg-green-50 text-green-800 border-green-200",
    icon: ShieldCheck,
  },
  gpai: {
    title: "General-Purpose AI",
    tone: "bg-purple-50 text-purple-800 border-purple-200",
    icon: Sparkles,
  },
};

export function RiskCheckWizard() {
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<ClassifierAnswers>(initialAnswers);
  const [finished, setFinished] = useState(false);

  const section = SECTIONS[step];
  const isLast = step === SECTIONS.length - 1;

  const result = useMemo(() => (finished ? classify(answers) : null), [finished, answers]);

  function setAnswer(id: keyof ClassifierAnswers, value: Answer) {
    setAnswers((prev) => ({ ...prev, [id]: value }));
  }

  function next() {
    if (isLast) {
      setFinished(true);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      setStep((s) => s + 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  function back() {
    if (finished) {
      setFinished(false);
      return;
    }
    setStep((s) => Math.max(0, s - 1));
  }

  function reset() {
    setStep(0);
    setAnswers(initialAnswers);
    setFinished(false);
  }

  if (finished && result) {
    const meta = levelMeta[result.level];
    const Icon = meta.icon;
    return (
      <div className="rounded-2xl border border-gray-200 bg-white p-8 shadow-sm">
        <div className={`flex items-start gap-4 rounded-xl border p-6 ${meta.tone}`}>
          <Icon className="h-8 w-8 flex-shrink-0" />
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider opacity-70">
              Klassifikations-Ergebnis
            </p>
            <h2 className="mt-1 text-2xl font-bold">{meta.title}</h2>
            <p className="mt-2 text-sm">{result.rationale}</p>
          </div>
        </div>

        <div className="mt-8">
          <h3 className="text-lg font-semibold text-ink">Pflichten für Ihr System</h3>
          <ul className="mt-4 space-y-3">
            {result.obligations.map((o) => (
              <li
                key={o.id}
                className="rounded-xl border border-gray-200 bg-gray-50 p-4"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h4 className="text-sm font-semibold text-ink">{o.title}</h4>
                    <p className="mt-1 text-sm text-gray-700">{o.description}</p>
                  </div>
                  <span className="flex-shrink-0 rounded-full bg-white px-2.5 py-0.5 text-xs font-medium text-primary ring-1 ring-primary-100">
                    {o.reference}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-8">
          <h3 className="text-sm font-semibold text-ink">Rechtsgrundlagen</h3>
          <p className="mt-2 text-xs text-gray-600">
            {result.legalReferences.join(" · ")} · Engine-Version {result.engineVersion}
          </p>
        </div>

        <div className="mt-8 rounded-xl border border-primary-100 bg-primary-50 p-6">
          <h3 className="text-base font-semibold text-primary-700">
            So setzen Sie die Pflichten konkret um
          </h3>
          <p className="mt-2 text-sm text-primary-900">
            ComplAI generiert Ihnen die nötigen Dokumente (Risk Assessment, Technical Documentation,
            DPIA, Transparenzhinweise) auf Knopfdruck – inklusive AI-Literacy-Training für Ihr Team.
          </p>
          <Link
            href="/#waitlist"
            className="mt-4 inline-flex items-center rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary-700"
          >
            Auf Warteliste setzen <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>

        <div className="mt-6 flex items-center justify-between">
          <button
            type="button"
            onClick={back}
            className="text-sm text-gray-600 hover:text-ink"
          >
            ← Antworten ändern
          </button>
          <button
            type="button"
            onClick={reset}
            className="text-sm text-gray-600 hover:text-ink"
          >
            Neuen Check starten
          </button>
        </div>

        <p className="mt-6 border-t border-gray-200 pt-4 text-xs text-gray-500">
          Hinweis: Diese Klassifizierung ist eine erste Orientierung und ersetzt keine
          Rechtsberatung. Für rechtsverbindliche Auskünfte wenden Sie sich an eine spezialisierte
          Kanzlei.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-8 shadow-sm">
      {/* Progress */}
      <div className="mb-8">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>
            Schritt {step + 1} von {SECTIONS.length}
          </span>
          <span>{section.title}</span>
        </div>
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${((step + 1) / SECTIONS.length) * 100}%` }}
          />
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold text-ink">{section.title}</h2>
        <p className="mt-2 text-sm text-gray-700">{section.description}</p>
      </div>

      <ol className="mt-8 space-y-6">
        {section.questions.map((q, idx) => (
          <li key={q.id} className="rounded-xl border border-gray-200 bg-gray-50 p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-semibold text-ink">
                  {idx + 1}. {q.text}
                </p>
                {q.helper && <p className="mt-1 text-xs text-gray-600">{q.helper}</p>}
              </div>
              <span className="flex-shrink-0 rounded-full bg-white px-2 py-0.5 text-[10px] font-medium text-primary ring-1 ring-primary-100">
                {q.reference}
              </span>
            </div>

            <div className="mt-4 flex gap-2">
              {(["yes", "no", "unknown"] as const).map((option) => {
                const selected = answers[q.id] === option;
                const labels: Record<typeof option, string> = {
                  yes: "Ja",
                  no: "Nein",
                  unknown: "Weiß nicht",
                };
                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => setAnswer(q.id, option)}
                    className={`flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
                      selected
                        ? "border-primary bg-primary text-white"
                        : "border-gray-300 bg-white text-gray-700 hover:border-primary"
                    }`}
                  >
                    {labels[option]}
                  </button>
                );
              })}
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-8 flex items-center justify-between">
        <button
          type="button"
          onClick={back}
          disabled={step === 0}
          className="inline-flex items-center text-sm text-gray-600 hover:text-ink disabled:opacity-40"
        >
          <ArrowLeft className="mr-1 h-4 w-4" /> Zurück
        </button>
        <button
          type="button"
          onClick={next}
          className="inline-flex items-center rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-white hover:bg-primary-700"
        >
          {isLast ? "Auswerten" : "Weiter"} <ArrowRight className="ml-1 h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

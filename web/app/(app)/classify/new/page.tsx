import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { RiskCheckWizard } from "@/components/risk-check/wizard";

export default function NewClassificationPage() {
  return (
    <div className="mx-auto max-w-3xl">
      <Link
        href="/classify"
        className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-ink"
      >
        <ArrowLeft className="h-4 w-4" />
        Zum Classifier
      </Link>

      <h1 className="mt-4 text-3xl font-bold text-ink">Neue Klassifizierung</h1>
      <p className="mt-2 text-sm text-gray-600">
        Die gleiche Engine, die auch unser öffentliches Risiko-Check-Tool antreibt – jetzt
        direkt in Ihrer Inventur. Ergebnis wird automatisch gespeichert und an das System
        gebunden.
      </p>

      <div className="mt-8">
        <RiskCheckWizard />
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { z } from "zod";
import { createAiSystem } from "@/app/actions/inventory";

const schema = z.object({
  name: z.string().min(2, "Name zu kurz"),
  provider: z.string().min(1, "Anbieter erforderlich"),
  department: z.string().min(1, "Abteilung erforderlich"),
  deploymentType: z.enum(["saas", "api", "onprem"]),
  purpose: z.string().min(10, "Bitte mindestens 10 Zeichen"),
});

type FormState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "error"; message: string };

export function NewSystemForm() {
  const router = useRouter();
  const [state, setState] = useState<FormState>({ kind: "idle" });
  const [errors, setErrors] = useState<Record<string, string>>({});

  async function onSubmit(formData: FormData) {
    setErrors({});
    const parsed = schema.safeParse({
      name: formData.get("name"),
      provider: formData.get("provider"),
      department: formData.get("department"),
      deploymentType: formData.get("deploymentType"),
      purpose: formData.get("purpose"),
    });

    if (!parsed.success) {
      const next: Record<string, string> = {};
      for (const issue of parsed.error.issues) {
        next[String(issue.path[0])] = issue.message;
      }
      setErrors(next);
      return;
    }

    setState({ kind: "submitting" });
    try {
      const result = await createAiSystem(parsed.data);
      if (result.ok) {
        router.push(`/inventory/${result.id}`);
      } else {
        setState({ kind: "error", message: result.error });
      }
    } catch (e) {
      setState({
        kind: "error",
        message: "Unerwarteter Fehler. Bitte erneut versuchen.",
      });
    }
  }

  return (
    <form action={onSubmit} className="space-y-5">
      <Field label="Name des Systems" error={errors.name}>
        <input
          type="text"
          name="name"
          required
          placeholder="z.B. ChatGPT Enterprise"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary focus:ring-1 focus:ring-primary"
        />
      </Field>

      <Field label="Anbieter" error={errors.provider}>
        <input
          type="text"
          name="provider"
          required
          placeholder="z.B. OpenAI"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary focus:ring-1 focus:ring-primary"
        />
      </Field>

      <Field label="Verantwortliche Abteilung" error={errors.department}>
        <input
          type="text"
          name="department"
          required
          placeholder="z.B. Marketing"
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary focus:ring-1 focus:ring-primary"
        />
      </Field>

      <Field label="Deployment-Typ" error={errors.deploymentType}>
        <select
          name="deploymentType"
          defaultValue="saas"
          className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm focus:border-primary focus:ring-1 focus:ring-primary"
        >
          <option value="saas">Cloud / SaaS</option>
          <option value="api">API-Integration</option>
          <option value="onprem">On-Premises / Selbst gehostet</option>
        </select>
      </Field>

      <Field
        label="Zweck des Systems"
        error={errors.purpose}
        hint="Kurze Beschreibung, wozu das System eingesetzt wird."
      >
        <textarea
          name="purpose"
          required
          rows={3}
          placeholder="z.B. Erstellt Marketing-Texte für Produkte und Kampagnen."
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary focus:ring-1 focus:ring-primary"
        />
      </Field>

      <div className="flex items-center justify-between border-t border-gray-200 pt-5">
        <button
          type="submit"
          disabled={state.kind === "submitting"}
          className="rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-white hover:bg-primary-700 disabled:opacity-60"
        >
          {state.kind === "submitting" ? "Wird gespeichert..." : "System anlegen"}
        </button>
        {state.kind === "error" && (
          <p className="text-xs text-accent-crimson">{state.message}</p>
        )}
      </div>
    </form>
  );
}

function Field({
  label,
  error,
  hint,
  children,
}: {
  label: string;
  error?: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium text-ink">{label}</span>
      {children}
      {hint && !error && <p className="mt-1 text-xs text-gray-500">{hint}</p>}
      {error && <p className="mt-1 text-xs text-accent-crimson">{error}</p>}
    </label>
  );
}

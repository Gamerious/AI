"use client";

import { useState } from "react";
import { z } from "zod";
import { joinWaitlist } from "@/app/actions/waitlist";

const schema = z.object({
  email: z.string().email("Bitte gib eine gültige Email-Adresse ein."),
  company: z.string().min(2, "Firmenname zu kurz.").max(120),
  size: z.enum(["1-49", "50-249", "250-999", "1000+"]),
});

type FormValues = z.infer<typeof schema>;
type FormState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "success" }
  | { kind: "error"; message: string };

export function WaitlistForm() {
  const [state, setState] = useState<FormState>({ kind: "idle" });
  const [errors, setErrors] = useState<Partial<Record<keyof FormValues, string>>>({});

  async function onSubmit(formData: FormData) {
    setErrors({});
    const raw = {
      email: String(formData.get("email") ?? ""),
      company: String(formData.get("company") ?? ""),
      size: String(formData.get("size") ?? "") as FormValues["size"],
    };
    const parsed = schema.safeParse(raw);
    if (!parsed.success) {
      const next: Partial<Record<keyof FormValues, string>> = {};
      for (const issue of parsed.error.issues) {
        next[issue.path[0] as keyof FormValues] = issue.message;
      }
      setErrors(next);
      return;
    }

    setState({ kind: "submitting" });
    try {
      const result = await joinWaitlist(parsed.data);
      if (result.ok) {
        setState({ kind: "success" });
      } else {
        setState({ kind: "error", message: result.error });
      }
    } catch (e) {
      setState({
        kind: "error",
        message: "Unerwarteter Fehler. Bitte versuche es später noch einmal.",
      });
    }
  }

  if (state.kind === "success") {
    return (
      <div className="rounded-2xl border border-accent-leaf/30 bg-accent-leaf/10 p-8 text-center text-white">
        <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-accent-leaf/20 text-accent-leaf">
          ✓
        </div>
        <h3 className="text-lg font-semibold">Sie stehen auf der Liste.</h3>
        <p className="mt-2 text-sm text-gray-300">
          Wir melden uns mit Ihrem Early-Access-Link, sobald wir live sind.
          Im Mai 2026 ist es soweit.
        </p>
      </div>
    );
  }

  return (
    <form action={onSubmit} className="space-y-3">
      <div>
        <label htmlFor="email" className="sr-only">
          Geschäfts-Email
        </label>
        <input
          id="email"
          name="email"
          type="email"
          required
          autoComplete="email"
          placeholder="ihre.email@firma.de"
          className="block w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-sm text-white placeholder-gray-500 shadow-sm ring-primary focus:border-primary focus:ring-1"
        />
        {errors.email && <p className="mt-1 text-xs text-accent-citrus">{errors.email}</p>}
      </div>

      <div>
        <label htmlFor="company" className="sr-only">
          Firma
        </label>
        <input
          id="company"
          name="company"
          type="text"
          required
          placeholder="Firmenname"
          className="block w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-sm text-white placeholder-gray-500 shadow-sm focus:border-primary focus:ring-1 focus:ring-primary"
        />
        {errors.company && <p className="mt-1 text-xs text-accent-citrus">{errors.company}</p>}
      </div>

      <div>
        <label htmlFor="size" className="sr-only">
          Firmengröße
        </label>
        <select
          id="size"
          name="size"
          defaultValue=""
          required
          className="block w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-sm text-white shadow-sm focus:border-primary focus:ring-1 focus:ring-primary"
        >
          <option value="" disabled>
            Firmengröße wählen
          </option>
          <option value="1-49">1–49 Mitarbeiter</option>
          <option value="50-249">50–249 Mitarbeiter</option>
          <option value="250-999">250–999 Mitarbeiter</option>
          <option value="1000+">1000+ Mitarbeiter</option>
        </select>
        {errors.size && <p className="mt-1 text-xs text-accent-citrus">{errors.size}</p>}
      </div>

      <button
        type="submit"
        disabled={state.kind === "submitting"}
        className="block w-full rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-primary-700 disabled:opacity-60"
      >
        {state.kind === "submitting" ? "Wird gesendet..." : "Auf Warteliste setzen"}
      </button>

      {state.kind === "error" && (
        <p className="text-center text-xs text-accent-citrus">{state.message}</p>
      )}
    </form>
  );
}

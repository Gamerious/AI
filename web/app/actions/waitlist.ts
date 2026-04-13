"use server";

import { z } from "zod";
import { db } from "@/lib/db";
import { waitlist } from "@/db/schema";

const inputSchema = z.object({
  email: z.string().email(),
  company: z.string().min(2).max(120),
  size: z.enum(["1-49", "50-249", "250-999", "1000+"]),
});

export type JoinWaitlistInput = z.infer<typeof inputSchema>;

export type JoinWaitlistResult =
  | { ok: true }
  | { ok: false; error: string };

export async function joinWaitlist(input: JoinWaitlistInput): Promise<JoinWaitlistResult> {
  const parsed = inputSchema.safeParse(input);
  if (!parsed.success) {
    return { ok: false, error: "Eingaben unvollständig oder ungültig." };
  }

  try {
    await db.insert(waitlist).values({
      email: parsed.data.email.toLowerCase(),
      company: parsed.data.company.trim(),
      companySize: parsed.data.size,
      source: "landing-page",
    });
    return { ok: true };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unbekannter Fehler";
    if (message.includes("duplicate") || message.includes("unique")) {
      // Idempotent: bereits eingetragen ist auch ok.
      return { ok: true };
    }
    console.error("[waitlist] insert failed:", message);
    return { ok: false, error: "Konnte Eintrag nicht speichern. Bitte versuche es erneut." };
  }
}

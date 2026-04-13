"use server";

import { z } from "zod";
import { db } from "@/lib/db";
import { aiSystems } from "@/db/schema";

const inputSchema = z.object({
  name: z.string().min(2).max(200),
  provider: z.string().min(1).max(200),
  department: z.string().min(1).max(100),
  deploymentType: z.enum(["saas", "api", "onprem"]),
  purpose: z.string().min(10),
});

export type CreateSystemInput = z.infer<typeof inputSchema>;

export type CreateSystemResult =
  | { ok: true; id: string }
  | { ok: false; error: string };

// NOTE: Für das Scaffold ist `orgId` hart kodiert. In Produktion aus Auth-Session laden.
const DEMO_ORG_ID = "00000000-0000-0000-0000-000000000000";

export async function createAiSystem(input: CreateSystemInput): Promise<CreateSystemResult> {
  const parsed = inputSchema.safeParse(input);
  if (!parsed.success) {
    return { ok: false, error: "Eingaben unvollständig oder ungültig." };
  }

  try {
    const [row] = await db
      .insert(aiSystems)
      .values({
        orgId: DEMO_ORG_ID,
        name: parsed.data.name.trim(),
        provider: parsed.data.provider.trim(),
        department: parsed.data.department.trim(),
        deploymentType: parsed.data.deploymentType,
        purpose: parsed.data.purpose.trim(),
        riskClassification: "unclassified",
      })
      .returning({ id: aiSystems.id });

    return { ok: true, id: row.id };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unbekannter Fehler";
    console.error("[inventory] create failed:", message);
    return { ok: false, error: "Konnte System nicht anlegen. Bitte erneut versuchen." };
  }
}

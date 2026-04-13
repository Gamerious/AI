import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "@/db/schema";

const databaseUrl = process.env.DATABASE_URL;

if (!databaseUrl) {
  throw new Error(
    "DATABASE_URL is not set. Copy .env.example to .env.local and set your Supabase Postgres URL.",
  );
}

const client = postgres(databaseUrl, {
  prepare: false,
  max: 5,
});

export const db = drizzle(client, { schema });

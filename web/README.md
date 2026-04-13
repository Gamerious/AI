# ComplAI · Web

Next.js 15 App Router, TypeScript, Tailwind CSS v3, Drizzle ORM, Supabase.

## Schnellstart

```bash
pnpm install
cp .env.example .env.local
# .env.local befüllen
pnpm dev
```

Open `http://localhost:3000`.

## Struktur

```
web/
├── app/
│   ├── layout.tsx             # Root Layout, Inter-Font, Metadata
│   ├── page.tsx               # Landing Page
│   ├── actions/
│   │   └── waitlist.ts        # Server Action: Waitlist-Eintrag
│   ├── risiko-check/          # Öffentliches Klassifikations-Tool (Lead-Magnet)
│   ├── impressum/             # § 5 DDG
│   └── datenschutz/           # Art. 13 DSGVO
├── components/
│   ├── marketing/             # Landing Page Bausteine
│   └── risk-check/            # Wizard für öffentliches Tool
├── lib/
│   ├── classifier/
│   │   ├── engine.ts          # Klassifikations-Logik (AI Act Art. 5/6/50/51)
│   │   ├── engine.test.ts     # Sanity-Tests
│   │   └── questions.ts       # Fragebogen-Daten
│   ├── db.ts                  # Drizzle Postgres Client
│   └── utils.ts               # cn(), Datum, etc.
└── db/
    └── schema.ts              # Datenbank-Schema
```

## Classifier-Engine

Das Herzstück liegt in `lib/classifier/engine.ts`. Die Engine ist:
- **Regelbasiert** (kein LLM zur Laufzeit)
- **Versioniert** (`ENGINE_VERSION`)
- **Deterministisch**: gleiche Antworten = gleiches Ergebnis
- **Erklärbar**: jeder Klassifizierung liegt ein `rationale` und Gesetzesverweise zugrunde

Tests laufen mit:

```bash
npx tsx lib/classifier/engine.test.ts
```

## Deployment

Vercel, Region `fra1` (Frankfurt), Free Tier ausreichend für MVP-Phase.

```bash
vercel link
vercel deploy --prod
```

Environment Variables müssen in Vercel gesetzt sein (siehe `.env.example`).

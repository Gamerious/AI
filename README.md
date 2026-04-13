# ComplAI

> **EU AI Act Compliance für den deutschsprachigen Mittelstand.**
> Self-Service-Software statt Big4-Beratung.

[![Status](https://img.shields.io/badge/status-pre--MVP-orange)]()
[![Stack](https://img.shields.io/badge/stack-Next.js%2015%20%C2%B7%20Postgres%20%C2%B7%20Drizzle-1E3A8A)]()
[![Hosting](https://img.shields.io/badge/hosting-Frankfurt%20%C2%B7%20DSGVO-10B981)]()

ComplAI ist die erste deutschsprachige Self-Service-Plattform für die Anforderungen des
**EU AI Act (Verordnung 2024/1689)**. Wir helfen Mittelständlern in Stunden, was bei Big4 Monate
und €50.000+ kostet:

- KI-Inventur über das gesamte Unternehmen
- Geführte Risikoklassifizierung nach Art. 5, 6, 50, 51
- Generierung auditfähiger Dokumente (Risk Assessment, Tech Doc, DPIA, Transparenzhinweise)
- AI-Literacy-Training und Zertifizierung nach Art. 4
- Revisionssicherer Audit Trail

---

## 📁 Repository-Struktur

```
.
├── docs/             Vollständige Unternehmens-Dokumentation
│   ├── 01-vision.md
│   ├── 02-market-analysis.md
│   ├── 03-product-spec.md
│   ├── 04-tech-stack.md
│   ├── 05-business-model.md
│   ├── 06-go-to-market.md
│   ├── 07-financial-plan.md
│   ├── 08-legal-setup.md
│   ├── 09-roadmap.md
│   └── 10-risks.md
├── brand/            Brand-Identity, Naming, Visual Identity
├── legal/            Vorlagen für AGB, AVV, Sub-Processors
├── finance/          Forecast & Modelle
├── web/              Next.js 15 App (Marketing + MVP)
│   ├── app/          App Router (Marketing, Risiko-Check, App-Bereich)
│   ├── components/   React-Komponenten
│   ├── lib/          Classifier-Engine, DB-Client, Utils
│   └── db/           Drizzle Schema & Migrations
└── README.md
```

---

## 🚀 Entwicklung starten

### Voraussetzungen
- Node.js ≥ 20
- pnpm (oder npm/yarn)
- Supabase-Account (Free Tier ausreichend)
- Vercel-Account für Deployment

### Setup

```bash
cd web
pnpm install
cp .env.example .env.local
# .env.local mit deinen Supabase-Credentials befüllen
pnpm db:push           # Schema in deine Supabase-Instanz pushen
pnpm dev               # Dev-Server auf http://localhost:3000
```

### Wichtige Skripte

```bash
pnpm dev          # Lokaler Dev-Server
pnpm build        # Production-Build
pnpm typecheck    # TypeScript-Check
pnpm lint         # Biome Linter
pnpm db:generate  # Drizzle Migrations generieren
pnpm db:push      # Schema in Supabase pushen
pnpm db:studio    # Drizzle Studio lokal
```

### Classifier-Engine testen

Die Risk-Classifier-Engine hat eingebettete Sanity-Tests:

```bash
cd web
npx tsx lib/classifier/engine.test.ts
```

---

## 🧠 Was ist hier los? Eine Lese-Reihenfolge

Wenn du das Projekt verstehen willst, lies in dieser Reihenfolge:

1. **`docs/01-vision.md`** – Was wir bauen und warum
2. **`docs/02-market-analysis.md`** – Markt, Wettbewerb, Personas
3. **`docs/03-product-spec.md`** – Was im MVP drin ist
4. **`docs/04-tech-stack.md`** – Technische Entscheidungen
5. **`docs/05-business-model.md`** – Pricing & Unit Economics
6. **`docs/06-go-to-market.md`** – Wie wir Kunden gewinnen
7. **`docs/09-roadmap.md`** – Die nächsten 6 Monate Schritt für Schritt

Operative Dokumente:
- `docs/07-financial-plan.md` – 12-Monats-Forecast
- `docs/08-legal-setup.md` – Gründung, AGB, DSGVO
- `docs/10-risks.md` – Risiko-Register

---

## 🛣️ Roadmap (Kurzform)

| Monat (2026) | Meilenstein                                    | KPI                  |
| ------------ | ---------------------------------------------- | -------------------- |
| April        | Foundation, Landing, 10 Design Partner          | 0 Kunden, 100 Leads  |
| Mai          | MVP Core: Inventory + Classifier                | 5 aktive Partner     |
| Juni         | MVP Complete, erste Paid-Conversions            | 10 Kunden, €2k MRR   |
| Juli         | Public Launch (PH, heise, t3n)                  | 18 Kunden, €3,6k MRR |
| August       | EU AI Act Enforcement Day → Marketing-Welle     | 35 Kunden, €7k MRR   |
| September    | Reseller-Pilot, Skalierungs-Setup               | 55 Kunden, €11k MRR  |

Details: [`docs/09-roadmap.md`](./docs/09-roadmap.md)

---

## 💰 Finanzielle Leitplanken

- **Bootstrap**: < €5k Anfangskapital, kein VC in Jahr 1
- **Break-Even**: erwartet in Monat 3
- **Ziel Ende Jahr 1**: 120 zahlende Kunden, ~€286k ARR
- **Margen**: > 90 % brutto

Details: [`docs/07-financial-plan.md`](./docs/07-financial-plan.md)

---

## ⚖️ Rechtliches

ComplAI ist eine **Software-Lösung**, **keine Rechtsberatung**. Wir helfen bei Strukturierung und
Dokumentation. Für rechtsverbindliche Auskünfte sollte eine spezialisierte Kanzlei eingeschaltet
werden – wir streben dazu Partnerschaften mit IT-/Datenschutzkanzleien an.

Hosting ausschließlich in der EU (Frankfurt). DSGVO-konform. Keine Tracking-Cookies.

---

## 🤝 Mitwirken

Aktuell sind wir Solo-Gründung. Wenn du Interesse hast, mitzubauen oder als Design Partner
einzusteigen: hello@complai.de.

---

## 📜 Lizenz

Alle Rechte vorbehalten · ComplAI UG (haftungsbeschränkt) i.G.
Open-Sourcing ausgewählter Komponenten (z.B. Classifier-Engine) ist für H2/2026 angedacht.

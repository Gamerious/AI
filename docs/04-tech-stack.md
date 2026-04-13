# Tech-Stack & Architektur

## 1. Entscheidungsprinzipien

1. **Kosten < €50/Monat** in der ersten Phase (Bootstrap-Budget)
2. **EU-Hosting pflicht** (DSGVO, Kunden-Vertrauen)
3. **Schnelles MVP**: Full-Stack-Framework statt selbstgebautem Backend
4. **TypeScript überall**: kein Kontext-Switch, solide Tooling
5. **No-Lock-in wo möglich**: Postgres statt DynamoDB, Standard-SQL
6. **Erprobte Technologie**: Keine frischen 0.x-Libraries in kritischen Pfaden

---

## 2. Der Stack im Überblick

| Schicht              | Technologie                         | Begründung                                                |
| -------------------- | ----------------------------------- | --------------------------------------------------------- |
| **Frontend**         | Next.js 15 (App Router) + React 19  | Server Components, SEO, schnelles Prototyping             |
| **UI-Komponenten**   | Tailwind CSS v4 + shadcn/ui         | Hochwertige UI in Stunden, voll anpassbar                 |
| **Forms**            | React Hook Form + Zod               | Type-safe Validation, funktioniert client+server          |
| **Auth**             | Auth.js (NextAuth v5)               | Email+Password, Google OAuth, kostenfrei                  |
| **Datenbank**        | Postgres via Supabase EU (Frankfurt)| Free-Tier ausreichend für MVP, EU-Hosting                 |
| **ORM**              | Drizzle ORM                         | Typsicher, leichtgewichtig, SQL-nah                       |
| **File Storage**     | Supabase Storage (EU)               | Eingebaut, DSGVO-konform                                  |
| **PDF Generation**   | @react-pdf/renderer                 | Läuft serverless, kein Browser-Overhead                   |
| **Email**            | Resend (free tier: 3.000/Monat)     | Einfach, gute Deliverability, React-Email Support         |
| **Analytics**        | Plausible (self-hosted) oder Umami  | DSGVO-konform, cookiefrei                                 |
| **Error Tracking**   | Sentry (free tier: 5k errors/Monat) | Industriestandard                                         |
| **Hosting**          | Vercel (free tier, später Pro)      | Zero-Config, Edge, schnell                                |
| **Payments**         | Stripe (Subscriptions)              | Später: Paddle als MoR wegen EU-MwSt                      |
| **Code Quality**     | Biome (lint+format)                 | Schneller als ESLint/Prettier                             |
| **Testing**          | Vitest + Playwright                 | Unit + E2E                                                |
| **CI/CD**            | GitHub Actions (free)               | Auto-Deploy, Tests, Linting                               |
| **Dokumentation**    | Mintlify (free tier) oder Docusaurus| Content-led Growth: Dokumentation ist Marketing           |

---

## 3. Kostenschätzung (monatlich, Bootstrap-Phase)

| Service              | Free Tier Sufficient? | Kosten             |
| -------------------- | --------------------- | ------------------ |
| Vercel               | Ja, bis ~100 Nutzer   | €0                 |
| Supabase             | Ja, bis 500 MB DB     | €0                 |
| Resend               | Ja, 3000 Mails/Monat  | €0                 |
| Sentry               | Ja                    | €0                 |
| Stripe               | Keine Grundgebühr     | €0 (nur Transaktionsgebühr) |
| Domain (complai.de o.ä.)|                    | €10/Jahr (~€1/Mo)  |
| Google Workspace     | Unternehmens-Email    | €6/Mo              |
| **Gesamt**           |                       | **~€7/Monat**      |

Nach Launch (Monat 2–3):
- Supabase Pro (€25/Mo) sobald Traffic wächst
- Vercel Pro (€20/Mo) sobald nötig
- Plausible (€9/Mo) oder Umami self-hosted
- Domain EU-TLDs (€20/Jahr)

**Total ab Monat 3**: ~€60/Monat – voll im Budget.

---

## 4. Architektur-Übersicht

```
┌─────────────────────────────────────────────────────┐
│                    Users (Browser)                   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│             Vercel Edge (Frankfurt)                  │
│  ┌──────────────────────────────────────────────┐   │
│  │         Next.js 15 App Router                │   │
│  │  - Marketing Pages (SSG)                     │   │
│  │  - App Pages (SSR / RSC)                     │   │
│  │  - API Routes (Server Actions)               │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┼───────────┬───────────┐
          ▼           ▼           ▼           ▼
  ┌──────────────┐ ┌──────┐ ┌─────────┐ ┌────────┐
  │   Supabase   │ │Resend│ │ Stripe  │ │ Sentry │
  │  (Postgres + │ │Email │ │Payments │ │ Errors │
  │   Storage)   │ │      │ │         │ │        │
  │  EU-Central  │ │      │ │ EU-MoR  │ │        │
  └──────────────┘ └──────┘ └─────────┘ └────────┘
```

---

## 5. Verzeichnisstruktur

```
/
├── docs/                    # Unternehmens-Dokumentation (du liest das hier)
├── web/                     # Next.js App
│   ├── app/
│   │   ├── (marketing)/     # Landing Page, Blog, Pricing
│   │   ├── (auth)/          # Login, Signup, Password Reset
│   │   ├── (app)/           # Authentifizierte App
│   │   │   ├── dashboard/
│   │   │   ├── inventory/   # Modul 1: AI Inventory
│   │   │   ├── classify/    # Modul 2: Risk Classifier
│   │   │   ├── documents/   # Modul 3: Document Generator
│   │   │   ├── literacy/    # Modul 4: AI Literacy
│   │   │   └── audit/       # Modul 5: Audit Trail
│   │   └── api/
│   ├── components/
│   ├── lib/
│   ├── db/
│   │   ├── schema.ts
│   │   └── migrations/
│   ├── content/             # MDX: Blog, Wiki
│   └── public/
├── brand/                   # Logo, Farben, Assets
├── legal/                   # AGB, Datenschutz, AVV
├── finance/                 # Excel-Modelle, Forecast
└── README.md
```

---

## 6. Deploy- und Entwicklungsworkflow

1. **Lokale Entwicklung**: `pnpm dev` mit `.env.local` gegen Supabase Staging
2. **Feature-Branches**: `feature/...` oder `fix/...` → PR in `main`
3. **Vercel Preview Deployments** automatisch pro PR
4. **Production Deploy**: Merge nach `main` → auto-deploy auf Vercel
5. **Migrations**: Drizzle Kit generiert und pushed Migrations (CI/CD)
6. **Secrets**: Vercel Env Vars + `.env.example` im Repo

---

## 7. Sicherheits- und Compliance-Überlegungen

Weil wir Compliance-Software verkaufen, müssen wir selbst vorbildlich sein:

- **EU-Hosting**: Supabase Frankfurt, Vercel EU-Regions
- **Verschlüsselung in Transit + at Rest**: TLS 1.3, AES-256 (Supabase Standard)
- **DSGVO-konformes Sign-Up**: Double-Opt-In für Newsletter, klare Cookie-Banner (oder noch besser: keine Cookies)
- **AVV (Auftragsverarbeitungsvertrag)**: Template bereit für Kundenanfragen
- **SoD (Separation of Duties)**: Admin/Editor/Viewer-Rollen
- **Audit-Log**: append-only für alle sensiblen Aktionen
- **Backup**: Supabase PITR (ab Pro)
- **Secrets**: nie im Code, immer Vercel Env
- **Dependency-Checks**: Dependabot + GitHub Security Alerts
- **ISO 27001 Self-Assessment**: Richtung Monat 6 anvisieren

---

## 8. Verworfen und warum

| Nicht gewählt      | Warum nicht                                                |
| ------------------ | ---------------------------------------------------------- |
| T3 Stack (tRPC)    | Server Actions in Next 15 sind einfacher für kleines Team  |
| Remix              | Next.js hat mehr EU-Hosting-Optionen + bessere Content-Tools |
| Astro              | Zu marketing-fokussiert, App-Teil schwerer                 |
| AWS / Terraform    | Zu komplex, zu teuer, zu langsam für Bootstrap             |
| Hetzner + VPS      | Ops-Aufwand tötet Entwicklungszeit                         |
| Python / FastAPI   | Zwei-Sprach-Stack, mehr Reibung                            |
| MongoDB            | Relational passt besser zu Compliance-Daten (Revisionen)   |
| Prisma             | Langsamer als Drizzle, DX für Edge schlechter              |
| Clerk / WorkOS     | Kosten skalieren zu früh, Vendor-Lock-in                   |

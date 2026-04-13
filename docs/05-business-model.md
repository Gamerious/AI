# Business Model & Pricing

## 1. Value Proposition

> **"Ersparen Sie sich 50.000 € Beratung und 300 Stunden Zeit – erfüllen Sie den EU AI Act in einem Nachmittag."**

Unsere Kunden zahlen uns, weil wir ihnen drei Dinge geben, die sie sonst nicht bekommen:

1. **Sicherheit**: Auditfähige Dokumente, die vor Behörden standhalten
2. **Zeit**: 10x schneller als interne Projekte oder Beratung
3. **Aktualität**: Wir updaten Templates und Entscheidungsbäume, wenn sich die Rechtslage ändert (dafür gibt es keine Alternative, nicht einmal bei Big4)

---

## 2. Pricing-Modell

### 2.1 Tiers

| Tier           | Preis/Monat   | Enthält                                                     | Ziel-Persona               |
| -------------- | ------------- | ----------------------------------------------------------- | -------------------------- |
| **Free**       | €0            | 3 KI-Systeme, Basis-Klassifizierung, 1 PDF/Monat            | Lead-Gen, Einstieg         |
| **Starter**    | €49           | 20 KI-Systeme, unlimitierte PDFs, AI Literacy (20 Mitarbeiter) | Small Biz (50–100 MA)      |
| **Pro**        | €199          | Unlimitiert, Audit Trail, Mehrere Benutzer, Prioritäts-Support | Mittelstand (100–500 MA)   |
| **Business**   | €499          | + SSO, API, AVV on demand, dedizierter CSM, Compliance Reviews | Oberer Mittelstand (500–2000) |
| **Enterprise** | ab €1.500     | + Custom Branding, On-Prem-Option, Legal-Review, SLA        | Enterprise + Reseller      |

### 2.2 Preislogik

- **Anker-Preis**: Die Business-Stufe bei €499 macht den Pro-Plan (€199) "günstig" wirken
- **Psychologisch unter Beratungsbudget**: Selbst Enterprise bei €18k/Jahr ist deutlich unter einer einzelnen Big4-Session
- **Pro ist der Sweet-Spot**: Wir optimieren auf Pro-Plan-Signups. Das ist unser wirtschaftliches Ziel.
- **Jahreszahlung**: −2 Monate Rabatt (17 %) für Vorauszahlung → verbessert Cashflow massiv
- **Keine Seat-Pricing im Free/Starter**: weniger Friction beim Einstieg

### 2.3 Add-ons

| Add-on                                    | Preis          | Beschreibung                                     |
| ----------------------------------------- | -------------- | ------------------------------------------------ |
| **Compliance-Audit durch uns**            | €1.900 einmalig| Wir reviewen alle Dokumente des Kunden in 5 Tagen |
| **Custom AI-Literacy-Training**           | €900 einmalig  | Wir erstellen rollenspezifisches Training        |
| **Extra Seats (Pro+)**                    | €15/Seat/Monat | Nur ab Pro                                       |
| **White-Label** (Reseller)                | 20 % vom Umsatz| Für DPOs/Beratungen                              |

---

## 3. Unit Economics

### 3.1 Cost of Goods Sold (COGS) pro Kunde

| Position                          | Pro Kunde/Monat |
| --------------------------------- | --------------- |
| Hosting (Vercel + Supabase Anteil)| €0,80           |
| Email (Resend Anteil)             | €0,10           |
| Zahlungsgebühr Stripe (~3 %)      | €6,00 (bei €199)|
| Support (im Schnitt 15 min/Monat) | €3,00 @ €12/h   |
| **Total COGS**                    | **~€10/Kunde**  |

### 3.2 Beispiel-Margen

| Tier     | Preis   | COGS     | Bruttomarge | Marge % |
| -------- | ------- | -------- | ----------- | ------- |
| Starter  | €49     | €6       | €43         | 88 %    |
| Pro      | €199    | €10      | €189        | 95 %    |
| Business | €499    | €20      | €479        | 96 %    |

→ Klassische SaaS-Margen. Extrem profitabel pro Kunde.

### 3.3 CAC & Payback

- **CAC (Customer Acquisition Cost)** im Bootstrap (Content + Cold Outreach): ~€50–150
- **Payback Period**: 1–3 Monate (bei Pro-Plan)
- **LTV bei 24 Monaten Retention**: €4.776 (Pro-Plan)
- **LTV/CAC Ratio**: 30–90x – traumhaft

---

## 4. Revenue Streams (Jahr 1–3)

| Stream                    | Jahr 1 | Jahr 2  | Jahr 3    |
| ------------------------- | ------ | ------- | --------- |
| Starter (Subscriptions)   | 25 %   | 20 %    | 15 %      |
| Pro (Subscriptions)       | 55 %   | 55 %    | 50 %      |
| Business (Subscriptions)  | 15 %   | 20 %    | 25 %      |
| Audit Add-ons             | 5 %    | 4 %     | 5 %       |
| Reseller-Marge (Neu Jahr 2) | 0 %  | 1 %     | 5 %       |
| **Total ARR (Ziel)**      | €80k   | €480k   | €1.140k   |

---

## 5. Customer-Lifecycle-Strategie

### 5.1 Akquise
- **Content**: SEO + LinkedIn als primäre Kanäle
- **Free Tools**: "EU AI Act Risiko-Check" (öffentlich, ohne Sign-Up) als Lead-Magnet
- **Cold Outreach**: Personalisierte LinkedIn-DMs an Compliance-Officer

### 5.2 Aktivierung
- **Onboarding**: 15-Minuten-Wizard
- **"Aha"-Moment**: Erster fertiger PDF-Report
- **14 Tage Trial** auf Pro-Features

### 5.3 Retention
- **Rechtliche Updates**: Monatlicher Newsletter "AI-Act-Update" nur für Kunden
- **Neue Templates**: jede 2 Wochen
- **Quarterly Reviews** (ab Pro) mit Compliance-Ratschlägen

### 5.4 Expansion
- **Seat-Expansion**: Wenn Kunde mehr User hinzufügt
- **Modul-Upgrade**: Starter → Pro bei Erreichen der 20-System-Grenze
- **Cross-Sell**: CSRD-Modul in Q4/2026 (EU Corporate Sustainability Reporting)

### 5.5 Churn-Minimierung
- **Annual-Billing-Rabatt** reduziert Monats-Churn
- **Switching Cost**: Je mehr Dokumente/Historie im System, desto härter der Wechsel
- **Regulatorischer Lock-in**: Solange AI Act existiert, existiert Bedarf

---

## 6. Financial Projections (zusammengefasst)

| Monat | Zahlende Kunden | MRR      | ARR      | Kommentar                             |
| ----- | --------------- | -------- | -------- | ------------------------------------- |
| 1     | 0               | €0       | €0       | Launch, Design Partner kostenfrei     |
| 3     | 10              | €1.990   | €23.880  | Erste Free → Paid Conversions         |
| 6     | 35              | €6.965   | €83.580  | Erste Business-Kunden                 |
| 9     | 70              | €13.930  | €167.160 | Content-SEO greift                    |
| 12    | 120             | €23.880  | €286.560 | Reseller-Pilot mit 2–3 DPOs           |
| 18    | 220             | €43.780  | €525.360 | EU-Expansion startet                  |
| 24    | 350             | €69.650  | €835.800 | ARR-Ziel Jahr 2                       |

Details im `finance/forecast.md`.

---

## 7. Warum dieses Modell Bootstrap-kompatibel ist

1. **Kein Seed nötig**: Ab ~10 Pro-Kunden decken wir alle laufenden Kosten
2. **Hohe Bruttomargen (90 %+)**: Jeder neue Kunde ist reiner Gewinn
3. **Vorhersehbarer Umsatz**: Jahresverträge erlauben Planung
4. **Compound Growth**: Content-SEO wird über Zeit billiger und besser
5. **Reseller-Kanal** (ab Monat 9): multipliziert ohne zusätzliche Marketing-Kosten

---

## 8. Wann wir Funding aufnehmen würden (und wann nicht)

**Nicht:**
- In den ersten 12 Monaten – wir beweisen erst PMF
- Wenn wir keine klare Skalierungsherausforderung haben

**Vielleicht:**
- Wenn wir €500k ARR erreichen und EU-weit expandieren wollen
- Wenn ein Wettbewerber eine dicke Runde raised und wir das Kapital für schnelleres Hiring brauchen
- Strategisch: CVC eines großen Beratungshauses, das uns als Distribution nutzt

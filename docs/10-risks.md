# Risiko-Register

> **Prinzip**: Wir ignorieren Risiken nicht. Für jedes identifizierte Risiko haben wir einen Frühwarnindikator und einen Mitigations-Plan.

Bewertung: **Wahrscheinlichkeit × Auswirkung** (je 1–5), Risk Score = Produkt.

---

## 1. Top-Risiken

### R1 – Gesetzliche Änderung oder Interpretations-Shift [Score: 12]

**Beschreibung**: Die Europäische Kommission, nationale Aufsichten oder Gerichte interpretieren den AI Act anders als wir. Unsere Klassifizierungen werden plötzlich ungültig.

- **Wahrscheinlichkeit**: 4/5 (sehr hoch – Act ist neu, viele Fragen offen)
- **Auswirkung**: 3/5 (wir müssen Templates updaten, Re-Klassifizierung triggern, Kunden kommunizieren)
- **Frühwarnung**: Offizielle Leitlinien (Kommissions-Guidelines), nationale Aufsichts-Opinions, EDPB-Statements
- **Mitigation**:
  - Versions-System für alle Templates und Entscheidungsbäume
  - Automatisches Re-Klassifizieren bei Template-Version-Update
  - Disclaimer: "Basierend auf Gesetzesstand YYYY-MM-DD"
  - Legal-Retainer mit spezialisierter Kanzlei (ab Monat 4)
  - Monatlicher Legal-Review der Content-Pipeline

---

### R2 – Großer Wettbewerber betritt Markt [Score: 12]

**Beschreibung**: OneTrust, ServiceNow oder ein deutscher Spieler (DataGuard, Proliance) bauen ein ähnliches Self-Service-Produkt für den Mittelstand.

- **Wahrscheinlichkeit**: 3/5 (wahrscheinlich innerhalb 12 Monaten)
- **Auswirkung**: 4/5 (großer Vertriebs-Apparat wäre hart)
- **Frühwarnung**: Job-Ausschreibungen der Wettbewerber, Pressemitteilungen, PitchBook-Monitoring
- **Mitigation**:
  - **Nischen-Vertiefung**: wir bleiben klein, schnell, deutschsprachig; Enterprise-GRCs sind zu langsam
  - **Content-Moat**: Wir sind SEO-Lead für "EU AI Act" im DACH – Lead-Flow-Vorsprung
  - **Community-Building**: Kunden-Community als Switching-Cost
  - **Integration statt Konkurrenz**: Wenn OneTrust groß wird, werden wir Partner statt Gegner

---

### R3 – Produkt ist "falsch" / nicht vertriebsreif zum Enforcement-Event [Score: 15]

**Beschreibung**: Wir verpassen den Launch vor August 2026. Das Produkt ist nicht stabil oder das falsche Produkt.

- **Wahrscheinlichkeit**: 3/5 (realistisch bei MVP-Druck)
- **Auswirkung**: 5/5 (verpassen des Momentums)
- **Frühwarnung**: Sprint-Velocity, Design-Partner-Aktivierungsrate, Aha-Moment-Zeit
- **Mitigation**:
  - MVP-Scope aggressiv schneiden (lieber weniger Features, besser)
  - Design Partner ab Woche 6 testen lassen (nicht erst Woche 12)
  - Wöchentliche Friction-Logs mit allen Tests
  - Fallback-Plan: Wenn Produkt nicht fertig, Landing-Page-Version mit "Service-angebot" (wir machen Analyse manuell für €800/einmalig)

---

### R4 – Rechtliche Haftung für falsche Klassifizierung [Score: 10]

**Beschreibung**: Kunde wird mit einer auf unserem Ergebnis basierten Klassifizierung abgestraft und verklagt uns.

- **Wahrscheinlichkeit**: 2/5
- **Auswirkung**: 5/5 (existenzbedrohend)
- **Frühwarnung**: Erste nationale AI-Act-Bußgelder in der Presse
- **Mitigation**:
  - **Klare Haftungsbeschränkung in AGB**: "Tool unterstützt bei der Dokumentation, keine Rechtsberatung"
  - **Disclaimer im Produkt**: auf jeder Classification-Result-Seite
  - **Vermögensschadenhaftpflicht** ab Launch (€3 Mio. Deckung)
  - **UG-Haftungsbeschränkung** als letzter Schirm
  - **Peer-Review** durch externe Kanzlei für Template-Changes

---

### R5 – Konzentrationsrisiko einzelner Design Partner [Score: 8]

**Beschreibung**: 3–4 Design Partner geben widersprüchliches Feedback, wir bauen Feature-Creep statt PMF.

- **Wahrscheinlichkeit**: 4/5
- **Auswirkung**: 2/5
- **Frühwarnung**: Feature Requests, die nur 1 Partner will
- **Mitigation**:
  - "3-Kunden-Regel": neue Features nur bauen, wenn 3+ Kunden dasselbe verlangen
  - Persona-basierte Entscheidungsprotokolle
  - Product-Vision-Check vor jedem Sprint-Planning

---

### R6 – Gründer-Burnout / Ein-Personen-Risiko [Score: 9]

**Beschreibung**: Wir sind solo. Krankheit, persönliche Krisen, Erschöpfung stoppen das Unternehmen.

- **Wahrscheinlichkeit**: 3/5
- **Auswirkung**: 3/5
- **Frühwarnung**: Schlafqualität, Output-Rate, emotionale Zähigkeit
- **Mitigation**:
  - Ehrliche wöchentliche Selbstchecks
  - Jeden Sonntag frei (nicht-verhandelbar)
  - Keine Telefonate nach 19:00
  - Ab Monat 6: erster Freelancer – nicht erst, wenn es brennt
  - Notfallplan: "Wer übernimmt das Unternehmen für 2 Wochen, wenn ich krank werde?" (dokumentiert)

---

### R7 – Datenpanne oder Security Incident [Score: 10]

**Beschreibung**: Kunden-Daten werden geleakt oder kompromittiert. Als Compliance-Anbieter wäre das existenzbedrohend.

- **Wahrscheinlichkeit**: 2/5
- **Auswirkung**: 5/5
- **Frühwarnung**: Dependabot Alerts, Sentry Errors, gescheiterte Login-Versuche
- **Mitigation**:
  - Supabase RLS (Row Level Security) für alle Tabellen
  - Secrets nur in Vercel Env
  - Regelmäßige Dependency-Updates
  - MFA für alle Admin-Zugänge
  - Cyber-Versicherung ab Kunde #5
  - Incident-Response-Plan dokumentiert
  - Pen-Test vor €100k ARR

---

### R8 – Schlechter Name / Markenkonflikt [Score: 4]

**Beschreibung**: "ComplAI" ist bereits geschützt oder wirkt unseriös.

- **Wahrscheinlichkeit**: 2/5
- **Auswirkung**: 2/5
- **Mitigation**:
  - Vor MVP-Launch: DPMA- & EUIPO-Recherche
  - 3 Backup-Namen in Reserve (siehe `brand/naming.md`)
  - Domain-Portfolio reservieren (.de, .com, .eu, .io)

---

### R9 – Zahlungsausfälle / Mahnwesen [Score: 4]

**Beschreibung**: Kunden zahlen nicht, wir verbringen Zeit mit Mahnungen.

- **Wahrscheinlichkeit**: 2/5
- **Auswirkung**: 2/5
- **Mitigation**:
  - Stripe auto-dunning (3 Retry-Versuche)
  - Kreditkartenpflicht bei Subscription
  - Automatische Sperrung bei zweitem fehlgeschlagenen Monat
  - Keine Rechnungen in Jahr 1 (nur Kartenzahlung)

---

### R10 – Falsches Marktsegment (Mittelstand zahlt nicht) [Score: 8]

**Beschreibung**: Unser Pricing passt nicht, Mittelstand kauft weniger als erwartet.

- **Wahrscheinlichkeit**: 2/5
- **Auswirkung**: 4/5
- **Frühwarnung**: Conversion-Rates < 3 % im Outreach, sehr viele Rabatt-Anfragen
- **Mitigation**:
  - A/B-Test der Preise mit ersten 30 Kunden
  - Möglichkeit zum Pivot nach oben (Business/Enterprise) oder nach unten (kleinere Firmen, DPO-Reseller)
  - Sofortiges Preisexperiment bei unter 10 % Conversion

---

## 2. Risiko-Matrix

```
           Auswirkung →
         1    2    3    4    5
         ┌────┬────┬────┬────┬────┐
       1 │    │    │    │    │    │
         ├────┼────┼────┼────┼────┤
       2 │    │ R8 │    │R10 │R7  │
         │    │ R9 │    │    │    │
         ├────┼────┼────┼────┼────┤
       3 │    │    │ R6 │R2  │R3  │
         │    │    │    │    │    │
         ├────┼────┼────┼────┼────┤
       4 │    │ R5 │ R1 │    │    │
         │    │    │    │    │    │
         ├────┼────┼────┼────┼────┤
       5 │    │    │    │    │    │
         └────┴────┴────┴────┴────┘
W ↑
```

---

## 3. Monitoring-Rhythmus

- **Wöchentlich**: R3, R5, R6 (kurzfristig, operativ)
- **Monatlich**: R1, R2, R4, R7, R10 (strategisch)
- **Quartalsweise**: R8, R9 (niedrig, Background)

Review erfolgt im monatlichen Retro. Jedes Risiko wird dort neu bewertet und Mitigations-Fortschritt festgehalten.

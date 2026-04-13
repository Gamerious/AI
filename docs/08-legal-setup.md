# Rechtliches & Gründungs-Setup

> **Disclaimer**: Dieses Dokument ist kein Rechtsberatung. Vor finaler Umsetzung ist mindestens eine Stunde mit einem Steuerberater und einer auf IT-Recht spezialisierten Kanzlei Pflicht. Wir bauen eine Compliance-Firma – wir müssen selbst vorbildlich sein.

---

## 1. Gesellschaftsform

### 1.1 Empfehlung: **UG (haftungsbeschränkt)** ("Mini-GmbH")

**Warum**:
- **Mindeststammkapital €1** (wir nehmen €500 für Sicherheitspuffer)
- **Haftungsbeschränkung** wie bei einer GmbH
- **Umwandlung in GmbH** möglich, sobald Eigenkapital €25.000 erreicht (aus thesaurierten Gewinnen)
- **Vertrauenswürdig bei B2B-Kunden** (mehr als Einzelunternehmen)
- **Gründungskosten gering**: ~€500–800 inkl. Notar und HR-Eintragung (mit Musterprotokoll)

### 1.2 Alternativen (verworfen)

| Form             | Warum nicht                                                      |
| ---------------- | ---------------------------------------------------------------- |
| Einzelunternehmen| Volle Privathaftung – bei Compliance-SaaS zu riskant             |
| GbR              | Volle Privathaftung, wenig Vertrauen bei B2B                     |
| GmbH             | €25k Stammkapital nötig – sprengt Bootstrap-Budget               |
| AG               | Viel zu komplex und teuer                                        |
| gUG              | Nicht passend für gewinnorientiertes SaaS                        |

### 1.3 Schritte zur Gründung (DE)

1. **Name prüfen** (IHK + Handelsregister + DPMA-Markensuche) – **1 Tag**
2. **Gesellschaftsvertrag** (Musterprotokoll verwendbar bis zu 3 Gesellschafter) – **1 Tag**
3. **Notartermin** (Beurkundung + Handelsregister-Anmeldung) – **1 Tag, ~€150**
4. **Stammkapital einzahlen** (Geschäftskonto bei z.B. Qonto, Kontist, Holvi) – **1–3 Tage**
5. **Handelsregister-Eintragung** – **1–4 Wochen**
6. **Gewerbeanmeldung** bei Gemeinde – **1 Tag, ~€30**
7. **Steuerliche Erfassung** (Elster) – **1–3 Wochen**
8. **Umsatzsteuer-ID beantragen** (wenn EU-Umsätze geplant) – **1–2 Wochen**
9. **IHK-Zwangsmitgliedschaft** (automatisch, Jahr 1 ggf. ermäßigt)

**Gesamtdauer**: ca. 4–6 Wochen
**Gesamtkosten**: ca. €800–€1.000 inkl. Stammkapital

### 1.4 Alternative für schnellen Start: **Vorratsgesellschaft**

- Bestehende UG/GmbH kaufen (Vorratsgesellschaft), sofort einsatzbereit
- Kosten: €2.000–€4.000
- Vorteil: Start binnen einer Woche
- Nachteil: teurer

**Empfehlung**: Selbst gründen, es sei denn, wir brauchen *sofort* Handelsregister für einen konkreten Deal.

---

## 2. Geschäftskonto

### Empfehlungen (2026)
- **Qonto**: €9–29/Monat, gute API, Rechnungs-Tool, DE-IBAN
- **Kontist**: €9–49/Monat, Steuerschätzer, Lexoffice-Integration
- **Finom**: sehr günstig, gute Starter-Option
- **Holvi**: Jahresabschluss-Features, €9/Monat
- **Teilauto N26 Business**: nur Freiberufler, nicht UG-geeignet

**Empfehlung**: **Qonto Solo** (€9/Monat) für Start, später Business.

---

## 3. Buchhaltung

### Tools
- **lexoffice Rechnung&Finanzen**: €12,90/Monat – reicht für Solo-UG
- **sevdesk**: €9/Monat Basic – gut für kleine Firmen
- **DATEV Unternehmen online**: ab €13/Monat – wenn Steuerberater DATEV nutzt

**Empfehlung**: Lexoffice (intuitiv, moderne UI).

### Steuerberater
- **Muss** eingeschaltet werden spätestens für Jahresabschluss Ende 2026
- Budget: ca. €500–1.500/Jahr für kleine UG
- Suche: lokaler StB mit SaaS/Startup-Erfahrung

---

## 4. Rechtliche Dokumente für die Website

### Pflicht-Dokumente bei Launch

1. **Impressum** (§ 5 DDG) – Pflicht
2. **Datenschutzerklärung** (Art. 13 DSGVO + TTDSG) – Pflicht
3. **AGB** (Allgemeine Geschäftsbedingungen) – nicht gesetzlich Pflicht, aber bei SaaS unerlässlich
4. **Auftragsverarbeitungsvertrag (AVV)** (Art. 28 DSGVO) – Pflicht wegen Verarbeitung personenbezogener Daten der Kunden
5. **Cookie-Banner** – nur wenn Cookies gesetzt werden (wir nutzen Plausible/Umami → **keine Cookies nötig → kein Banner!** Wettbewerbsvorteil.)
6. **Widerrufsbelehrung** – nein, B2B gilt nicht
7. **Kündigungsbutton** (für B2C-Verträge) – nein, B2B

### Wo bekomme ich die Texte?
- **e-Recht24 Premium**: ~€130/Jahr, generiert alles DSGVO-konform
- **trustedshops Legal Texts**: zuverlässig
- **Spezialisierte Kanzlei**: €800–1.500 für vollständiges Paket (empfohlen)

**Empfehlung**: Mit e-Recht24 starten, nach Erreichen €50k MRR auf Kanzlei-Paket upgraden.

---

## 5. Markenschutz

### ComplAI als Marke anmelden
- **DPMA** (Deutsches Patent- und Markenamt): €290 für 3 Klassen Wortmarke
- **EUIPO** (EU-weit): €850
- **Empfehlung**: DPMA zuerst (DACH-Fokus), dann EUIPO nach €100k ARR

**Wichtig**: Vor Anmeldung Rechercheprüfung mit Markenanwalt (oder selbst mit DPMAregister/tmview) – wir dürfen nicht in Konflikt mit bestehenden Marken geraten.

**Vorsicht**: "Compl" + AI ist potenziell schon belegt. Wir müssen vor Launch prüfen und notfalls auf alternativen Namen ausweichen (siehe `brand/naming.md`).

---

## 6. DSGVO & interner Datenschutz

### Pflichten, die UNS betreffen (als Verarbeiter UND Verantwortlicher)

1. **Verzeichnis von Verarbeitungstätigkeiten (VVT)** (Art. 30 DSGVO) – Pflicht
2. **Technisch-organisatorische Maßnahmen (TOMs)** dokumentieren
3. **Datenschutzbeauftragter (DSB)**?:
   - Intern: bei >20 Personen mit regelmäßiger Datenverarbeitung (wir nicht im Jahr 1)
   - Extern: sobald wir sensible Daten verarbeiten → **freiwillig ratsam**, aber nicht Pflicht
4. **Meldung an Aufsichtsbehörde (LfDI BW o.ä.)** bei Datenpannen – Prozess dokumentieren
5. **Betroffenenrechte** (Auskunft, Löschung, Übertragbarkeit) – Prozess dokumentieren

### Unterauftragsverarbeiter (Sub-Processors)

Wir dokumentieren, wen wir für die Datenverarbeitung einsetzen:

| Sub-Processor  | Zweck                        | Land                   |
| -------------- | ---------------------------- | ---------------------- |
| Supabase Inc.  | Datenbank + Storage          | AWS Frankfurt (EU)     |
| Vercel Inc.    | Hosting                      | EU-Regionen            |
| Stripe Inc.    | Payments                     | DE/IE (MOR-Option)     |
| Resend Inc.    | Transactional Emails         | AWS (Region wählbar)   |
| Sentry GmbH    | Error Tracking               | DE oder US wählbar     |

Alle Sub-Processors brauchen AVV + Standardvertragsklauseln, wo USA involviert. Wir listen sie öffentlich in `/legal/subprocessors`.

---

## 7. Versicherungen

| Versicherung                      | Wann            | Jahresbeitrag |
| --------------------------------- | --------------- | ------------- |
| Berufshaftpflicht/Vermögensschaden| ab Launch       | €300–600      |
| Cyber-Versicherung                | ab ~10 Kunden   | €400–800      |
| D&O-Versicherung                  | ab €250k ARR    | €800+         |
| Betriebshaftpflicht               | ab Launch       | €200–300      |
| Rechtsschutz (Firmen)             | ab €100k ARR    | €400–600      |

**Empfehlung**: **exali.de** oder **hiscox.de** für günstige SaaS-Versicherungen.

---

## 8. Arbeitsrechtlich (für später)

Sobald wir ersten Mitarbeiter/Freelancer einsetzen:
- Scheinselbstständigkeit vermeiden (max. 5/6 Auftraggeber-Anteil, klare Projektverträge)
- Werkverträge vs. Dienstverträge unterscheiden
- Minijob-Zentrale für Teilzeit
- SFirm-/ELStAM-Meldungen bei Festanstellung

**Jahr 1**: vermutlich reiner Freelancer-Einsatz → einfach.

---

## 9. Checklist für Launch (rechtlich)

- [ ] UG gegründet & im Handelsregister eingetragen
- [ ] Geschäftskonto eröffnet (Qonto)
- [ ] Buchhaltung eingerichtet (Lexoffice)
- [ ] Impressum + Datenschutzerklärung + AGB + AVV auf Website
- [ ] VVT angelegt
- [ ] TOMs dokumentiert
- [ ] Sub-Processors gelistet
- [ ] Vermögensschadenhaftpflicht abgeschlossen
- [ ] Cyberversicherung recherchiert (Abschluss ab Kunde #5)
- [ ] Markenrecherche + ggf. Anmeldung vorbereitet
- [ ] Lizenzprüfung: dürfen wir Paragrafen aus dem AI Act zitieren? (Ja, amtliche Werke sind frei, § 5 UrhG)
- [ ] Hinweis auf "kein Rechtsberatung" in Produkt und Dokumenten

# Produkt-Spezifikation – ComplAI MVP

## 1. Produkt-Leitplanken

- **Time-to-First-Value < 15 Minuten**: Neue Nutzer müssen in unter 15 Minuten eine vollständige erste Risikoklassifizierung abgeschlossen haben
- **Self-Service von Tag 1**: Kein Sales-Call nötig, um Wert zu sehen
- **"Done-for-you"-Dokumente**: Nach 3 Klicks hat der Kunde ein PDF, das sie einem Auditor vorlegen können
- **DSGVO-konform by design**: EU-Hosting, Datenverschlüsselung, DSB-ready
- **Schrittweise Verbesserung**: Kunden können unvollständige Inventuren speichern und über Wochen erweitern

---

## 2. MVP – 5 Kernmodule

### Modul 1: AI Inventory (KI-System-Register)

**Zweck**: Zentraler Katalog aller im Unternehmen eingesetzten KI-Systeme.

**Features (MVP)**:
- Manuelles Anlegen von KI-Systemen über Formular
- Import über CSV (für bestehende Excel-Listen)
- Vordefinierte Templates für häufige Systeme (ChatGPT, GitHub Copilot, Microsoft Copilot, Salesforce Einstein, DeepL, Jasper, Midjourney, eigene Modelle)
- Felder pro System: Name, Anbieter, Version, Zweck, Abteilung, verantwortliche Person, verarbeitete Datenkategorien, Deployment-Typ (API/On-Prem/SaaS), Training-Daten
- Tag-System (z.B. "HR", "Produktiv", "Pilot")
- Suchfunktion + Filter

**Post-MVP**:
- Automatische Discovery via Browser-Extension (scanned SaaS-Tools, die Mitarbeiter nutzen)
- Integration mit SSO-Provider → welche AI-Tools sind angebunden
- API-Scanner (Network Monitoring Light)

---

### Modul 2: Risk Classifier (Risiko-Klassifizierungs-Wizard)

**Zweck**: Geführter Fragebogen, der jedes KI-System nach AI Act (Art. 5, 6, 50) klassifiziert.

**Klassifikations-Ergebnisse**:
1. **Prohibited** (Art. 5) – Einsatz verboten, Deadline Feb 2025 bereits verstrichen
2. **High-Risk** (Art. 6 + Anhang III) – strenge Pflichten: Tech Docs, Risk Management, Monitoring, Human Oversight, Conformity Assessment
3. **Limited Risk** (Art. 50) – Transparenzpflichten (z.B. Chatbots müssen sich zu erkennen geben, Deepfakes müssen gekennzeichnet sein)
4. **Minimal Risk** – keine spezifischen Pflichten, Best Practices empfohlen
5. **General Purpose AI (GPAI)** (Art. 51 ff.) – separate Regel-Kaskade

**Fragebogen-Logik**:
- Entscheidungsbaum mit ~15–25 Fragen je System
- Jede Frage mit Legal-Reference zum Gesetzestext
- Ergebnis erklärt: **Welche Pflichten folgen? Welche Dokumente müssen erstellt werden? Welche Deadlines gelten?**

**Output**: Klassifizierungs-Protokoll als PDF (auditfähig, mit Timestamp + verantwortlicher Person)

**Technisch**: 
- JSON-basierte Entscheidungsbaum-Engine
- Versioniert – wenn Gesetz sich ändert (guidance notes), re-triggert Re-Klassifizierung

---

### Modul 3: Document Generator (Dokument-Generator)

**Zweck**: Generiert alle Pflichtdokumente nach Klassifizierung – befüllt mit den Inventur-Daten.

**Dokument-Templates (MVP)**:

| Dokument                                      | AI-Act-Referenz        | Pflicht bei     |
| --------------------------------------------- | ---------------------- | --------------- |
| Technical Documentation                       | Art. 11, Annex IV      | High-Risk       |
| Risk Management System                        | Art. 9                 | High-Risk       |
| Data Governance Documentation                 | Art. 10                | High-Risk       |
| Transparency Notice (Chatbot-Disclaimer)      | Art. 50                | Limited Risk    |
| Human Oversight Plan                          | Art. 14                | High-Risk       |
| Post-Market Monitoring Plan                   | Art. 72                | High-Risk       |
| AI Literacy Training Record                   | Art. 4                 | ALLE            |
| DPIA-Addendum (AI-spezifisch, DSGVO-Link)     | DSGVO Art. 35 + AI Act | Wenn personenbezogen |
| Provider Declaration of Conformity            | Art. 47                | High-Risk Provider |

**Ausgabeformat**: PDF + DOCX (zur Weiterbearbeitung)
**Sprache**: Deutsch zuerst, Englisch in Sprint 3

**Technisch**:
- Handlebars-ähnliche Template-Engine
- Templates in `/templates/*.md` mit Variablen
- Markdown → PDF über `puppeteer` + Custom CSS

---

### Modul 4: AI Literacy Tracker (Schulungs-Tracking)

**Zweck**: Art. 4 verpflichtet Unternehmen, dass **alle** Mitarbeiter, die mit AI arbeiten, über "ausreichende AI-Kompetenz" verfügen. Das ist bereits **seit Februar 2025 in Kraft** – unerfüllt in ~95 % der Unternehmen.

**Features (MVP)**:
- Import von Mitarbeiterliste (CSV oder manuell)
- Vordefiniertes, von uns erstelltes Online-Training: "EU AI Act Basics" (20 min, Deutsch)
- Kurze Wissens-Checks (Multiple Choice)
- Zertifikat als PDF nach Abschluss (mit Timestamp + Namen)
- Dashboard: "82 % Ihrer Mitarbeiter sind AI-Literacy-zertifiziert"
- Reminder-Emails an Mitarbeiter, die noch nicht abgeschlossen haben

**Post-MVP**:
- Rollenspezifische Trainings (Entwickler, HR, Marketing)
- SCORM-Export für bestehende LMS-Systeme
- Eigene Trainings des Kunden integrieren

---

### Modul 5: Audit Trail & Dashboard

**Zweck**: Revisionssichere Historie aller Compliance-Aktivitäten + Statusübersicht.

**Features**:
- **Compliance-Score**: % der Inventur-Systeme korrekt klassifiziert + dokumentiert
- **Aktivitäten-Log**: Wer hat wann was geändert? (append-only)
- **Ablauf-Warnungen**: "Dokument XYZ ist älter als 12 Monate und muss aktualisiert werden"
- **Export**: PDF-Report für Vorstand / Auditor
- **Incident-Register**: Meldung und Dokumentation ernster KI-Vorfälle (Art. 73)

---

## 3. Was NICHT im MVP ist (bewusst ausgelassen)

- ❌ SSO / SCIM Provisioning (nur Email + Password, Google Login)
- ❌ SOC 2 / ISO 27001 Zertifizierung (kommt in Monat 6)
- ❌ Mehrsprachigkeit außer DE/EN
- ❌ White-Label / Reseller Portal
- ❌ Mobile App
- ❌ Custom Branding pro Kunde
- ❌ Workflow-Automation / Genehmigungsketten
- ❌ Integrationen (Slack, Teams, Jira, Confluence)
- ❌ Eigene AI-Features im Produkt (Ironie vermeiden: wir verkaufen Compliance, nicht noch mehr AI-Risiko)

Diese Features kommen, aber nur wenn Kunden sie aktiv verlangen.

---

## 4. User Journey (MVP, Happy Path)

1. **Landing Page** → "Kostenlos starten, keine Kreditkarte"
2. **Sign Up** (Email + Passwort oder Google)
3. **Onboarding Wizard** (5 Schritte):
   - Unternehmens-Daten (Name, Größe, Branche, HQ-Land)
   - "Welche bekannten KI-Tools setzen Sie ein?" (Checkbox-Liste mit Logos)
   - → Automatische Vor-Erstellung von 3–10 AI-Systemen in der Inventur
4. **Risk Classifier** für erstes System starten → 5 Min → Klassifikationsergebnis
5. **Dokument erstellen** → PDF-Download
6. **Dashboard** zeigt Compliance-Score: "2 von 10 Systemen sind dokumentiert – 20 %"
7. **Upgrade-CTA** nach X Aktionen oder nach 14 Tagen Trial

**Gamification**: Progress-Bar, Onboarding-Checkliste, Compliance-Score als "XP"

---

## 5. Datenmodell (MVP)

```
Organization
  - id, name, industry, size, country, created_at
  - users[] (User)
  - ai_systems[] (AISystem)
  - documents[] (Document)
  - trainings[] (TrainingRecord)
  - audit_events[] (AuditEvent)

User
  - id, email, name, role (admin/editor/viewer), org_id

AISystem
  - id, org_id, name, provider, version, purpose, department,
  - responsible_user_id, data_categories[], deployment_type,
  - risk_classification (enum), classified_at, classified_by

ClassificationSession
  - id, ai_system_id, answers (json), result, rationale (text),
  - performed_by, performed_at

Document
  - id, org_id, ai_system_id, template_key, version,
  - generated_at, generated_by, file_url

TrainingRecord
  - id, org_id, employee_name, employee_email, course_key,
  - completed_at, score, certificate_url

AuditEvent
  - id, org_id, user_id, entity_type, entity_id, action,
  - before, after, created_at (append-only)
```

---

## 6. Definition of Done (MVP)

Der MVP gilt als fertig, wenn ein echter Kunde ohne unsere Hilfe:

1. ✅ Sich registrieren kann
2. ✅ 5 KI-Systeme in seine Inventur einträgt
3. ✅ Alle 5 durch den Risk Classifier führt
4. ✅ Mindestens 1 Risk-Assessment-PDF generiert und herunterlädt
5. ✅ 3 Mitarbeiter durch das AI-Literacy-Training schickt und Zertifikate erhält
6. ✅ Einen Compliance-Status-Report für den Vorstand exportiert

… alles ohne, dass wir eingreifen oder Support geben müssen.

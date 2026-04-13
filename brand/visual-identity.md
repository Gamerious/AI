# Visual Identity – ComplAI

## 1. Leitidee

> **"Rechtliche Klarheit"** in visueller Form – sachlich, vertrauenswürdig, modern.

Keine Gimmicks, keine Gradient-Blobs, keine Unicorn-Maskottchen. Wir verkaufen an Compliance-Officer, nicht an TikTok-Teens.

---

## 2. Farbpalette

### Primary

| Name          | Hex       | Verwendung                       |
| ------------- | --------- | -------------------------------- |
| **Ink**       | `#0B1120` | Haupttext, Headings              |
| **Deep Blue** | `#1E3A8A` | CTA-Buttons, aktive Elemente     |
| **Paper**     | `#FAFBFC` | Hintergrund, Flächen             |

### Accent

| Name           | Hex       | Verwendung                            |
| -------------- | --------- | ------------------------------------- |
| **Citrus**     | `#F59E0B` | Warnungen, High-Risk-Kennzeichnung    |
| **Leaf**       | `#10B981` | Erfolg, Compliance erfüllt            |
| **Crimson**    | `#DC2626` | Prohibited, kritische Fehler          |

### Neutrals

| Name      | Hex       |
| --------- | --------- |
| Gray 50   | `#F9FAFB` |
| Gray 100  | `#F3F4F6` |
| Gray 300  | `#D1D5DB` |
| Gray 500  | `#6B7280` |
| Gray 700  | `#374151` |
| Gray 900  | `#111827` |

---

## 3. Typografie

- **Headlines**: [Inter](https://fonts.google.com/specimen/Inter) – 600/700, tight tracking
- **Body**: Inter – 400/500
- **Monospace** (Code, IDs): [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono)

Warum Inter? Kostenlos, hervorragende Lesbarkeit bei Compliance-Text, breite Sprachunterstützung (DE-Sonderzeichen sauber), sehr neutral – vertrauenswürdig.

### Größen (Web)

| Level   | Desktop | Mobile | Line-Height |
| ------- | ------- | ------ | ----------- |
| H1      | 48px    | 32px   | 1.1         |
| H2      | 36px    | 28px   | 1.2         |
| H3      | 24px    | 20px   | 1.3         |
| Body    | 16px    | 16px   | 1.6         |
| Small   | 14px    | 14px   | 1.5         |

---

## 4. Logo-Richtlinie

**Start-MVP**: **Wortmarke**, kein Symbol.

```
ComplAI
```

- Font: Inter Semibold
- "AI" im gleichen Grundton, aber leicht heller für subtile Betonung
- Alternativ: Monogramm `C•` als Icon für Favicon

**Später** (ab Monat 6):
- Symbol-Entwicklung durch Freelancer (Dribbble, ~€300)
- Geometrisches Element, das "Überprüfung/Stempel/Siegel" andeutet
- Keine Gesichter, keine Roboter, keine "connecting dots"-Klischees

---

## 5. UI-Prinzipien

1. **Max. 3 Farben pro Bildschirm** (Primary, 1 Accent, Neutrals)
2. **Weißraum ist dein Freund** – wir sind kein Enterprise-Dashboard-Gruselkabinett
3. **Content First** – Text lesbar, Aktionen klar
4. **Konsistente Icon-Sprache**: [Lucide Icons](https://lucide.dev/) (passt zu shadcn/ui)
5. **Accessibility**: WCAG AA minimum, AAA für Haupt-Text
6. **Dark Mode**: ja, aber erst nach MVP

---

## 6. Foto/Illustration-Stil

- **Keine Stock-Fotos** mit Handshakes oder lächelnden Business-Menschen
- **Abstrakte Illustrationen** (Open-Source wie unDraw, angepasst an Farbpalette)
- **Screenshots des Produkts** (echte UI, keine Mockups) – das ist das ehrlichste Marketing
- **Kein AI-generiertes Bildmaterial** (Ironie-Abstand zur eigenen Zielgruppe)

---

## 7. Assets-Plan (MVP-Phase)

- [ ] Wortmarke SVG (primary + monochrome)
- [ ] Favicon (32x32 + 16x16)
- [ ] Open-Graph-Image für Social-Share (1200×630)
- [ ] Email-Header (600×120)
- [ ] LinkedIn-Profilbild-Vorlage
- [ ] PDF-Cover-Template für Dokument-Export

Alle Assets werden in `/brand/assets/` abgelegt.

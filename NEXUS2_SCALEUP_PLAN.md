# NEXUS-2 — Scale-Up & Daten-Plan

Stand: 31.05.2026. Ziel: von „Story-Weitererzähler auf TinyStories (22–37M)" zu einem
echten, vergrößerten LM, das auf **Web-Daten** + **Konversationsdaten** trainiert ist —
auf gemieteter Cloud-GPU, nicht auf der lokalen 4060.

> **Leitprinzip (aus der bisherigen Arbeit):** Erst die CISA-Frage billig auf der Cloud
> klären, *dann* das teure Pretraining. Kein Geld in eine Architektur stecken, die sich
> nicht gegen einen gleich großen Standard-Transformer bewährt hat.

---

## Wichtiger Architektur-Hinweis (kostenrelevant!)

NEXUS/CISA wendet **dieselbe Zelle n_iterations-mal** an (Weight Sharing). Das heißt:
**Rechenaufwand ≈ n_iter × der eines normalen Transformers gleicher Parameterzahl.**
Bei n_iter=4 zahlst du also ~4× FLOPs pro Parameter. Für den Scale-Up folgt daraus
(gestützt durch das Iso-Depth-Ergebnis φ=0.46 aus der Recherche):

- **Beim Vergrößern eher BREITER (d_model↑) als TIEFER (n_iter↑)** gehen — Iterationen
  sind sub-linear effizient.
- n_iter beim großen Lauf evtl. auf 2–3 senken, um Kosten zu sparen.

---

## Stufe 0 — CISA-Validierung auf der Cloud (zuerst! ~5–15 €)

Bevor irgendein großes Pretraining: die offene Frage „bringt der Iterations-State echten
Mehrwert?" beantworten. Auf einer gemieteten **5090** (~1 €/h):

1. **A1–A4-A/B** (small 22M, 20k Steps, ~1,5 h/Lauf): Baseline (PPL 4,96) vs. `channel+NoPE`
   vs. `+QK-Norm+Stabilisierung+Deep-Supervision`. Klärt, ob die Fixes PPL senken.
2. **MQAR-Test** (winziges 3–7M-Modell, Minuten): UT-ohne-State vs. GRU-State vs. DeltaState.
   **Der eigentliche Beweis**, ob ein Iterations-Gedächtnis auf einer Recall-Aufgabe trägt.

**Gate:** Nur wenn MQAR einen klaren Sprung zeigt UND die A-Fixes helfen, lohnt der teure
Scale-Up mit CISA. Sonst: Standard-Transformer-Rezept vergrößern (auch völlig ok).
Kosten der Klärung: **~5–15 €, wenige Stunden.**

---

## Stufe 1 — Web-Pretraining (5090, ~150–350M Params)

### GPU & Größe
| GPU | VRAM | Realistische Modellgröße (mit Grad-Checkpointing) | Miete (grob) |
|-----|------|---------------------------------------------------|--------------|
| RTX 5090 | 32 GB | ~150–400M (n_iter=2–3) | ~0,7–1,2 €/h |
| A100 | 80 GB | ~0,5–1B | ~1,5–2 €/h |
| H100 | 80 GB | ~1–3B | ~2–4 €/h |

**Empfehlung Start:** 5090, **~200–300M reale Params**. Günstig, validiert die ganze
Pipeline, und ist schon ein spürbarer Sprung von 37M.

### Daten: FineWeb-Edu (Web)
- **Quelle:** FineWeb-Edu (HuggingFace) — gefilterte, qualitativ hochwertige Web-Texte.
  Verfügbar in Sample-Größen (10BT, 100BT, …) — wir nehmen ein **Subset**.
- **Menge (Faustregeln):**
  - Chinchilla-optimal ~20 Tokens/Param → 300M ⇒ ~6B Tokens.
  - „Overtrained" für bessere Qualität (modern üblich bei kleinen Modellen): **15–40B Tokens**.
  - Start-Empfehlung: **~10–20B Tokens** vom FineWeb-Edu-Sample.
- **Tokenizer vergrößern:** 16k → **32k** BPE, neu trainiert auf einem Web-Sample
  (16k ist für TinyStories ok, für Web zu klein). Pipeline analog `prepare_lm_data.py`.
- **Aufbereitung:** Download-Stream → Tokenisieren → in `.bin`/`.pt`-Shards packen
  (gepackte Sequenzen mit Dokument-Trennern). Disk: grob ~2 Byte/Token ⇒ 20B Tok ≈ 40 GB.

### Rezept (aus SOTA-Recherche, „kostenlose" Hebel)
- **muP** (maximal update parametrization): LR auf kleinem Proxy fitten, dann zero-shot auf
  die große Größe übertragen — spart teures LR-Raten.
- **QK-Norm**, sorgfältiges Init, RMSNorm, kosinus-LR mit Warmup. Längere Sequenzen (1024–2048).
- **Grad-Checkpointing pro Iteration** gegen den BPTT-Aktivierungsspeicher (CISA-spezifisch).

### Kosten/Zeit (grobe Schätzung, MUSS gemessen werden)
- 300M × ~15B Tokens × (n_iter≈2–3 Faktor) auf 5090: **grob 1–3 Tage**, **~25–75 €**.
- ⚠️ Throughput ist hardware-/implementierungsabhängig — erst mit einem kurzen Mess-Lauf
  (1000 Steps) die echten Tokens/s ermitteln, dann hochrechnen.

---

## Stufe 2 — Vom „Weitererzähler" zum Gesprächspartner (Konversation)

Pretraining macht das Modell sprachfähig, aber es *antwortet* nicht im Dialog. Dafür eine
zweite Phase (**SFT — Supervised Fine-Tuning**) auf Konversationsdaten:

### Daten (offen, frei nutzbar)
- **OASST2** (OpenAssistant) — echte Mensch-Assistent-Dialoge, mehrsprachig, Baumstruktur.
- **UltraChat / UltraChat-200k** — große Menge synthetischer, sauberer Dialoge.
- **Tülu-/SmolTalk-Stil-Mischungen** — kuratierte Instruktions-/Chat-Mischungen.
- Größenordnung: **einige 100k bis ~1M Dialoge** (~1–5B Token-Äquivalent). SFT ist viel
  kürzer/billiger als Pretraining (~Stunden, wenige €).

### Chat-Template
Spezial-Tokens einführen, z. B. ChatML-Stil:
```
<|system|> ... <|user|> ... <|assistant|> ...
```
Loss nur auf den Assistant-Antworten (User/System maskieren). Das ist die Standard-Methode,
um aus einem Basis-LM einen Chat-Assistenten zu machen.

### (Optional, später) Präferenz-Tuning
DPO/Preference-Optimierung für „nettere"/hilfreichere Antworten — erst sinnvoll, wenn SFT steht.

---

## Gesamt-Reihenfolge (empfohlen)

1. **Stufe 0** (CISA-Validierung, ~5–15 €) — Architektur-Frage endgültig klären.
2. **Tokenizer 32k + FineWeb-Edu-Pipeline** bauen (lokal vorbereitbar, GPU-arm).
3. **Stufe 1** (Web-Pretraining 200–300M auf 5090, ~25–75 €).
4. **Stufe 2** (Konv-SFT, wenige €) — Chat-fähig machen.
5. Erst danach über H100 / 1B+ / mehr Daten nachdenken.

**Gesamt für einen ersten „echten" Mini-Chat-Assistenten: grob 40–120 €** Cloud-Kosten,
über wenige Tage verteilt — abhängig davon, wie groß/lang wir gehen.

---

## Ehrliche Risiken & offene Punkte

- **CISA könnte sich nicht bewähren** (Stufe 0 negativ). Dann: Standard-Transformer
  vergrößern — kein Beinbruch, der Rest des Plans (Daten, SFT) gilt unverändert.
- **Throughput/Kosten sind geschätzt** — vor jedem großen Lauf ein kurzer Mess-Lauf.
- **Ein 200–300M-Modell ist KEIN ChatGPT.** Es wird einfache Dialoge führen, aber begrenzt
  bleiben. Realistische Erwartung: ein kleiner, eigener Assistent — beachtlich als Eigenbau,
  aber kein GPT-4.
- **Daten-Lizenzen** prüfen (FineWeb-Edu, OASST: offen; bei anderen Datasets checken).
- **CISA-Compute-Overhead** (n_iter×) macht uns teurer als ein Standard-Modell gleicher
  Größe — beim Budget einplanen.

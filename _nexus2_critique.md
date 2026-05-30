Ich habe genug verifiziert. Alle drei zitierten Papers existieren und werden korrekt wiedergegeben (φ=0.46 stimmt, UT-Need-Memory ist real, GDN/KDA-Linie ist real). Die "Delta-on-depth"-Achse scheint tatsächlich nicht direkt publiziert — DeltaNet/GDN fahren die Delta-Rule über die Sequenz, nicht über die Iterations-Achse. Damit kann ich das Gutachten erstellen.

---

# ADVERSARIALES GUTACHTEN: NEXUS-2

Ich habe Code (`model.py`) und alle drei zitierten arXiv-Referenzen verifiziert. Vorab das Wichtigste: **Der Vorschlag ist intellektuell ehrlicher als die meisten — er verkauft 85% Engineering nicht als Durchbruch.** Aber genau das ist auch sein Problem (siehe c).

## Was der Vorschlag RICHTIG macht (zur Entlastung)

- **Code-Lesart ist exakt.** Ich habe jede zitierte Zeile geprüft: Z.122-123 (log-Bias ins gemeinsame Softmax), Z.100/114 (RoPE nur auf spatial-k, nicht auf k_temporal), Z.135 (`spatial_out + temporal_out` ungeskaliert), Z.330 (tanh), Z.199 (ungezähmte iter_emb), Z.360 (hartkodiert). Alles korrekt. Kein erfundener Befund.
- **Die zitierten Papers existieren wirklich** und werden korrekt referiert: arXiv 2604.21999 (UT Need Memory), 2604.21106 (Iso-Depth, φ=0.46 verifiziert), GDN-2/KDA-Linie (2605.22791, 2412.06464). Keine Halluzination — das ist bei generierten Vorschlägen selten.

## (a) Neuheit — überwiegend NEIN, und der Vorschlag gibt das zu

Die Selbsteinschätzung "85% Rekombination" ist ehrlich, aber selbst die behauptete 15%-Wette ist dünner als dargestellt:

- **A1-A4 sind reines, bekanntes Engineering.** Getrennte Softmaxe, channel-Gating, NoPE auf nicht-positionaler Achse, QK-Norm, ReZero — alles Standard vor 2026. Keine Neuheit, das räumt der Vorschlag ein.
- **Die "genuin neue" Delta-on-depth-Achse ist schwächer abgegrenzt als behauptet.** Recherche zeigt: (1) "Understanding Transformer from the Perspective of Associative Memory" (2505.19488) reinterpretiert bereits diverse Update-Regeln als Memory-Mechanismen. (2) Recurrent-depth-Transformer mit explizitem State über die Tiefenachse sind eine etablierte Kategorie. (3) DeltaNet selbst (2406.06484) ist über die *Sequenz* parallelisiert. Die exakte Kombination "Delta-Rule-Speicher pro Position, fortgeschrieben über die *Iterations*-Achse" finde ich nicht 1:1 publiziert — **aber das ist eine winzige, inkrementelle Permutation eines bekannten Bausteins, kein neues Prinzip.** "Publizierbar als Workshop-Note" ist realistisch; "distinkter Beitrag gegen Ouro/Mamba-3" (vom Vorschlag selbst auf 25-35% taxiert) ist optimistisch. Ich würde eher 15-20% sagen.

**Das eigentliche, vom Vorschlag nicht laut genug ausgesprochene Problem:** Iso-Depth (φ=0.46) und UT-Need-Memory liefern zusammen ein vernichtendes Argument *gegen* das gesamte Unterfangen auf TinyStories — dazu (c).

## (b) Implementierbarkeit — JA für A-Kern, RISKANT für DeltaState

**A1-A4 + Deep Supervision + depth-sampling:** Trivial auf der 4060. Das sind ~50-100 Zeilen, keine OOM-Gefahr, kein Cloud-Bedarf. Realistisch in 1-2 Tagen. Hier lauert nichts.

**DeltaState (Komponente B) — hier sind drei konkrete Fallen, die der Vorschlag unterschätzt:**

1. **Speicher-Claim ist falsch gerechnet.** Der Vorschlag sagt "S ist B×N×H×dk×dk, ~2 GB schon bei small". Rechne nach für small (d_model=512, H=8, dk=64, N=512, B=16): 16×512×8×64×64×4 Byte = **8.6 GB allein für S in fp32**, pro Iteration im Autograd-Graph gehalten ×4 Iterationen für BPTT → **zig GB**. Das sprengt die 4060 (8GB) sofort und ist auch auf der 5090 (32GB) eng. Pro-Position-State-Matrizen sind der klassische OOM-Killer. Gradient-Checkpointing pro Iteration (vom Vorschlag genannt) mildert das, aber die N im Nenner bleibt brutal. **Der "~2 GB"-Wert ist um Faktor ~4-10 zu optimistisch.**
2. **chunked/parallelisierte Delta-Rule ist nicht-trivial.** GDN/DeltaNet brauchen den hardware-effizienten Householder-Chunk-Algorithmus, um nicht 10× langsamer als Attention zu sein. Über die Iterations-Achse (nur 4 Schritte) ist Chunking nicht direkt anwendbar — naiv sequenziell ist es lahm, aber bei nur 4 Iterationen evtl. tolerierbar. Das ist Implementierungs-Risiko, kein Showstopper.
3. **BPTT durch 4 Iterationen × DeltaState-Update = tiefe, instabile Gradientenkette.** Spektralradius<1 via sigmoid-α ist nötig (richtig erkannt), aber die Kombination mit dem bereits fragilen State-Pfad ist ein Stabilitäts-Risiko.

**Fazit (b):** A-Kern ist gratis und sicher. DeltaState ist auf der 4060 nur in der Mini-PoC-Größe (3-7M, d_model=256) machbar — was der Vorschlag in Stufe 2 zum Glück auch genau so plant. Aber die "small mit DeltaState"-Andeutung ist auf der 4060 nicht lauffähig.

## (c) Schlägt es die +0.9%-Baseline? — A-Kern wahrscheinlich JA, aber der Effekt ist Bug-Fix, nicht Architektur

Hier ist die zentrale skeptische Korrektur:

- **Die ~75% für "schlägt TinyStories-Baseline um 2-5% PPL" sind zu hoch.** Begründung: A1 macht das Gate gradientenfähig — *aber das beweist nur, dass das Gate sich bewegt, nicht dass es sich öffnet*. UT-Need-Memory zeigt, dass State-Mechanismen auf nicht-kombinatorischen Tasks keinen Vorteil bringen, und der Vorschlag selbst zitiert das. Das logische Resultat: **Mit A1 wird das Gate frei beweglich und schließt sich dann sauber, weil der temporal-Pfad auf TinyStories redundant ist** (das τ=t-1-Residual-Argument ist korrekt). Der PPL-Gewinn käme dann fast ausschließlich aus **A3 (QK-Norm, +1.5× LR) und A4 (Stabilisierung) und Deep Supervision** — also aus dem generischen Engineering, nicht aus CISA. Das ist ein Gewinn, aber er beweist die Architektur-These *nicht*; er begräbt sie sauber.
- **Iso-Depth φ=0.46 ist das Damoklesschwert.** Bei n_iter=4 ist der Kapazitätsgewinn des Loopings sub-linear (4 Iterationen ≈ 1.9× unique blocks wert). Das deckelt, wie viel jeder State-Mechanismus *überhaupt* beitragen kann, weil das Residual über die Iterationen schon das meiste trägt.
- **Wahrscheinlichstes Ergebnis:** A-Kern bringt **1-3% PPL auf TinyStories**, getrieben von QK-Norm/Stabilisierung/Deep-Supervision. Das Gate öffnet *nicht* deutlich über 30% (entgegen der Erwartung im Vorschlag). DeltaState auf MQAR ist ein echtes Coin-Flip — UT-Need-Memory stützt einen Sprung, aber n_iter=4 ist an der "borderline T=4"-Grenze des Papers, also Schwellenrisiko.

## (d) Das EINE billigste Experiment

Der Vorschlag nennt "Test 0" (Gradient-Sanity-Check des A1-Gates, 5 Min). **Das ist richtig priorisiert, aber es testet die falsche Kernannahme.** Ein bewegter Gradient beweist nur Mechanik, nicht Nutzen.

**Mein Gegenvorschlag für das billigste *aussagekräftige* Experiment — und es ist sogar noch billiger:**

> **Lese das aktuelle, gefixte (`detach_state_history=False`) Modell aus dem letzten 20k-Lauf und logge `sigmoid(temporal_gate)` über den Trainingsverlauf, plus eine Ablation: setze zur Eval-Zeit den temporal-Pfad hart auf 0 (`out = spatial_out` in Z.135) und miss den PPL-Delta.**

Kosten: **0 GPU-Minuten, nur ein Forward-Pass über das Eval-Set mit einem schon trainierten Checkpoint.** Wenn das Nullsetzen des temporal-Pfads die PPL kaum verändert (Erwartung: <0.5%), ist bewiesen, dass der State-Pfad *schon im gefixten Modell* nichts trägt — und die ganze A1-Hypothese ("der Pfad ist nur zugeschnürt, nicht nutzlos") ist falsifiziert, bevor eine einzige Zeile neuer Code geschrieben wird. Das ist strikt billiger und aussagekräftiger als Test 0.

Falls der Delta-fokussierte Pfad zählt: dann ist **PoC-B (MQAR) das richtige erste Training**, nicht TinyStories — auch das hat der Vorschlag korrekt erkannt.

## GO / NO-GO

**GO — aber nur für Stufe 1 (A-Kern auf der 4060, 0 USD), und mit umgehängter Erfolgsdefinition.** Begründung in einem Satz: Die A1-A4-Fixes sind mechanistisch zwingend, fast gratis und werden mit hoher Wahrscheinlichkeit 1-3% PPL bringen — aber dieser Gewinn beweist *nicht* CISA, sondern wird sie sauber als redundant entlarven; deshalb ist der eigentliche Lackmustest nicht TinyStories, sondern der 0-USD-Ablations-Test (d) am bestehenden Checkpoint plus PoC-B/MQAR, und **jede Cloud-Ausgabe (Stufe 3/4) ist NO-GO, bis PoC-B einen Größenordnungs-Sprung auf einem State-Task zeigt** — was ich auf <50% schätze.

**Konkrete Empfehlung zur Sequenz:** (1) 0-USD-Ablation am vorhandenen Checkpoint [heute], (2) A-Kern-A/B auf 4060 [diese Woche], (3) MQAR-PoC-B als echter State-Beweis [diese Woche], (4) DeltaState nur in 3-7M-Größe testen, niemals in "small" auf der 4060 — der Speicher-Claim "~2 GB" ist real ~8-9 GB. Erst danach über die 5090 nachdenken.

Relevante Datei: `C:\Users\janhe\nexus\nexus\lm\model.py` (alle zitierten Zeilen verifiziert).
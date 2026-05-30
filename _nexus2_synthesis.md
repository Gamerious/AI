Der Code bestätigt alle Audit-Befunde exakt (Z. 122-123 log-Bias-Gate, Z. 100/114 RoPE-Inkonsistenz, Z. 135 ungeskaliertes `spatial_out + temporal_out`, Z. 330 tanh, Z. 199 ungezähmte iter_emb, Z. 360 hartkodiertes avg_depth). Der Detach-Fix ist über `detach_state_history=False` korrekt eingebaut. Ich habe genug, um zu entscheiden.

---

# NEXUS-2: Entscheidung des leitenden Architekten

## Diagnose-Synthese (warum überhaupt eine neue Architektur)

Drei Befunde sind über alle Audits/Recherchen **einstimmig** und damit mein Fundament:

1. **CISA ist mechanistisch zugeschnürt** (Audit 1): Gate als log-Bias im gemeinsamen Softmax → temporale Masse im Promillebereich → Gate-Gradient ≈ 0. Plus RoPE-Fehlrotation (Audit 1/2/3) → temporaler Pfad ist verrauscht. Das sind echte Bugs, kein Tuning-Problem.
2. **Selbst sauber repariert ist der temporal-Pfad auf TinyStories größtenteils redundant** (Audit 2/3): Eine Softmax über ≤3 alte Positions-States kann nur eine Konvexkombination bereits existierender Vektoren bilden — das Residual trägt τ=t−1 ohnehin verlustfrei. Iso-Depth-Gesetz (φ=0.46, arXiv 2604.21106) quantifiziert das: Weight-Sharing ist real, aber sub-linear effizient.
3. **Der Messort ist falsch** (Audit 3 einstimmig, beide State-Recherchen): TinyStories ist greedy-lokal lösbar und datenlimitiert bei PPL ~4.5. Ein State-Mechanismus kann dort prinzipiell nicht glänzen. „Universal Transformers Need Memory" (arXiv 2604.21999) zeigt den Alles-oder-Nichts-Sprung (2.5% → 57.4% EM) **nur** auf kombinatorischen Tasks.

**Architekten-Schluss:** Die drei Design-Vorschläge konvergieren auf dieselben Fixes (getrennter Softmax, channel-Gate, NoPE-temporal). Wo sie sich unterscheiden, liegt das eigentliche Risiko. Ich kombiniere statt einen zu wählen — aber mit einer harten Sequenzierung, die teure/riskante Teile hinter billige Falsifikationstests stellt.

---

## (1) Finale Architektur-Spezifikation: NEXUS-2

**Leitprinzip:** Trenne sauber zwei Dinge, die CISA vermischt — (A) den **Iterations-Loop-Mechanismus** (was der State tut) und (B) den **Trainings-/Mess-Aufbau** (wo er es zeigt). Baue alles als orthogonale Config-Flags, Default = aktuelles Verhalten, damit jeder Baustein einzeln A/B-bar ist.

### Komponente A — Fix-Kern (übernimmt aus allen 3 Vorschlägen, niedrigstes Risiko)

Diese vier Änderungen sind über alle Audits konsensual und fast gratis. Sie sind **die Baseline von NEXUS-2**, nicht optional:

**A1 — Getrennte Softmaxe + multiplikatives channel-weises Gate** (löst Audit-Befund 1+5):
```python
# __init__: self.temporal_gate = nn.Parameter(torch.full((d_model,), temporal_gate_init))
spatial_w  = F.softmax(spatial_scores, dim=-1)   # eigener Softmax
temporal_w = F.softmax(temporal_scores, dim=-1)  # eigener Softmax über T
g = torch.sigmoid(self.temporal_gate)            # (D,) channel-weise
out = spatial_out + g * temporal_out             # Gate skaliert Amplitude, nicht Masse
```
Begründung: Der Gate-Gradient hängt jetzt an `temporal_out` (Amplitude O(1)) statt an ~0-Softmax-Masse. Channel-weise statt skalar, weil die Frage „welche Features aus dem State" lautet, nicht „wie viel State". Das ist der Haupthebel hinter dem 0.9%-Deckel.

**A2 — NoPE-Query für den temporal-Pfad** (löst Audit-Befund 2):
```python
q_nope = self.q_proj(x).view(B,N,H,dh).transpose(1,2)  # vor RoPE aufheben
q, k = apply_rotary_emb(q_nope, k, freqs_cis)          # RoPE nur spatial
# temporal_scores nutzen q_nope, nicht q
```
Die Iterations-Achse hat keine Positionssemantik — NoPE ist hier korrekt, nicht „fehlende Info" (Recherche 1, NoPE/iRoPE-Konsens).

**A3 — QK-Norm** (Recherche 3, „kostenloser" Hebel): RMSNorm auf q, k, k_state vor dem Dot-Product. Balanciert spatial- und temporal-Key-Skalen und erlaubt ~1.5× höhere LR. Eigene RMSNorm-Klasse (3 Zeilen), falls PyTorch < 2.4.

**A4 — Stabilisierung** (Audit-Befund 3+4, Parcae): `state = self.state_init(x)` **ohne** tanh (Z.330); eigener `state_norm = RMSNorm` vor dem GRU-Input (Z.211); iter_emb mit ReZero-Scale gezähmt: `x = x + tanh(self.iter_scale) * iter_emb`, `iter_scale` zero-init.

### Komponente B — der State-Mechanismus selbst (die eigentliche Architektur-Wette)

Hier liegt die Divergenz der Vorschläge. Meine Entscheidung: **gestaffelt nach Risiko.** Die GRU-Variante (mit A1-A4 repariert) ist die sichere Basis. Der DeltaState-Speicher (Vorschlag 2) ist das Upgrade mit dem höchsten Kapazitäts-Ceiling — aber teuer (S ist B×N×H×dk×dk, ~2 GB schon bei small) und über nur 4 Iterationen mit unklarer Korrektur-Reichweite. **Ich baue beide hinter ein Flag `state_mechanism: "gru" | "delta"`** und entscheide empirisch in Stufe 2, nicht jetzt.

DeltaState (wenn aktiviert): assoziativer KV-Speicher pro Position, über die Iterations-Achse mit Delta-Rule fortgeschrieben (`S = α⊙S + k⊗β(v−Sk)`), channel-weise α/β-Gates, α=sigmoid garantiert Spektralradius<1 (Parcae). Das ist der einzig genuin neue Teil (siehe (2)).

### Was ich aus den Vorschlägen VERWERFE und warum

- **Vorschlag 3 (Rauschinjektion / Consensus-Loss):** Verworfen für den Kern. Die Idee „Rauschen macht den temporal-Pfad nicht-redundant" ist elegant, aber sie löst ein **selbst erzeugtes** Problem — man beschädigt den Residualstrom, um den State künstlich nützlich zu machen. Das verbessert nicht das reale Sprachmodell, sondern baut einen Denoiser. Risiko-Ertrags-Verhältnis schlecht (Vorschlag 3 räumt selbst ein: TinyStories-PPL steigt wahrscheinlich). **Behalten** wird daraus nur **Deep Supervision** (Loss pro Iteration, aufsteigend gewichtet 0.5→1.0) — das ist der von TRM gemessene größte Einzelgewinn und gibt jeder Iteration ein O(1)-Gradientensignal statt 1e-6-BPTT-Reste.
- **Adaptives Halting / Ponder (Vorschlag 1):** Verschoben, nicht verworfen. Es bringt nachweislich Wert (Ouro/MoR), ist aber der wahrscheinlichste Instabilitäts-Stolperstein (Gate-Kollaps). Kommt erst in Stufe 3, nachdem der Kern stabil läuft. Stattdessen sofort billig: **per-sequence depth sampling** im Training (n_iter aus {3,4,5,6} ziehen, Parcae/Loop-Think) — stabilisiert über Seeds und ermöglicht Inferenz-Tiefen-Extrapolation, ohne Halting-Komplexität.

### Datenfluss NEXUS-2 (eine Iteration t)
```
x ← x + tanh(iter_scale)·iter_emb[t]                    # gezähmte Tiefen-Signatur
x ← x + Attn(RMSNorm(x)):                                # CISA-Kern, A1+A2+A3
      spatial = softmax(RoPE(q)·RoPE(k)/√d) · v          # eigener Softmax, RoPE
      temporal = softmax(q_nope·k_state/√d over T) · v_state   # eigener Softmax, NoPE
      out = spatial + sigmoid(gate_channel) ⊙ temporal   # multiplikatives channel-Gate
x ← x + FFN(RMSNorm(x))                                  # SwiGLU
state ← StateUpdate(state, RMSNorm(x))                   # GRU(A4) oder Delta(B)
loss += w[t]·CE(lm_head(out_norm(x)), labels)            # Deep Supervision
```

---

## (2) Was daran ehrlich NEU ist vs. Rekombination

**Schonungslos getrennt:**

- **Rekombination (nichts davon ist neu):** Universal/Looped Transformer (n_iter=4), getrennte Softmaxe, channel-weises Gating (GDN/KDA), NoPE auf einer nicht-positionalen Achse, QK-Norm, Deep Supervision (TRM), per-sequence depth sampling (Parcae), Spektralnorm-Stabilisierung. Alles publiziert vor Mai 2026.

- **Der einzige genuin neue Baustein:** **Channel-weises Delta-Rule-Gating angewandt auf die ITERATIONS-Achse statt der Zeit-Achse**, mit einem per-Position separaten assoziativen Speicher. GDN/KDA fahren die Delta-Rule über die Sequenzposition; die Looped-Linie (Ouro/LoopLM) verzichtet bewusst ganz auf separaten State. Ein fehlerkorrigierender assoziativer Speicher, der über die Tiefen-Rekurrenz eines Looped Transformers fortgeschrieben wird, ist publizierbar als *„what if GDN-style gating, but over recurrent depth"*. **Das ist die Achsen-Vertauschung aus Vorschlag 2** — und sie ist der einzige Teil, der nicht trivial aus der Literatur folgt.

- **Ehrliche Einordnung:** NEXUS-2 ist zu ~85% sauberes Engineering bekannter Bausteine (= die richtige Entscheidung, denn der 0.9%-Deckel kommt aus Bugs, nicht aus Ideenmangel) und zu ~15% eine echte, aber risikobehaftete Forschungswette (Delta-on-depth). Ich positioniere die Wette als testbare Hypothese, nicht als Verkaufsversprechen.

---

## (3) Priorisierte Experiment-Roadmap (billigstes/höchstes Signal zuerst)

### ⭐ ALLERERSTER TEST (vor allem anderen, ~5 Minuten, kein Training)
**Gradienten-Sanity-Check des A1-Fixes.** Bevor irgendein Training läuft: ein einzelner Forward+Backward auf einem Random-Batch mit dem alten vs. dem A1-Gate, und `temporal_gate.grad.abs().mean()` vergleichen.
- **Erwartetes Signal:** Mit dem log-Bias-Gate ist der Gradient ~1e-6…1e-4 (Audit-Befund bestätigt). Mit A1 (multiplikatives Amplituden-Gate) muss er um **2-4 Größenordnungen** springen (~1e-2…1e-1). Das ist der direkte, sekundenschnelle Beweis, dass der mechanistische Fix greift — **bevor** wir 1.5h GPU verbrennen. Springt er nicht, ist die A1-Hypothese sofort widerlegt.

### Stufe 1 — 4060-PoC, Mechanismus (~1.5h/Lauf, gratis)
**PoC-A: TinyStories, small 22M, 20k Steps, identisches Rezept zum bisherigen A/B.** Vier Läufe, Bausteine einzeln zugeschaltet:
1. Baseline (aktueller gefixter Stand) → Referenz PPL ~4.96
2. + A1 (getrennter Softmax + channel-Gate)
3. + A1+A2+A3 (NoPE-temporal + QK-Norm)
4. + A1-A4 + Deep Supervision

- **Kernmetrik:** PPL **und** `mean(sigmoid(temporal_gate))` über Kanäle/Training.
- **Erwartung:** Gate öffnet von 12% deutlich über 30%; PPL-Effekt steigt von 0.9% auf 2-4%. Wenn das Gate trotz sauberem Pfad geschlossen bleibt → temporal-Pfad ist auf Sprachdaten als redundant bestätigt (wertvolle, billige Erkenntnis).

### Stufe 2 — 4060-PoC, der EIGENTLICHE Beweis (~Minuten-Stunden/Lauf, gratis)
**PoC-B: kombinatorischer State-Task statt TinyStories.** Verkleinertes Modell (~3-7M, d_model=256-384) auf **MQAR** (Multi-Query Associative Recall, ~50 Zeilen Generator, kein Download) oder ListOps. Drei-Wege-A/B iso-Compute:
1. UT ohne State (temporal-Pfad aus)
2. GRU-State (A1-A4 repariert)
3. DeltaState (Komponente B)

- **Erwartung (UT-Need-Memory):** ohne State scheitert jede Config (~2.5% EM); mit State Sprung auf zig Prozent. Hier muss sich DeltaState gegen GRU beweisen (>5-10 EM-Punkte, nicht 0.9%).
- **Go/No-Go:** Das ist der Lackmustest. Zeigt PoC-B keinen Größenordnungs-Sprung, ist die ganze State-These widerlegt — dann behalten wir nur adaptive Tiefe + die A-Fixes und streichen den temporal-Pfad. **Kosten der Widerlegung: <1 Tag 4060, 0 USD.**

### Stufe 3 — gemietete 5090 (32GB), skalierter Lauf (~6-12h, ~6-15 USD)
Nur wenn Stufe 2 positiv. Hebel: **muP** (LR auf 4060-Proxy gefittet, zero-shot transferiert — spart das teure LR-Raten), **breiter statt tiefer** (Iso-Depth: d_model↑ statt n_iter↑), Daten-Wechsel zu FineWeb-Edu-Subset + längere Sequenzen (1024). Ziel: ~80-150M reale Params. Gradient-Checkpointing pro Iteration gegen BPTT-Aktivierungsspeicher.

### Stufe 4 — H100 (80GB), Voll-Lauf + adaptives Halting (~24-48h, ~50-200 USD)
Erst hier: adaptives Halting/Ponder (Vorschlag 1), echter KV-Cache mit Iteration-Sharing à la MELT (generate() von O(N²) auf O(N)), FP8 + FlashAttention-3, seq 2048. ~150-500M auf Reasoning-Korpus.

---

## (4) Ehrliche Erfolgswahrscheinlichkeit

Differenziert nach Ziel, nicht als eine Zahl:

- **„NEXUS-2 läuft stabil und schlägt die TinyStories-Baseline messbar (2-5% PPL)": ~75%.** Die A-Fixes sind mechanistisch fundiert und bug-getrieben; der Gradient-Sanity-Check (Test 0) wird das fast sicher früh bestätigen. Niedriges Risiko, weil es sauberes Engineering bekannter Bausteine ist.

- **„Der CISA-/Delta-State-Mechanismus zeigt auf einem State-Task einen Größenordnungs-Effekt (Beweis, dass die Architektur-Idee trägt)": ~45-55%.** Hier liegt die echte Unsicherheit. „UT Need Memory" stützt es stark, aber bei nur n_iter=4 ist die Delta-Korrektur-Reichweite klein, und Ouro/LoopLM beweisen, dass man State *ganz weglassen* kann und trotzdem skaliert. Das ist eine echte 50/50-Forschungswette — aber sie kostet nur Stufe-2-Zeit auf der 4060, um beantwortet zu werden.

- **„NEXUS-2 ist ein publizierbarer, gegenüber Ouro/Mamba-3 distinkter Beitrag": ~25-35%.** Nur die Delta-on-depth-Achse ist neu, und sie muss empirisch klar gewinnen, um sich gegen den state-freien Looped-Mainstream zu behaupten.

**Schärfster ehrlicher Satz:** Der größte Wert dieser Roadmap ist nicht die Garantie eines Durchbruchs, sondern dass sie die zentrale Architektur-Frage — *trägt ein Iterations-State überhaupt etwas, das das Residual nicht ohnehin liefert?* — für **0 USD und <1 Tag** auf der 4060 falsifizierbar macht (Test 0 + PoC-B), bevor ein einziger Cloud-Dollar fließt. Das ist die wichtigste Design-Entscheidung: nicht *welche* State-Variante, sondern *zuerst beweisen, dass State überhaupt zählt.*

Relevante Datei für die Umsetzung: `C:\Users\janhe\nexus\nexus\lm\model.py` — Änderungen an `CISAttention.__init__/forward` (Z. 59-141), `GRUStateUpdate` → optional `DeltaStateMemory` (Z. 148-161), `NexusLMCell.forward` (Z. 192-213), `NexusLM.forward` (Z. 326-358), `NexusLMConfig` (neue Flags), `get_diagnostics`/`count_parameters` (Z. 360, 426-432, ehrliches Framing).
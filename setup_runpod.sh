#!/usr/bin/env bash
# ============================================================================
# NEXUS-2 RunPod Setup
#
# Auf dem Pod (RunPod "PyTorch"-Template) so benutzen:
#   git clone https://github.com/Gamerious/AI.git
#   cd AI && git checkout claude/advanced-ai-architecture-VSb4j
#   bash setup_runpod.sh
#
# Danach die Experimente IN tmux starten (ueberlebt SSH-Abbruch):
#   tmux new -s train
# ============================================================================
set -e

echo "==> 1/3  Python & GPU-Check"
python -c "import torch; print('  torch', torch.__version__, '| CUDA verfuegbar:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('  KEINE GPU gefunden!')"

echo "==> 2/3  Extra-Abhaengigkeiten (torch kommt vom Template, wird NICHT angefasst)"
pip install -q requests

echo "==> 3/3  TinyStories-Daten vorbereiten (Download + 16k-BPE-Tokenizer)"
if [ -f "lm_data/train.pt" ]; then
  echo "  lm_data/train.pt existiert bereits - ueberspringe."
else
  python prepare_lm_data.py
fi

echo
echo "============================================================"
echo "  Setup fertig. Jetzt in tmux starten:"
echo
echo "  # NEXUS-2 (alle Fixes) vs. Baseline (PPL 4.96):"
echo "  python train_nexus_lm.py --config small --max-steps 20000 --cisa-v2"
echo "  python train_nexus_lm.py --config small --max-steps 20000"
echo
echo "  # Der eigentliche State-Beweis (Recall-Aufgabe):"
echo "  python mqar_train.py --mode ut"
echo "  python mqar_train.py --mode gru"
echo "  python mqar_train.py --mode v2"
echo
echo "  # 0-Kosten-Ablation auf einem fertigen Checkpoint:"
echo "  python ablation_temporal.py nexus_lm_small_v2.pt"
echo
echo "  ----- SCALE-UP (Webdaten, FineWeb-Edu) -----"
echo "  pip install -r requirements-web.txt          # datasets + fast tokenizer"
echo "  python prepare_fineweb.py --tokens 300M       # erst klein (Pipeline-Test, ~Minuten)"
echo "  python prepare_fineweb.py --tokens 2B         # echter Stage-1-Lauf"
echo "  python train_nexus_lm.py --config base --data-dir lm_data_web --cisa-v2 --bs 32"
echo
echo "  WICHTIG: Pod stoppen, wenn fertig - sonst tickt die Miete weiter!"
echo "============================================================"

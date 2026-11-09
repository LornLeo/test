#!/usr/bin/env bash
# Role D design sensitivity sweep (docs/D_SWEEP_PLAN.md). Dev only; Test is never used.
#
#   bash scripts/d_sweep_runs.sh all            # every run in the plan, MAX_JOBS at a time
#   bash scripts/d_sweep_runs.sh depth|width|ablation
#
# Each run writes results/sweep/<name>_s<seed>/ and checkpoints/sweep/<name>_s<seed>.pt.
# The 1-layer, width-128, full-recipe point is the existing R runs (results/stage2/d_stage1_ref_s*).
set -euo pipefail

PY="${PY:-.venv/Scripts/python.exe}"
LOGS="${D_LOG_DIR:-$(mktemp -d)}"
MAX_JOBS="${MAX_JOBS:-8}"
SEEDS=(42 43 44)
mkdir -p "$LOGS" checkpoints/sweep results/sweep

# Stage 1 weight-2 recipe without the two components under test; they are added per run.
CORE=(
  --word-vectors embeddings/glove_train_100.txt --decoder crf
  --intent-pooling mean_max --slot-weight 2
  --embedding-dim 100 --hidden-dim 128 --layers 1 --dropout 0.3
  --learning-rate 0.001 --weight-decay 0.00001 --batch-size 64
  --epochs 60 --patience 8 --device cpu --threads 2
)
FULL=(--char-features --word-dropout 0.1)

run() {  # name seed [extra args...]; later arguments override the core recipe
  local name="$1" seed="$2"
  shift 2
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n || true; done
  "$PY" -X utf8 -m src.model.train "${CORE[@]}" "$@" --seed "$seed" \
    --run-dir "results/sweep/${name}_s${seed}" \
    --checkpoint "checkpoints/sweep/${name}_s${seed}.pt" \
    > "$LOGS/${name}_s${seed}.out" 2>&1 &
  echo "$(date '+%F %T') started ${name}_s${seed} (pid $!, console: $LOGS/${name}_s${seed}.out)"
}

depth() { for s in "${SEEDS[@]}"; do
  run d_sweep_layers3 "$s" "${FULL[@]}" --layers 3
  run d_sweep_layers2 "$s" "${FULL[@]}" --layers 2
done; }
width() { for s in "${SEEDS[@]}"; do
  run d_sweep_hidden256 "$s" "${FULL[@]}" --hidden-dim 256
  run d_sweep_hidden64 "$s" "${FULL[@]}" --hidden-dim 64
done; }
ablation() { for s in "${SEEDS[@]}"; do
  run d_abl_nochar "$s" --word-dropout 0.1
  run d_abl_nowd "$s" --char-features --word-dropout 0.0
  run d_abl_neither "$s" --word-dropout 0.0
done; }

case "${1:?usage: d_sweep_runs.sh all|depth|width|ablation}" in
  all) depth; width; ablation ;;  # largest models first, so the slowest runs start earliest
  depth) depth ;;
  width) width ;;
  ablation) ablation ;;
  *) echo "unknown sweep: $1" >&2; exit 2 ;;
esac

status=0
for job in $(jobs -p); do
  wait "$job" || status=1
done
echo "$(date '+%F %T') all sweep runs finished, status=$status"
exit "$status"

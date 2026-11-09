#!/usr/bin/env bash
# Role D, Stage 2 on Role C's proposals (report Sections 4 and 5.3). Dev only; Test is never used.
#
#   bash scripts/d_c_runs.sh reference 42 43 44   # unchanged Stage 1 recipe, same machine (R)
#   bash scripts/d_c_runs.sh c1 42 43 44          # C1: class-weighted intent loss
#   bash scripts/d_c_runs.sh c2 42 [43 44]        # C2: C1 + longer training (100 epochs, patience 12)
#
# Each run writes results/stage2/<name>_s<seed>/ and checkpoints/stage2/<name>_s<seed>.pt.
# Console copies go to a temporary folder; each run's training.log holds the same lines.
# The reference and C1 runs in results/stage2/ were produced with these exact arguments.
set -euo pipefail

PY="${PY:-.venv/Scripts/python.exe}"
LOGS="${D_LOG_DIR:-$(mktemp -d)}"
mkdir -p "$LOGS" checkpoints/stage2

# Stage 1 weight-2 recipe (report Section 5.1).
BASE=(
  --word-vectors embeddings/glove_train_100.txt --char-features --decoder crf
  --word-dropout 0.1 --intent-pooling mean_max --slot-weight 2
  --embedding-dim 100 --hidden-dim 128 --layers 1 --dropout 0.3
  --learning-rate 0.001 --weight-decay 0.00001 --batch-size 64
  --epochs 60 --patience 8 --device cpu --threads 2
)

run() {  # name seed [extra args...]; later arguments override the base recipe
  local name="$1" seed="$2"
  shift 2
  "$PY" -X utf8 -m src.model.train "${BASE[@]}" "$@" --seed "$seed" \
    --run-dir "results/stage2/${name}_s${seed}" \
    --checkpoint "checkpoints/stage2/${name}_s${seed}.pt" \
    > "$LOGS/${name}_s${seed}.out" 2>&1 &
  echo "started ${name}_s${seed} (pid $!, console: $LOGS/${name}_s${seed}.out)"
}

kind="${1:?usage: d_c_runs.sh reference|c1|c2 SEED...}"
shift
for seed in "$@"; do
  case "$kind" in
    reference) run d_stage1_ref "$seed" ;;
    c1) run d_intentcw "$seed" --intent-class-weights ;;
    c2) run d_intentcw_long "$seed" --intent-class-weights --epochs 100 --patience 12 ;;
    *) echo "unknown run kind: $kind" >&2; exit 2 ;;
  esac
done

status=0
for job in $(jobs -p); do
  wait "$job" || status=1
done
echo "all runs finished, status=$status"
exit "$status"

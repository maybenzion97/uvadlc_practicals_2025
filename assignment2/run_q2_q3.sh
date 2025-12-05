#!/bin/bash
# Helper script to run the required code for:
#   - Q2.7  (LLM training on Grimm corpus)
#   - Q2.8b (generation experiments / accumulated knowledge)
#   - Q3.4d (graph convolution vs graph attention training)
#
# This script is designed to be called from anywhere; it will cd into the
# assignment2 directory where it lives and then run the appropriate commands.
#
# Usage:
#   Local (e.g., on your Mac with MPS):
#       bash assignment2/run_q2_q3.sh
#   On SLURM (inside a job, using srun):
#       bash assignment2/run_q2_q3.sh slurm

set -u  # fail on undefined variables; do NOT use -e so later parts still run

# Decide whether to wrap python calls in srun (for SLURM) or not (local)
MODE="${1:-local}"   # default: local
if [[ "${MODE}" == "slurm" ]]; then
  PYTHON_RUNNER="srun python"
else
  PYTHON_RUNNER="python"
fi

# Always work relative to this script's directory (assignment2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1
echo "====================================================================="
echo "Q2.7: Train LLM on 'Fairy Tales by Brothers Grimm' for 1 epochs without FlashAttention and compile"
echo "====================================================================="
(
  cd part2 || exit 0

  # Command recommended in the assignment README:
  # uses cfg.py defaults (Grimm corpus), enables FlashAttention and compile.
  ${PYTHON_RUNNER} train.py --num_epochs 1 --num_workers 8
)
if [ $? -ne 0 ]; then
  echo "Q2.7 training FAILED (continuing to Q2.7 with FlashAttention and compile) without FlashAttention and compile..."
fi




cd "${SCRIPT_DIR}" || exit 1
echo "====================================================================="
echo "Q2.7: Train LLM on 'Fairy Tales by Brothers Grimm' for 5 epochs"
echo "====================================================================="
(
  cd part2 || exit 0

  # Command recommended in the assignment README:
  # uses cfg.py defaults (Grimm corpus), enables FlashAttention and compile.
  ${PYTHON_RUNNER} train.py --use_flash_attn --compile --num_epochs 5 --num_workers 8
)
if [ $? -ne 0 ]; then
  echo "Q2.7 training FAILED (continuing to Q2.8b)..."
fi

echo
echo "====================================================================="
echo "Q2.8b: Evaluate generations (accumulated knowledge)"
echo "====================================================================="
(
  cd part2 || exit 0

  # Determine latest version_* folder under logs/gpt-mini for model weights
  LATEST_CHECKPOINT_DIR=""
  if [ -d "./logs/gpt-mini" ]; then
    LATEST_CHECKPOINT_DIR=$(ls -d ./logs/gpt-mini/version_* 2>/dev/null | sort -V | tail -n 1)
  fi

  if [ -z "${LATEST_CHECKPOINT_DIR}" ]; then
    echo "No checkpoint directory ./logs/gpt-mini/version_* found; skipping Q2.8b."
    exit 0
  fi

  LATEST_CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR}/checkpoints"
  echo "Using checkpoints from: ${LATEST_CHECKPOINT_DIR}"

  # Uses default model_weights_folder and prompts/configurations from
  # evaluate_generation.py. Adjust arguments if you want a different run.
  ${PYTHON_RUNNER} eval_generation.py \
    --model_weights_folder "${LATEST_CHECKPOINT_DIR}" \
    --output_file q2_8b_generation_results.json
)
if [ $? -ne 0 ]; then
  echo "Q2.8b evaluation FAILED (continuing to Q3.4d)..."
fi

echo
echo "====================================================================="
echo "Q3.4d: Train Graph CNN and Graph Attention (TensorBoard plots)"
echo "====================================================================="
(
  cd part3 || exit 0
  # Train and log all three model variants as required in Q3.4:
  #   - gcn          (message-passing GCN)
  #   - matrix-gcn   (matrix-based GCN)
  #   - gat          (graph attention)
  for MODEL in gcn matrix-gcn gat; do
    echo "---- Training GraphNN with --model ${MODEL} ----"
    ${PYTHON_RUNNER} train.py --model "${MODEL}"
  done
)
if [ $? -ne 0 ]; then
  echo "Q3.4d training FAILED."
fi

echo
echo "Done. Check TensorBoard logs and JSON outputs for results."




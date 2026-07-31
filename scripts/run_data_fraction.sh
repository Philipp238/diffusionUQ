#!/bin/bash
# Runs the data-fraction epistemic/aleatoric disentanglement experiment.
#
# Default: uses the mixednormal config (paper method arXiv:2510.04583).
# Both distributional methods (mixednormal and normal) use the paper's
# single-model decomposition (arXiv:2510.04583, Appendix F):
#   mixednormal  — closed-form law-of-total-variance on K mixture components
#   normal       — EU = V[mu_θ] ∝ sigma_θ²(x_t,t); AU = beta_tilde_t (schedule)
#
# Step 1: trains 15 models (5 fractions × 3 seeds) on the concrete UCI dataset.
# Step 2: analyzes epistemic/aleatoric decomposition and saves a plot.
#
# Usage:
#   bash scripts/run_data_fraction.sh              # mixednormal (default)
#   bash scripts/run_data_fraction.sh --normal     # normal distributional method
#
# To only re-plot from an already-trained result dir (EU/AU already in test.csv):
#   python experiments/plot_data_fraction.py results/<dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="data_fraction_mixednormal.ini"
EXPERIMENT="data_fraction_mixednormal"

if [[ "${1:-}" == "--normal" ]]; then
    CONFIG="data_fraction_normal.ini"
    EXPERIMENT="data_fraction_normal"
fi

echo "=== Config: $CONFIG ==="
echo "=== Step 1: Training (5 fractions × 3 seeds = 15 runs) ==="
python main.py -c "$CONFIG"

# Find the most-recently created results directory for this experiment
RESULTS_DIR=$(ls -dt results/*"${EXPERIMENT}"* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: Could not find results directory. Did training succeed?"
    exit 1
fi

echo ""
echo "=== Step 2: Plot EU/AU vs. data fraction (numbers already in test.csv) ==="
echo "Results directory: $RESULTS_DIR"
python experiments/plot_data_fraction.py "$RESULTS_DIR"

echo ""
echo "Done. Outputs written to $RESULTS_DIR"

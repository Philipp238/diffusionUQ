#!/usr/bin/env bash
# Run derive_variance_methods.sh across all 8 UCI datasets.
#
# Usage:
#   scripts/derive_all_uci.sh [splits] [methods] [ocm_epochs]
#
# Args (all optional, all forwarded to derive_variance_methods.sh):
#   splits      Split indices (default: "0..19"; protein & yacht may skip some)
#   methods     Methods to run (default: "analytic_dpm analytic_dpm_diag ocm")
#   ocm_epochs  OCM stage-2 epochs (default: 1000)
#
# Continues on per-dataset failure so one bad dataset does not abort the rest;
# a summary is printed at the end.

set -uo pipefail

SPLITS="${1:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
METHODS="${2:-analytic_dpm analytic_dpm_diag ocm}"
OCM_N_EPOCHS="${3:-1000}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRIVER="${REPO_ROOT}/scripts/derive_variance_methods.sh"

DATASETS=(concrete energy kin8nm naval power protein wine yacht)

failures=()
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "########## ${ds} ##########"
    if ! bash "${DRIVER}" "${ds}" "${SPLITS}" "${METHODS}" "${OCM_N_EPOCHS}"; then
        echo "!! ${ds} failed; continuing with next dataset"
        failures+=("${ds}")
    fi
done

echo ""
echo "=========================================="
if [[ ${#failures[@]} -eq 0 ]]; then
    echo "All datasets processed successfully."
else
    echo "Failed datasets: ${failures[*]}"
    exit 1
fi

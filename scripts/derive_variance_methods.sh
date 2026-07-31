#!/usr/bin/env bash
# Derive Analytic-DPM (scalar + diag) and OCM results from existing
# deterministic CARD-baseline checkpoints for a UCI dataset.
#
# Reuses the same trained backbone across all three variance methods,
# so the resulting comparison isolates the variance mechanism from the
# mean predictor. Source folders under results/UCI/<dataset>/ are
# auto-discovered — splits are matched to whichever folder contains
# the corresponding *_deterministic_*.pt.
#
# Usage:
#   scripts/derive_variance_methods.sh <dataset> [splits] [methods] [ocm_epochs]
#
# Args:
#   dataset     Short dataset name matching the folder under results/UCI/
#               (yacht, concrete, energy, kin8nm, naval, power, protein, wine)
#   splits      Space-separated split indices (default: "0..19")
#   methods     Space-separated methods to run
#               (default: "analytic_dpm analytic_dpm_diag ocm")
#   ocm_epochs  Epochs for OCM stage-2 head training (default: 1000)
#
# Output layout:
#   results/variance_methods/<dataset>/<method>/split<i>/...
#
# Notes:
#   - Source folders auto-discovered from results/UCI/<dataset>/*/.
#   - Per-dataset batch_size and canonical UCI dataset name are looked up
#     from the tables below (edit those if you add datasets).
#   - Runs sequentially. Parallelise externally if you have multiple GPUs.

set -euo pipefail

DATASET_SHORT="${1:?dataset short name required}"
SPLITS="${2:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
METHODS="${3:-analytic_dpm analytic_dpm_diag ocm}"
OCM_N_EPOCHS="${4:-1000}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Short folder name -> canonical UCI dataset name. This name is used both
# in the config's dataset_name field (data.data_utils.UCI_DATASET_NAMES)
# and as the prefix in the checkpoint filename (e.g. "yacht0", "naval-propulsion-plant3").
case "${DATASET_SHORT}" in
    concrete) DATASET_FULL="concrete" ;;
    energy)   DATASET_FULL="energy" ;;
    kin8nm)   DATASET_FULL="kin8nm" ;;
    naval)    DATASET_FULL="naval-propulsion-plant" ;;
    power)    DATASET_FULL="power-plant" ;;
    protein)  DATASET_FULL="protein-tertiary-structure" ;;
    wine)     DATASET_FULL="wine-quality-red" ;;
    yacht)    DATASET_FULL="yacht" ;;
    *) echo "Unknown dataset: ${DATASET_SHORT}"; exit 1 ;;
esac

# Per-dataset batch size used in the CARD baseline. Only affects OCM
# stage-2 training; Analytic-DPM is eval-only so unaffected.
case "${DATASET_SHORT}" in
    concrete|energy|wine|yacht) BATCH_SIZE=32 ;;
    kin8nm|naval|power)         BATCH_SIZE=64 ;;
    protein)                    BATCH_SIZE=256 ;;
    *)                          BATCH_SIZE=32 ;;
esac

SRC_DIR_ROOT="${REPO_ROOT}/results/UCI/${DATASET_SHORT}"
if [[ ! -d "${SRC_DIR_ROOT}" ]]; then
    echo "Source root not found: ${SRC_DIR_ROOT}"
    exit 1
fi
# All dated run folders that may contain deterministic checkpoints for splits.
mapfile -t SRC_DIRS < <(ls -d "${SRC_DIR_ROOT}"/*/ 2>/dev/null)
if [[ ${#SRC_DIRS[@]} -eq 0 ]]; then
    echo "No source folders under ${SRC_DIR_ROOT}"
    exit 1
fi

TEMPLATE_DIR="${REPO_ROOT}/config/variance_methods"
TMP_CFG_DIR="${TEMPLATE_DIR}/_generated"
mkdir -p "${TMP_CFG_DIR}"

find_checkpoint() {
    # Search all discovered source folders for a deterministic checkpoint
    # matching this (dataset, split). Prints the first hit and returns 0;
    # returns 1 if none found.
    local split="$1"
    for src in "${SRC_DIRS[@]}"; do
        local ckpt
        ckpt="$(ls "${src}"*_Loss_${DATASET_FULL}${split}_MLP_diffusion_deterministic_*.pt 2>/dev/null | head -1 || true)"
        if [[ -n "${ckpt}" ]]; then
            echo "${ckpt}"
            return 0
        fi
    done
    return 1
}

instantiate_config() {
    local template="$1" out="$2" ckpt="$3" split="$4" results_path="$5" exp="$6"
    sed \
        -e "s|__CHECKPOINT_PATH__|${ckpt}|g" \
        -e "s|__SPLIT__|${split}|g" \
        -e "s|__DATASET__|${DATASET_FULL}|g" \
        -e "s|__RESULTS_PATH__|${results_path}|g" \
        -e "s|__EXPERIMENT_NAME__|${exp}|g" \
        -e "s|__OCM_N_EPOCHS__|${OCM_N_EPOCHS}|g" \
        -e "s|__BATCH_SIZE__|${BATCH_SIZE}|g" \
        "${TEMPLATE_DIR}/${template}" > "${out}"
}

n_processed=0
n_skipped=0

for split in ${SPLITS}; do
    ckpt="$(find_checkpoint "${split}")" || {
        echo "[skip] ${DATASET_SHORT} split=${split}: no deterministic checkpoint in ${SRC_DIR_ROOT}"
        n_skipped=$((n_skipped + 1))
        continue
    }
    echo "=== ${DATASET_SHORT} split=${split} ==="
    echo "checkpoint: ${ckpt}"

    for method in ${METHODS}; do
        RESULTS_PATH="results/variance_methods/${DATASET_SHORT}/${method}/split${split}/"
        EXP_NAME="${DATASET_SHORT}${split}_${method}"
        CFG_NAME="_gen_${DATASET_SHORT}${split}_${method}.ini"
        CFG_PATH="${TMP_CFG_DIR}/${CFG_NAME}"
        instantiate_config "${method}.template.ini" "${CFG_PATH}" \
            "${ckpt}" "${split}" "${RESULTS_PATH}" "${EXP_NAME}"
        REL_CFG="variance_methods/_generated/${CFG_NAME}"
        echo "--- ${method} ---"
        (cd "${REPO_ROOT}" && python main.py -c "${REL_CFG}")
    done
    n_processed=$((n_processed + 1))
done

echo "Done. ${DATASET_SHORT}: processed=${n_processed}, skipped=${n_skipped}"

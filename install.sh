#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required but was not found on PATH." >&2
  echo "Please install Miniconda or Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n diffusion_uq python=3.12
conda activate diffusion_uq

conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y numpy pandas scikit-learn matplotlib
conda install -y wandb -c conda-forge
conda install -y -c conda-forge xarray dask netCDF4 bottleneck
conda install -y h5py
conda install -y -c conda-forge py-pde

echo "Environment 'diffusion_uq' created and populated."

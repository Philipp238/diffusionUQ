# Improved Probabilistic Regression Using Diffusion Models

This repository provides the supplementary code accompanying the paper **"Improved probabilistic regression using diffusion models"**. It implements the diffusion-based framework introduced in the manuscript for learning predictive distributions in a nonparametric manner, enabling calibrated probabilistic regression across low- and high-dimensional tasks.

## Abstract

Probabilistic regression models the entire predictive distribution of a response variable, offering richer insights than classical point estimates and directly allowing for uncertainty quantification. While diffusion-based generative models have shown remarkable success in generating complex, high-dimensional data, their usage in general regression tasks often lacks uncertainty-related evaluation and remains limited to domain-specific applications. We propose a novel diffusion-based framework for probabilistic regression that learns predictive distributions in a nonparametric way. More specifically, we propose to model the full distribution of the diffusion noise, enabling adaptation to diverse tasks and enhanced uncertainty quantification. We investigate different noise parameterizations, analyze their trade-offs, and evaluate our framework across a broad range of regression tasks, covering low- and high-dimensional settings. For several experiments, our approach shows superior performance against existing baselines, while delivering calibrated uncertainty estimates, demonstrating its versatility as a tool for probabilistic prediction.

## Repository Layout

- `config/`: Experiment configuration files grouped by benchmark (e.g., Burgers, KS, CARD, T2M). Each `.ini` file specifies training, model, and data parameters used in the paper.
- `data/`: Data loaders, preprocessing utilities, and scripts to download or generate datasets (see `data/download/` and dataset-specific folders such as `data/1D_Burgers/`).
- `evaluation/`: Evaluation routines, plotting helpers, and post-processing scripts for quantitative and qualitative analysis.
- `models/`: Diffusion backbones, baseline models, and pre-trained checkpoints used for ablations and comparisons.
- `results/`: Default output directory for experiment artifacts, including metrics, logs, and model weights.
- `utils/`: Shared utilities for configuration handling, logging, metrics, and training helpers.
- `main.py`: Entry point that orchestrates training and evaluation for a given configuration.
- `train.py` / `evaluate.py`: Core training loop and evaluation utilities invoked by `main.py`.
- `run.sh`: Convenience script showing how to launch a Burgers experiment.

## Getting Started

1. **Clone the project**
   ```bash
   git clone <REPO_URL>
   cd diff
   ```
2. **Install dependencies**
   ```bash
   bash install.sh
   ```
   The script bootstraps a Conda environment named `diffusion_uq`, installs PyTorch with CUDA 11.8 support, and adds the scientific Python stack plus `wandb`, `xarray`, and `py-pde`. Inspect or tweak the command list in `install.sh` if you need different CUDA or package versions.
3. **Activate the environment**
   ```bash
   conda activate diffusion_uq
   ```
   Make sure Conda is initialized in your shell (run `conda init <shell>` once if needed) so that activation works outside the script.
4. **Prepare data**
   - Consult the dataset-specific instructions in `data/`.
   - For automated downloads (e.g., ERA5, PDEBench), run the scripts in `data/download/`.
   - Generated or processed datasets are stored under the respective benchmark folder (e.g., `data/1D_Burgers/`).

## Running Experiments

- **Train & evaluate via configuration file**
  ```bash
  python main.py -c Burgers/deterministic.ini
  ```
  Replace the configuration with any file from `config/` to reproduce experiments from the paper. The script creates a time-stamped subdirectory in the configured `results_path`, copies the config for provenance, and logs training/evaluation summaries.

- **Specify a custom results directory**
  ```bash
  python main.py -c KS/your_experiment.ini -f custom_results
  ```

- **Batch experiments**: Use `run.sh` as a template for launching repeated runs or for integration into HPC job scripts.

## Evaluation & Postprocessing

- Use the evaluation utilities under `evaluation/` (e.g., plotting scripts) to compute distributional metrics such as CRPS or energy scores from saved model outputs.
- The `results/` directory stores serialized predictions, training logs, and diagnostics that can be aggregated with `evaluation/` notebooks or scripts.

## Citing

If you use this codebase in your work, please cite the accompanying paper:

> *Improved probabilistic regression using diffusion models*, 2025.

```
@article{diffusionuq2024,
  title   = {Improved probabilistic regression using diffusion models},
  year    = {2025},
  author  = {Carlo Kneissl, Christopher Bülte, Philipp Scholl, and Gitta Kutyniok},
  journal = {To be updated}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the authors of the paper.

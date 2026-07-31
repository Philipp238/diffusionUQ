"""Compute rank histograms (PIT histograms) for trained models on the test set.

For each (dataset, method) we load the first checkpoint (lexicographically), draw
M=100 samples per test example, and record the rank of the true observation among
the M ensemble members. Spatial datasets (Burgers, KS, T2M) use a spatial-mean
pre-rank function: both the prediction ensemble and the observation are reduced
to a scalar via the spatial mean before ranking.

Run from the repo root:
    python evaluation/calibration.py
    python evaluation/calibration.py --datasets yacht --methods deterministic
"""

import argparse
import ast
import configparser
import glob
import os
import pathlib
import re
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import get_datasets, get_dataset_metadata
from evaluate import generate_samples
from models import Diffusion
from utils import process_dict, train_utils


# Map dataset key -> source directory containing one or more run-dirs (UCI) or method-dirs (spatial)
UCI_DATASETS = {
    "energy":   "results/UCI/energy",
    "kin8nm":   "results/UCI/kin8nm",
    "naval":    "results/UCI/naval",
    "power":    "results/UCI/power",
    "protein":  "results/UCI/protein",
    "wine":     "results/UCI/wine",
    "yacht":    "results/UCI/yacht",
    "concrete": "results/UCI/concrete",
}

SPATIAL_DATASETS = {
    "Burgers": "results/Burgers",
    "KS":      "results/KS",
    "T2M":     "results/T2M",
}

# crps_ensemble lives in its own results tree with a flat layout:
#   results/crps_ensemble/<dataset>/{*_crps_*.pt, <dataset>.ini}
CRPS_ENSEMBLE_UCI_ROOT = "results/crps_ensemble"
CRPS_ENSEMBLE_SPATIAL_ROOT = "results/crps_ensemble"

METHODS = ["deterministic", "normal", "mvnormal", "mixednormal", "sample", "crps_ensemble"]


def find_first_uci_checkpoint(dataset_dir, method):
    """Find the first (sorted) .pt under any run-dir of dataset_dir matching method.

    Returns (ckpt_path, ini_path) or (None, None) if not found.
    """
    pts = sorted(glob.glob(os.path.join(dataset_dir, "*", f"*_{method}_*.pt")))
    if not pts:
        return None, None
    ckpt = pts[0]
    run_dir = os.path.dirname(ckpt)
    inis = glob.glob(os.path.join(run_dir, "*.ini"))
    if not inis:
        return None, None
    return ckpt, inis[0]


def find_first_crps_ensemble_checkpoint(dataset_key, kind):
    """crps_ensemble layout: results/crps_ensemble/<dataset>/*_crps_*.pt + <dataset>.ini.

    The on-disk filename uses the inner CRPS variant (e.g. '_crps_mvnormal_'),
    not 'crps_ensemble', so we glob on '_crps_' instead of the method name.
    """
    base = os.path.join(REPO_ROOT, CRPS_ENSEMBLE_UCI_ROOT, dataset_key)
    if not os.path.isdir(base):
        return None, None
    pts = sorted(glob.glob(os.path.join(base, "*_crps_*.pt")))
    if not pts:
        return None, None
    ini = os.path.join(base, f"{dataset_key}.ini")
    if not os.path.isfile(ini):
        inis = glob.glob(os.path.join(base, "*.ini"))
        if not inis:
            return None, None
        ini = inis[0]
    return pts[0], ini


def find_first_spatial_checkpoint(dataset_dir, method):
    """Spatial layout: <dataset_dir>/<method>/<...method...>.pt and <method>.ini."""
    method_dir = os.path.join(dataset_dir, method)
    if not os.path.isdir(method_dir):
        return None, None
    pts = sorted(glob.glob(os.path.join(method_dir, f"*_{method}_*.pt")))
    if not pts:
        return None, None
    ini = os.path.join(method_dir, f"{method}.ini")
    if not os.path.isfile(ini):
        inis = glob.glob(os.path.join(method_dir, "*.ini"))
        if not inis:
            return None, None
        ini = inis[0]
    return pts[0], ini


def parse_split_from_filename(ckpt_path, dataset_canonical_name):
    """Extract split index from filenames like '..._Loss_<canonical><digit>_MLP_...'."""
    fname = os.path.basename(ckpt_path)
    m = re.search(rf"_Loss_{re.escape(dataset_canonical_name)}(\d+)_", fname)
    return int(m.group(1)) if m else 0


def load_config(ini_path):
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cfg.optionxform = str
    cfg.read(ini_path)
    return cfg


def parameters_from_config(cfg, override_method, override_split=None):
    """Mirror of main.py:135-149 + apply overrides + collapse to single dict."""
    train_dict = {k: ast.literal_eval(v) for k, v in cfg.items("TRAININGPARAMETERS")}
    data_dict = {k: ast.literal_eval(v) for k, v in cfg.items("DATAPARAMETERS")}

    # 'crps_ensemble' isn't a distributional_method value — uq=='crps' selects
    # the CRPS branch in generate_samples and the inner distributional value
    # (set in the config) is irrelevant for sampling. Leave it untouched.
    if override_method != "crps_ensemble":
        train_dict["distributional_method"] = [override_method]
    if override_split is not None:
        data_dict["yarin_gal_uci_split_indices"] = [override_split]

    data_params = process_dict.process_data_parameters(data_dict)
    train_params = process_dict.process_training_parameters(train_dict)

    train_params = train_utils.get_hyperparameters_combination(train_params)[0]
    data_params = train_utils.get_hyperparameters_combination(data_params)[0]
    return data_params, train_params


def build_regressor(train_params, data_params, device, target_dim, input_dim):
    """Replicate main.py's regressor branch — needed when x_T_sampling_method == 'CARD'."""
    reg = train_params.get("regressor", None)
    if reg is None:
        return None
    if reg == "orig_CARD_pretrain":
        dataset_name = data_params["dataset_name"]
        split = data_params["yarin_gal_uci_split_indices"]
        model_path = os.path.join(REPO_ROOT, "models", "orig_CARD_pretrain",
                                  dataset_name, f"split_{split}", "aux_ckpt.pth")
        config_path = os.path.join(REPO_ROOT, "models", "orig_CARD_pretrain",
                                   dataset_name, f"split_{split}", "config.yml")
        aux_states = torch.load(model_path, map_location=device)
        with open(config_path, "r") as f:
            card_cfg = yaml.unsafe_load(f)
        regressor = train_utils.setup_CARD_model(
            image_dim=target_dim,
            label_dim=input_dim,
            hidden_layers=card_cfg.diffusion.nonlinear_guidance.hid_layers,
            use_batchnorm=card_cfg.diffusion.nonlinear_guidance.use_batchnorm,
            negative_slope=card_cfg.diffusion.nonlinear_guidance.negative_slope,
            dropout_rate=card_cfg.diffusion.nonlinear_guidance.dropout_rate,
        ).to(device)
        regressor.load_state_dict(aux_states[0])
        regressor.eval()
        return regressor
    # Other regressor pretrain dirs (model-based) — skip for now; not used by current configs.
    return None


def build_diffusion(train_params, target_dim, device):
    if train_params["uncertainty_quantification"] != "diffusion":
        return None
    return Diffusion(
        noise_steps=train_params["n_timesteps"],
        img_size=target_dim,
        ddim_churn=train_params["ddim_churn"],
        device=device,
        x_T_sampling_method=train_params["x_T_sampling_method"],
        noise_schedule=train_params["noise_schedule"],
        beta_endpoints=train_params["beta_endpoints"],
        tau=train_params["tau"],
    )


NOMINAL_LEVELS = np.arange(0.05, 1.0, 0.05)


class QuantileCoverage(object):
    """Quantile coverage: indicator that the target lies below the alpha-quantile of the prediction.

    For a well-calibrated model and nominal level alpha, exactly alpha of targets
    should be smaller than the predicted alpha-quantile.

    In functional mode (1D PDEs), ``smaller`` is only true when every spatial
    value of the target is smaller than the corresponding quantile of the
    ensemble at that point.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        reduction: str = "mean",
        reduce_dims: bool = True,
        functional: bool = False,
        **kwargs: dict,
    ):
        self.alpha = alpha
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        self.functional = functional

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(
        self, x: torch.Tensor, y: torch.Tensor, ensemble_dim: int = -1
    ) -> torch.Tensor:
        n_dims = len(x.shape) - 2

        q = torch.quantile(x, self.alpha, dim=ensemble_dim)
        assert q.size() == y.size()
        assert 0 < self.alpha < 1

        score = (y < q)
        if self.functional:
            # All spatial points must be below their quantile simultaneously.
            score = torch.all(score.reshape(score.shape[0], -1), dim=-1).float()
        else:
            score = score.float()
            if self.reduce_dims:
                score = score.mean(dim=[d for d in range(1, n_dims + 1)])
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


def compute_coverage_indicators(predictions, target, kind, nominal_levels=NOMINAL_LEVELS, functional=False):
    """Per-example quantile-coverage indicators for each nominal level.

    For each level alpha, the indicator is 1 iff the target lies below the
    alpha-quantile of the predictive ensemble. Under perfect calibration the
    empirical mean across examples should equal alpha.

    UCI: scalar reduction (flattened singleton), uses ensemble quantiles.
    Spatial: delegates to QuantileCoverage. With functional=True (1D PDEs)
    the indicator requires every spatial point to be below its quantile.

    Returns array of shape (batch, len(nominal_levels)) with values in {0, 1}.
    """
    if kind == "spatial":
        per_level = []
        for level in nominal_levels:
            cov = QuantileCoverage(
                alpha=float(level),
                reduction=None,
                reduce_dims=False,
                functional=functional,
            )
            score = cov(predictions, target, ensemble_dim=-1)
            if functional:
                # QuantileCoverage already returns shape (batch,) in functional mode.
                score = score.reshape(score.shape[0], -1)
                if score.shape[-1] != 1:
                    score = score.min(dim=-1).values
                else:
                    score = score.squeeze(-1)
            else:
                score = score.reshape(score.shape[0], -1).mean(dim=-1)
            per_level.append(score.detach().cpu().numpy().astype(np.float64))
        return np.stack(per_level, axis=-1)

    target_red = target.reshape(target.shape[0], -1).squeeze(-1)
    pred_red = predictions.reshape(predictions.shape[0], -1, predictions.shape[-1]).squeeze(1)

    levels = torch.as_tensor(nominal_levels, dtype=pred_red.dtype, device=pred_red.device)
    q = torch.quantile(pred_red, levels, dim=-1)
    below = (target_red.unsqueeze(0) < q)
    below = below.t().contiguous()
    return below.detach().cpu().numpy().astype(np.float64)


def compute_ranks(predictions, target, kind):
    """predictions: (batch, *target_shape, n_samples). target: (batch, *target_shape).

    Returns int ranks in [0, n_samples] of length batch. Ties resolved with a
    uniform random tiebreak.
    """
    if kind == "spatial":
        reduce_dims_t = tuple(range(1, target.ndim))
        reduce_dims_p = tuple(range(1, predictions.ndim - 1))
        target_red = target.mean(dim=reduce_dims_t)
        pred_red = predictions.mean(dim=reduce_dims_p)
    else:
        # UCI: target is (batch, 1) or (batch, 1, 1); predictions are (batch, ..., n_samples).
        target_red = target.reshape(target.shape[0], -1).squeeze(-1)
        pred_red = predictions.reshape(predictions.shape[0], -1, predictions.shape[-1]).squeeze(1)

    n_samples = pred_red.shape[-1]
    less = (pred_red < target_red.unsqueeze(-1)).sum(dim=-1)
    equal = (pred_red == target_red.unsqueeze(-1)).sum(dim=-1)

    tiebreak = torch.zeros_like(less)
    has_tie = equal > 0
    if has_tie.any():
        # Sample uniformly in [0, equal] for indices where there are ties.
        u = torch.rand(equal.shape, device=equal.device)
        tiebreak = (u * (equal.float() + 1.0)).floor().long()
        tiebreak = tiebreak.clamp(max=equal)

    ranks = (less + tiebreak).clamp(min=0, max=n_samples)
    return ranks.detach().cpu().numpy().astype(np.int64)


def run_pair(dataset_key, method, kind, ckpt_path, ini_path, n_samples, out_dir, device):
    print(f"[{dataset_key} / {method}] checkpoint = {os.path.basename(ckpt_path)}")
    cfg = load_config(ini_path)

    # Determine UCI split from filename (canonical dataset name comes from the ini)
    tmp_data_dict = {k: ast.literal_eval(v) for k, v in cfg.items("DATAPARAMETERS")}
    canonical_name = tmp_data_dict["dataset_name"]
    if isinstance(canonical_name, list):
        canonical_name = canonical_name[0]
    split_override = (
        parse_split_from_filename(ckpt_path, canonical_name) if kind == "uci" else None
    )
    functional=False# = kind == "spatial" and canonical_name.startswith("1D")

    data_params, train_params = parameters_from_config(cfg, method, split_override)

    seed = train_params["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = os.path.join(REPO_ROOT, "data")

    target_dim, input_dim, _, _ = get_dataset_metadata(
        data_dir, data_params, train_params, seed,
    )

    regressor = build_regressor(train_params, data_params, device, target_dim, input_dim)
    diffusion = build_diffusion(train_params, target_dim, device)
    beta = diffusion.beta if diffusion is not None else None

    model = train_utils.setup_model(
        data_params, train_params, device, target_dim, input_dim, beta,
    )
    train_utils.resume(model, ckpt_path)
    model.eval()

    _, _, test_dataset = get_datasets(data_dir, data_params, train_params, seed)
    eval_batch_size = int(train_params["eval_batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg_scale = 3 if train_params["conditional_free_guidance_training"] else 0
    all_ranks = []
    all_coverage = []
    with torch.no_grad():
        for target, x in test_loader:
            x = x.to(device)
            target = target.to(device)
            preds = generate_samples(
                uncertainty_quantification=train_params["uncertainty_quantification"],
                model=model,
                n_timesteps=train_params["n_timesteps"],
                x=x,
                target=target,
                n_samples=n_samples,
                x_T_sampling_method=train_params["x_T_sampling_method"],
                distributional_method=train_params["distributional_method"],
                closed_form=train_params["closed_form"],
                regressor=regressor,
                cfg_scale=cfg_scale,
                ddim_churn=train_params["ddim_churn"],
                noise_schedule=train_params["noise_schedule"],
                metrics_plots=False,
                beta_endpoints=train_params["beta_endpoints"],
                tau=train_params["tau"],
            )
            if data_params["standardize"]:
                target = test_loader.dataset.destandardize_output(target)
                preds = test_loader.dataset.destandardize_output(preds)
            all_ranks.append(compute_ranks(preds, target, kind))
            all_coverage.append(compute_coverage_indicators(preds, target, kind, functional=functional))

    ranks = np.concatenate(all_ranks) if all_ranks else np.array([], dtype=np.int64)
    if all_coverage:
        coverage_indicators = np.concatenate(all_coverage, axis=0)
        empirical_coverage = coverage_indicators.mean(axis=0)
    else:
        coverage_indicators = np.zeros((0, len(NOMINAL_LEVELS)), dtype=np.float64)
        empirical_coverage = np.zeros(len(NOMINAL_LEVELS), dtype=np.float64)

    # Pin the trivial endpoints (0% and 100%) so the reliability diagram is complete.
    nominal_levels_full = np.concatenate(([0.0], NOMINAL_LEVELS, [1.0]))
    empirical_coverage = np.concatenate(([0.0], empirical_coverage, [1.0]))

    out_path = os.path.join(out_dir, f"{dataset_key}__{method}__ranks.npz")
    np.savez(
        out_path,
        ranks=ranks,
        dataset=dataset_key,
        method=method,
        checkpoint=os.path.basename(ckpt_path),
        n_samples=n_samples,
        kind=kind,
    )
    print(f"  -> saved {len(ranks)} ranks to {out_path}")

    coverage_path = os.path.join(out_dir, f"{dataset_key}__{method}__coverage.npz")
    np.savez(
        coverage_path,
        nominal_levels=nominal_levels_full,
        empirical_coverage=empirical_coverage,
        coverage_indicators=coverage_indicators,
        dataset=dataset_key,
        method=method,
        checkpoint=os.path.basename(ckpt_path),
        n_samples=n_samples,
        kind=kind,
    )
    print(f"  -> saved coverage at {len(NOMINAL_LEVELS)} levels to {coverage_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset keys; default: all configured.")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help=f"Comma-separated methods; default: {','.join(METHODS)}.")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--out-dir", default=os.path.join("results", "calibration"))
    args = parser.parse_args()

    all_datasets = {**UCI_DATASETS, **SPATIAL_DATASETS}
    if args.datasets:
        wanted = [d.strip() for d in args.datasets.split(",")]
        unknown = [d for d in wanted if d not in all_datasets]
        if unknown:
            raise SystemExit(f"Unknown dataset(s): {unknown}. Known: {list(all_datasets)}")
    else:
        wanted = list(all_datasets.keys())

    methods = [m.strip() for m in args.methods.split(",")]

    out_dir = os.path.join(REPO_ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}.")

    for ds in wanted:
        if ds in UCI_DATASETS:
            kind = "uci"
            base = os.path.join(REPO_ROOT, UCI_DATASETS[ds])
            finder = find_first_uci_checkpoint
        else:
            kind = "spatial"
            base = os.path.join(REPO_ROOT, SPATIAL_DATASETS[ds])
            finder = find_first_spatial_checkpoint

        for method in methods:
            if method == "crps_ensemble":
                ckpt, ini = find_first_crps_ensemble_checkpoint(ds, kind)
                search_root = os.path.join(REPO_ROOT, CRPS_ENSEMBLE_UCI_ROOT, ds)
            else:
                ckpt, ini = finder(base, method)
                search_root = base
            if ckpt is None:
                print(f"[{ds} / {method}] no checkpoint under {search_root} — skipping.")
                continue
            try:
                run_pair(ds, method, kind, ckpt, ini, args.n_samples, out_dir, device)
            except Exception as e:
                print(f"[{ds} / {method}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()

"""Selective prediction (rejection / retention curves) for KS, Burgers, T2M and UCI.

Several models are evaluated and can be compared against each other:

    normal       our distributional diffusion (diagonal Normal head).  Per test
                 example it yields
                     au  aleatoric uncertainty -- variance across the ensemble
                                                 of drawn samples
                     eu  epistemic uncertainty -- the model's predicted noise
                                                 variance accumulated over the
                                                 reverse diffusion trajectory
    mixednormal  the same distributional diffusion with a Normal-mixture head.
                 Its au is again the sample variance; its predicted noise
                 variance is the *marginal* variance of the mixture, i.e. the
                 weighted mean of the component variances plus the spread of
                 the component means, so it is the same quantity as above.
    mvnormal     the same distributional diffusion with a multivariate Normal
                 head (low-rank or Cholesky covariance over the domain).  Only
                 the marginal variances -- the diagonal of the predicted noise
                 covariance -- enter eu; the off-diagonal correlations are
                 ignored, which again makes eu the same per-point quantity as
                 for the diagonal head.
    ensemble     the deterministic diffusion used as a deep ensemble over
                 several training checkpoints.  Its uncertainty is the usual
                 law-of-total-variance split of the pooled sample distribution,
                     au  mean over members of the within-member sample variance
                     eu  variance across the member means (their disagreement)

All of them report tu = au + eu and the mse of their mean prediction against the
target, each averaged over the whole spatial domain, so they are directly
comparable in terms of PRR and of the correlation between their au/eu. Every
uncertainty term is a variance, so au and eu are on the same scale and their
sum is meaningful.

Unlike evaluation/eu.ipynb this does *not* roll the model out autoregressively:
every test example is a single one-step prediction.

Afterwards each uncertainty measure is used to reject the most uncertain
examples. The PRR over the retention curve (mean MSE over the retained examples
vs. the retention rate, on (0.5, 1)) is printed and the raw per-example arrays
are written to results/selective_prediction. The curves themselves are drawn in
evaluation/selective_prediction.ipynb: this script is meant to run on a compute
node, where matplotlib's usetex backend has no complete texmf tree and drawing
anything at all fails.

How EU is collapsed over the reverse-diffusion steps is configurable for the
distributional models: --eu-last-steps K restricts it to the K final (low-noise)
steps, and --eu-reduction max takes the largest value over the retained steps
instead of the mean. Note that under the linear schedule used here the EU weight
peaks at the very last reverse step, so "max" is equivalent to --eu-last-steps 1.
Neither option applies to the ensemble, whose EU has no diffusion-step axis.

Distributional checkpoints
    KS / Burgers  the seed-1 run of each head (results/<dataset>/<head>), with
                  the head hyper-parameters and beta endpoints that run was
                  trained under -- these differ between heads, so each is
                  configured separately below.
    T2M           the single run of each head.
    UCI           the split-0 run of the Normal and the mixture head; no
                  multivariate-Normal model was trained there (the target is a
                  scalar, so it would coincide with the Normal head) and that
                  model kind is skipped.

Ensemble members
    KS / Burgers  the five deterministic checkpoints trained with seeds 1-5 on
                  the same data, so the test set is unseen by every member.
    T2M           likewise five deterministic checkpoints with seeds 1-5, all
                  under results/T2M/deterministic.
    UCI           five deterministic checkpoints trained with seeds 1-5 on
                  Yarin-Gal split 0 (config/CARD_single_split, checkpoints under
                  results/selective_prediction/UCI/<dataset>), so they differ
                  only in initialisation and the split-0 test set is unseen by
                  every member -- the same held-out data the distributional
                  models are evaluated on, and the same split-0 CARD regressor.

Run from the repo root:
    python evaluation/selective_prediction.py                     # all datasets
    python evaluation/selective_prediction.py --datasets KS
    python evaluation/selective_prediction.py --datasets KS --models ensemble
    python evaluation/selective_prediction.py --models mixednormal,mvnormal
    python evaluation/selective_prediction.py --datasets UCI_concrete
    python evaluation/selective_prediction.py --datasets T2M --n-samples 64
    python evaluation/selective_prediction.py --eu-reduction max
"""

import argparse
import glob
import os
import pathlib
import re
import sys

import numpy as np
import torch
import yaml
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import PDE1D, WeatherBench
from data.data_utils import get_uci_data
from models import (
    Diffusion,
    MLP_CARD,
    MLP_diffusion_mixednormal,
    MLP_diffusion_normal,
    UNet_diffusion_mixednormal,
    UNet_diffusion_mvnormal,
    UNet_diffusion_normal,
    UNetDiffusion,
    generate_diffusion_samples_low_dimensional,
)
from models.mlp_diffusion import MLP_diffusion_CARD

NOISE_SCHEDULE = "linear"
DISTRIBUTIONAL_METHOD = "normal"

# The distributional heads, all of which are run by ``run_distributional`` and
# whose name doubles as the sampler's ``distributional_method``.
DISTRIBUTIONAL_KINDS = ("normal", "mixednormal", "mvnormal")
MODEL_KINDS = (*DISTRIBUTIONAL_KINDS, "ensemble")

# Results directory name -> dataset name as used by data/UCI_Datasets and by the
# checkpoint filenames.
UCI_DATASETS = {
    "concrete": "concrete",
    "energy": "energy",
    "kin8nm": "kin8nm",
    "naval": "naval-propulsion-plant",
    "power": "power-plant",
    "protein": "protein-tertiary-structure",
    "wine": "wine-quality-red",
    "yacht": "yacht",
}

# Split whose test set everything is evaluated on.  The deterministic ensemble
# members were all trained on it too (config/CARD_single_split/*.ini, seeds 1-5),
# so they differ only in initialisation and hold out the same test set as the
# distributional model.
UCI_SPLIT = 0
UCI_ENSEMBLE_DIR = "results/selective_prediction/UCI"
UCI_ENSEMBLE_MEMBERS = 5


def uci_entry(results_name, dataset_name):
    """Config for one UCI dataset, mirroring config/concrete.ini.

    All distributional heads and the deterministic model were trained with the
    same hyper-parameters, so they share the CARD backbone, the CARD x_T
    sampling and the (0.001, 0.35) beta endpoints; only the checkpoints and the
    head differ.  The mixture head was trained with three components.
    """
    common = dict(
        x_T_sampling_method="CARD",
        beta_endpoints=(0.001, 0.35),
        regressor="orig_CARD_pretrain",
    )
    return dict(
        kind="uci",
        dataset_name=dataset_name,
        results_name=results_name,
        split=UCI_SPLIT,
        batch_size=1024,
        hidden_dim=64,
        n_layers=2,
        normal=dict(
            ckpt_pattern=f"results/UCI/{results_name}/*/Datetime_*_Loss_"
            f"{dataset_name}{UCI_SPLIT}_MLP_diffusion_normal_T50_DDIM1.pt",
            **common,
        ),
        mixednormal=dict(
            ckpt_pattern=f"results/UCI/{results_name}/*/Datetime_*_Loss_"
            f"{dataset_name}{UCI_SPLIT}_MLP_diffusion_mixednormal_T50_DDIM1.pt",
            n_components=3,
            **common,
        ),
        # The UCI targets are scalars, so a multivariate Normal head would
        # reduce to the diagonal one and none was trained.
        mvnormal=None,
        ensemble=dict(
            # One glob for all members: they share the split index and are told
            # apart only by the timestamp in the filename, which sorts them by
            # training order.
            ckpt_patterns=[
                f"{UCI_ENSEMBLE_DIR}/{results_name}/Datetime_*_Loss_"
                f"{dataset_name}{UCI_SPLIT}_MLP_diffusion_deterministic_T50_DDIM1.pt"
            ],
            n_members=UCI_ENSEMBLE_MEMBERS,
            **common,
        ),
    )


# Per-dataset settings, mirroring evaluation/eu.ipynb and the training configs
# in results/<dataset>/{normal,mixednormal,mvnormal,deterministic}/*.ini.  T2M is
# a 160x220 field, so it needs a much smaller batch than the 1D PDEs.  Every run
# has to be sampled with the beta schedule and the head hyper-parameters it was
# trained under, and those differ per head: the deterministic runs never set
# beta_endpoints and so fell back to the (0.001, 0.35) default, the normal and
# mixture runs pinned (0.001, 0.2), and the mvnormal runs used (0.001, 0.35) on
# KS and T2M but (0.001, 0.2) on Burgers.  The number of mixture components
# likewise differs per dataset.  For KS and Burgers the seed-1 run of each head
# is taken, i.e. the earliest timestamp, matching the normal checkpoint.
DATASETS = {
    "KS": dict(
        kind="pde",
        pde="KS",
        downscaling_factor=1,
        d=1,
        conditioning_dim=3,
        batch_size=128,
        normal=dict(
            ckpt="results/KS/normal/"
            "Datetime_20250831_102648_Loss_1D_KS_UNet_diffusion_normal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.2),
        ),
        mixednormal=dict(
            ckpt="results/KS/mixednormal/Datetime_20250831_190236_"
            "Loss_1D_KS_UNet_diffusion_mixednormal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.2),
            n_components=50,
        ),
        mvnormal=dict(
            ckpt="results/KS/mvnormal/Datetime_20250901_062553_"
            "Loss_1D_KS_UNet_diffusion_mvnormal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.35),
            rank=1,
            mvnormal_method="lora",
        ),
        ensemble=dict(
            ckpt_patterns=["results/KS/deterministic/Datetime_*_"
                           "Loss_1D_KS_UNet_diffusion_deterministic_T50_DDIM1.pt"],
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.35),
        ),
    ),
    "Burgers": dict(
        kind="pde",
        pde="Burgers",
        downscaling_factor=4,
        d=1,
        conditioning_dim=3,
        batch_size=128,
        normal=dict(
            ckpt="results/Burgers/normal/"
            "Datetime_20250829_011108_Loss_1D_Burgers_UNet_diffusion_normal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.2),
        ),
        mixednormal=dict(
            ckpt="results/Burgers/mixednormal/Datetime_20250829_032152_"
            "Loss_1D_Burgers_UNet_diffusion_mixednormal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.2),
            n_components=2,
        ),
        mvnormal=dict(
            ckpt="results/Burgers/mvnormal/Datetime_20250830_022256_"
            "Loss_1D_Burgers_UNet_diffusion_mvnormal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.2),
            rank=1,
            mvnormal_method="lora",
        ),
        ensemble=dict(
            ckpt_patterns=["results/Burgers/deterministic/Datetime_*_"
                           "Loss_1D_Burgers_UNet_diffusion_deterministic_T50_DDIM1.pt"],
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.35),
        ),
    ),
    "T2M": dict(
        kind="weatherbench",
        pde=None,
        downscaling_factor=1,
        d=2,
        conditioning_dim=12,
        batch_size=16,
        normal=dict(
            ckpt="results/T2M/normal/"
            "Datetime_20250908_031720_Loss_WeatherBench_UNet_diffusion_normal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.2),
        ),
        mixednormal=dict(
            ckpt="results/T2M/mixednormal/Datetime_20250909_124553_"
            "Loss_WeatherBench_UNet_diffusion_mixednormal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.2),
            n_components=10,
        ),
        mvnormal=dict(
            ckpt="results/T2M/mvnormal/Datetime_20250911_114427_"
            "Loss_WeatherBench_UNet_diffusion_mvnormal_T50_DDIM1.pt",
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.35),
            rank=1,
            mvnormal_method="lora",
        ),
        # Five deterministic checkpoints trained with seeds 1-5, all gathered in
        # results/T2M/deterministic.  The seed-1 run predates the ``frac<f>``
        # segment the later filenames carry, hence the wildcard in front of
        # ``UNet``; the timestamp still sorts them into seed order.
        ensemble=dict(
            ckpt_patterns=["results/T2M/deterministic/Datetime_*_"
                           "Loss_WeatherBench_*UNet_diffusion_deterministic_T50_DDIM1.pt"],
            n_members=5,
            x_T_sampling_method="standard",
            beta_endpoints=(0.001, 0.35),
        ),
    ),
}

DATASETS.update(
    {f"UCI_{name}": uci_entry(name, dataset) for name, dataset in UCI_DATASETS.items()}
)


def reshape_to_x_sample(parameter, x):
    return parameter.view(*parameter.shape, *(1,) * (x.ndim - parameter.ndim)).expand(
        x.shape
    )


class EUDiffusion(Diffusion):
    """Diffusion sampler that also reports the per-step epistemic uncertainty.

    Identical to models.DistributionalDiffusion except that ``sample_noise``
    additionally returns the variance of the predicted noise, which is
    accumulated over the reverse trajectory into an epistemic-uncertainty
    profile.

    All three distributional heads report the same quantity, the *marginal*
    variance of the predicted noise at each point of the domain: sigma^2 for the
    diagonal Normal, the mixture variance (component variances plus the spread
    of the component means) for the mixture, and the diagonal of the predicted
    covariance for the multivariate Normal, whose off-diagonal correlations are
    ignored. So the EU of the three is on one scale, and on the scale of the
    aleatoric sample variance.

    The UCI models were trained with ``closed_form=True``, which propagates the
    predicted noise covariance analytically through the DDIM update instead of
    drawing the noise and pushing it through. For a diagonal Normal head the two
    give the same per-step distribution -- the closed-form coefficient
    ``A = sqrt(1 - alpha_hat_{t-1} - ddim_sigma^2) - sqrt((1 - alpha_hat)/alpha)``
    is exactly the coefficient the drawn noise picks up here -- and the mixture
    head's closed form draws a component first and then propagates that
    component's diagonal covariance, which is likewise the same distribution as
    drawing a component and then its noise, so this sampler is used for both.
    """

    def __init__(
        self,
        noise_steps=1000,
        noise_schedule="linear",
        img_size=256,
        device="cuda",
        distributional_method="normal",
        closed_form=False,
        x_T_sampling_method="standard",
        ddim_churn=1.0,
        beta_endpoints=(1e-4, 0.02),
        tau=1,
        **kwargs,
    ):
        super().__init__(
            noise_steps=noise_steps,
            noise_schedule=noise_schedule,
            img_size=img_size,
            device=device,
            x_T_sampling_method=x_T_sampling_method,
            ddim_churn=ddim_churn,
            beta_endpoints=beta_endpoints,
            tau=tau,
        )
        self.distributional_method = distributional_method
        self.closed_form = closed_form
        self.tau = tau

    def sample_noise(self, model, x, t, conditioning=None, pred=None):
        if self.distributional_method == "normal":
            predicted_noise = model(x, t, conditioning, pred=pred)
            # Last dim is (mu, sigma). The epistemic signal is the *variance* of
            # the predicted noise, so that it composes with the variance weight
            # applied in sample_low_dimensional and is on the same scale as the
            # aleatoric term (an ensemble variance).
            eu = predicted_noise[..., 1].squeeze() ** 2

            predicted_noise = predicted_noise[..., 0] + np.sqrt(
                self.tau
            ) * predicted_noise[..., 1] * torch.randn_like(
                predicted_noise[..., 0], device=self.device
            )

        elif self.distributional_method == "mvnormal":
            predicted_noise = model(x, t, conditioning, pred=pred)
            if predicted_noise.shape[-1] == predicted_noise.shape[-2] + 1:
                # Cholesky
                mu = predicted_noise[..., 0]
                L_full = np.sqrt(self.tau) * predicted_noise[..., 1:]
                mvnorm = MultivariateNormal(loc=mu, scale_tril=L_full)
            else:  # Lora
                mu = predicted_noise[..., 0]
                diag = self.tau * predicted_noise[..., 1]
                lora = np.sqrt(self.tau) * predicted_noise[..., 2:]
                mvnorm = LowRankMultivariateNormal(mu, lora, diag)
            # ``variance`` is the diagonal of the covariance, i.e. the marginal
            # variance per point of the domain -- the same quantity the diagonal
            # head reports.  The correlations the head predicts on top of it do
            # not enter, since the EU is averaged over the domain anyway.
            eu = mvnorm.variance.squeeze()
            predicted_noise = mvnorm.sample()

        elif self.distributional_method == "mixednormal":
            predicted_mixture = model(x, t, conditioning, pred=pred)
            mu = predicted_mixture[..., 0]
            sigma = np.sqrt(self.tau) * predicted_mixture[..., 1]
            weights = predicted_mixture[..., 2]
            # Variance of the mixture, not the weighted mean of the component
            # scales: the components' means also spread the distribution.
            mixture_mean = (weights * mu).sum(dim=-1, keepdim=True)
            eu = (
                (weights * (sigma**2 + (mu - mixture_mean) ** 2)).sum(dim=-1).squeeze()
            )
            sampled_weights = torch.distributions.Categorical(weights).sample()
            sampled_mu = torch.gather(mu, dim=-1, index=sampled_weights.unsqueeze(-1))
            sampled_sigma = torch.gather(
                sigma, dim=-1, index=sampled_weights.unsqueeze(-1)
            )
            predicted_noise = sampled_mu + sampled_sigma * torch.randn_like(
                sampled_mu, device=self.device
            )
            predicted_noise = predicted_noise.squeeze(-1)
        else:
            raise ValueError(
                f"distributional_method '{self.distributional_method}' has no EU decomposition."
            )
        return predicted_noise, eu

    def sample_low_dimensional(
        self, model, n, conditioning=None, cfg_scale=0, pred=None
    ):
        """Reverse-diffuse and return (sample, eu_profile).

        ``eu_profile`` is [n, noise_steps]: the epistemic uncertainty already
        averaged over the spatial domain. Reducing inside the loop keeps memory
        flat -- retaining the full [n, *domain, noise_steps] field would need
        several GB for the 160x220 T2M grid.
        """
        model.eval()
        eu_profile = torch.zeros(n, self.noise_steps, device=self.device)
        with torch.no_grad():
            x = self.sample_x_T((n, *self.img_size), pred, inference=True)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise, eu = self.sample_noise(model, x, t, conditioning, pred)
                alpha = reshape_to_x_sample(self.alpha[t], x).squeeze()
                alpha_hat = reshape_to_x_sample(self.alpha_hat[t], x).squeeze()
                # The reverse mean carries the predicted noise with coefficient
                # (1 - alpha) / (sqrt(alpha) sqrt(1 - alpha_hat)), so this weight
                # is that coefficient squared: applied to the predicted noise
                # variance it gives the variance this step contributes to x.
                eu_step = (1 - alpha) ** 2 / (alpha * (1 - alpha_hat)) * eu
                eu_profile[:, i] = eu_step.reshape(n, -1).mean(dim=1)
                if cfg_scale > 0:
                    uncond_predicted_noise, _ = self.sample_noise(model, x, t, None, pred)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                x = self.sample_x_t_inference_DDIM(x, t, predicted_noise, pred, i)

        return x, eu_profile


def generate_samples_with_eu(
    model,
    input,
    n_timesteps,
    target_shape,
    n_ensemble,
    pred=None,
    x_T_sampling_method="standard",
    distributional_method=DISTRIBUTIONAL_METHOD,
    cfg_scale=0,
    noise_schedule=NOISE_SCHEDULE,
    beta_endpoints=(0.001, 0.2),
    tau=1,
):
    """Draw ``n_ensemble`` samples and the matching epistemic-uncertainty profile.

    Returns
        samples:    [B, C, *domain, n_ensemble]
        eu_profile: [B, n_timesteps, n_ensemble]
    """
    diffusion = EUDiffusion(
        noise_steps=n_timesteps,
        img_size=target_shape[1:],
        device=input.device,
        distributional_method=distributional_method,
        x_T_sampling_method=x_T_sampling_method,
        noise_schedule=noise_schedule,
        beta_endpoints=beta_endpoints,
        tau=tau,
    )

    samples = torch.zeros(*target_shape, n_ensemble, device=input.device)
    eu_profile = torch.zeros(input.shape[0], n_timesteps, n_ensemble)
    for i in range(n_ensemble):
        with torch.no_grad():
            sample, eu = diffusion.sample_low_dimensional(
                model,
                n=input.shape[0],
                conditioning=input,
                pred=pred,
                cfg_scale=cfg_scale,
            )
            samples[..., i] = sample.detach()
            eu_profile[..., i] = eu.detach().cpu()
    return samples, eu_profile


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def build_dataset(name):
    """Test split for ``name``, configured as in evaluation/eu.ipynb.

    Returns (dataset, target_dim, input_dim); the dims are only needed to build
    the UCI models, so they are None for the field datasets, which read their
    shapes off the dataset itself.
    """
    cfg = DATASETS[name]
    if cfg["kind"] == "weatherbench":
        return (
            WeatherBench(
                var="test",
                downscaling_factor=cfg["downscaling_factor"],
                normalize=True,
            ),
            None,
            None,
        )
    if cfg["kind"] == "uci":
        (_, test_dataset), target_dim, input_dim = get_uci_data(
            cfg["dataset_name"],
            splits=cfg["split"],
            standardize=True,
            validation_ratio=0.0,
        )
        return test_dataset, target_dim, input_dim
    return (
        PDE1D(
            data_dir=os.path.join(REPO_ROOT, "data"),
            pde=cfg["pde"],
            var="test",
            downscaling_factor=cfg["downscaling_factor"],
            normalize=True,
            last_t_steps=2,
            temporal_downscaling_factor=2,
            select_timesteps="random",
        ),
        None,
        None,
    )


def materialize(dataset, seed):
    """Freeze a dataset into a TensorDataset of concrete (target, input) pairs.

    PDE1D with select_timesteps="random" draws the timestep inside __getitem__
    from the global torch RNG, so which example index 17 refers to depends on
    how much randomness the sampler has consumed before the loader reaches it.
    Two model runs would then be evaluated on different data and their
    per-example uncertainties could not be correlated. Drawing all examples up
    front, under a fixed seed and before any sampling, pins them down.
    """
    torch.manual_seed(seed)
    targets, inputs = [], []
    for target, x in dataset:
        targets.append(target)
        inputs.append(x)
    return TensorDataset(torch.stack(targets), torch.stack(inputs))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def resolve_checkpoints(patterns, expected=None):
    """Absolute, sorted checkpoint paths for one or more repo-relative globs.

    Each pattern contributes its own matches in filename order, so a list of
    per-member patterns keeps the member order given by the caller.
    """
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(REPO_ROOT, pattern)))
        if not matches:
            raise FileNotFoundError(f"No checkpoint matches {pattern}")
        paths.extend(matches)
    if expected is not None and len(paths) != expected:
        raise ValueError(
            f"Expected {expected} checkpoints, found {len(paths)}: "
            f"{[os.path.basename(p) for p in paths]}"
        )
    return paths


def build_field_backbone(cfg, target_dim):
    return UNetDiffusion(
        d=cfg["d"],
        conditioning_dim=cfg["conditioning_dim"],
        hidden_channels=64,
        in_channels=1,
        out_channels=1,
        init_features=64,
        domain_dim=target_dim,
    )


def build_uci_backbone(cfg, target_dim, input_dim):
    # concat_condition_diffusion=True doubles the CARD hidden width, and the
    # Normal head then projects from that same doubled width.
    return MLP_diffusion_CARD(
        target_dim=target_dim,
        conditioning_dim=input_dim,
        hidden_dim=2 * cfg["hidden_dim"],
        layers=cfg["n_layers"],
        use_regressor_pred=True,
    )


def load_model(cfg, model_kind, ckpt_path, device, target_dim, input_dim):
    """Instantiate and load one checkpoint, matching utils.train_utils.setup_model.

    ``model_kind`` is one of the distributional heads (see
    ``DISTRIBUTIONAL_KINDS``) or "ensemble" for a single deterministic ensemble
    member, which is the bare backbone.  The head hyper-parameters -- the number
    of mixture components, the covariance rank and parametrisation -- are read
    off the per-dataset config, since they differ between the runs.
    """
    model_cfg = cfg[model_kind]
    if cfg["kind"] == "uci":
        backbone = build_uci_backbone(cfg, target_dim, input_dim)
        if model_kind == "normal":
            model = MLP_diffusion_normal(
                backbone=backbone,
                target_dim=target_dim,
                concat=True,
                hidden_dim=cfg["hidden_dim"],
            )
        elif model_kind == "mixednormal":
            model = MLP_diffusion_mixednormal(
                backbone=backbone,
                target_dim=target_dim,
                concat=True,
                hidden_dim=cfg["hidden_dim"],
                n_components=model_cfg["n_components"],
            )
        elif model_kind == "mvnormal":
            raise ValueError("No multivariate Normal head was trained on UCI.")
        else:
            model = backbone
    else:
        backbone = build_field_backbone(cfg, target_dim)
        if model_kind == "normal":
            model = UNet_diffusion_normal(backbone=backbone, d=cfg["d"], target_dim=1)
        elif model_kind == "mixednormal":
            model = UNet_diffusion_mixednormal(
                backbone=backbone,
                d=cfg["d"],
                target_dim=1,
                n_components=model_cfg["n_components"],
            )
        elif model_kind == "mvnormal":
            # target_dim is (channels, *domain); the head only needs the domain.
            model = UNet_diffusion_mvnormal(
                backbone=backbone,
                d=cfg["d"],
                target_dim=1,
                domain_dim=target_dim[1:],
                rank=model_cfg["rank"],
                method=model_cfg["mvnormal_method"],
            )
        else:
            model = backbone
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def load_uci_regressor(cfg, split, device, target_dim, input_dim):
    """The pre-trained CARD regressor that supplies ``pred`` for the CARD sampler."""
    base = os.path.join(
        REPO_ROOT, "models", "orig_CARD_pretrain", cfg["dataset_name"], f"split_{split}"
    )
    with open(os.path.join(base, "config.yml"), "r") as f:
        card_config = yaml.unsafe_load(f)
    guidance = card_config.diffusion.nonlinear_guidance

    regressor = MLP_CARD(
        input_dim=input_dim,
        target_dim=target_dim,
        hid_layers=guidance.hid_layers,
        use_batchnorm=guidance.use_batchnorm,
        negative_slope=guidance.negative_slope,
        dropout_rate=guidance.dropout_rate,
    ).to(device)
    aux_states = torch.load(os.path.join(base, "aux_ckpt.pth"), map_location=device)
    regressor.load_state_dict(aux_states[0])
    return regressor.eval()


def split_of_checkpoint(path, dataset_name):
    """Yarin-Gal split index encoded in a UCI checkpoint filename."""
    match = re.search(rf"_Loss_{re.escape(dataset_name)}(\d+)_MLP_", os.path.basename(path))
    if match is None:
        raise ValueError(f"Cannot read the split index off {os.path.basename(path)}")
    return int(match.group(1))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def eu_variant_suffix(last_steps, reduction):
    """Filename suffix identifying the EU reduction, empty for the default."""
    parts = []
    if reduction != "mean":
        parts.append(reduction)
    if last_steps is not None:
        parts.append(f"last{last_steps}")
    return ("_" + "_".join(parts)) if parts else ""


def reduce_eu(eu_profile, last_steps=None, reduction="mean"):
    """Collapse a per-example EU profile [N, T] into one scalar per example.

    The reverse diffusion runs from high index to low index, so the steps taken
    *last* -- the low-noise ones that refine the final sample -- sit at the
    front of the profile. ``last_steps`` therefore keeps the leading entries.

    reduction
        "mean"  average over the retained diffusion steps
        "max"   take the largest value over the retained diffusion steps
    """
    profile = eu_profile if last_steps is None else eu_profile[:, :last_steps]
    if reduction == "max":
        return profile.max(axis=1)
    return profile.mean(axis=1)


def domain_mean(field):
    """Average everything but the batch axis, i.e. channels and the domain."""
    return field.mean(dim=tuple(range(1, field.ndim)))


def collect_statistics(model, test_loader, args, device, sampler, regressor=None):
    """Per-example AU, MSE and the raw EU profile over the diffusion steps.

    AU and MSE are averaged over the whole spatial domain. The EU profile keeps
    the diffusion-step axis so that any reduction (see ``reduce_eu``) can be
    derived afterwards without re-running the sampler.
    """
    au, mse, eu_profiles, target_means = [], [], [], []

    for batch_idx, (target, x) in enumerate(test_loader):
        x = x.to(device)
        target = target.to(device)
        pred = None if regressor is None else regressor(x)
        target_means.append(domain_mean(target).cpu().numpy())

        samples, eu_batch = generate_samples_with_eu(
            model=model,
            input=x,
            n_timesteps=args.n_timesteps,
            target_shape=target.shape,
            n_ensemble=args.n_ensemble,
            pred=pred,
            **sampler,
        )

        # Aleatoric: spread of the ensemble, averaged over the domain.  [B]
        au_batch = domain_mean(samples.var(dim=-1))

        # EU profile: average the per-member profiles over the ensemble.  [B, T-1]
        # Index 0 of the diffusion axis is never written (the reverse loop stops
        # at 1), so it is dropped rather than averaged in as a structural zero.
        eu_profile = eu_batch[:, 1:, :].mean(dim=2)

        mse_batch = domain_mean((samples.mean(dim=-1) - target) ** 2)

        au.append(au_batch.cpu().numpy())
        mse.append(mse_batch.cpu().numpy())
        eu_profiles.append(eu_profile.cpu().numpy())

        eu_shown = reduce_eu(eu_profiles[-1], args.eu_last_steps, args.eu_reduction)
        # One-step residuals are small in standardised units, so these are
        # printed in scientific notation.
        print(
            f"  batch {batch_idx + 1}/{len(test_loader)}  "
            f"mse={mse[-1].mean():.3e}  au={au[-1].mean():.3e}  eu={eu_shown.mean():.3e}",
            flush=True,
        )

    return (
        np.concatenate(au),
        np.concatenate(mse),
        np.concatenate(eu_profiles),
        np.concatenate(target_means),
    )


def collect_ensemble_statistics(
    models, test_loader, args, device, sampler, regressors=None, n_per_member=1
):
    """Per-example AU, EU and MSE for the deterministic diffusion as an ensemble.

    Every member draws ``n_per_member`` samples; the law of total variance then
    splits the variance of the pooled sample distribution into the mean of the
    within-member variances (aleatoric) and the variance of the member means
    (epistemic, i.e. how much the independently trained checkpoints disagree).
    All three, and the MSE of the pooled mean, are averaged over the domain.
    """
    if regressors is None:
        regressors = [None] * len(models)
    au, eu, mse, target_means = [], [], [], []

    for batch_idx, (target, x) in enumerate(test_loader):
        x = x.to(device)
        target = target.to(device)
        target_means.append(domain_mean(target).cpu().numpy())

        member_means, member_vars = [], []
        for model, regressor in zip(models, regressors):
            with torch.no_grad():
                samples = generate_diffusion_samples_low_dimensional(
                    model=model,
                    input=x,
                    n_timesteps=args.n_timesteps,
                    target_shape=target.shape,
                    n_samples=n_per_member,
                    distributional_method="deterministic",
                    regressor=regressor,
                    cfg_scale=0,
                    ddim_churn=1.0,
                    noise_schedule=sampler["noise_schedule"],
                    x_T_sampling_method=sampler["x_T_sampling_method"],
                    beta_endpoints=sampler["beta_endpoints"],
                )
            member_means.append(samples.mean(dim=-1))
            member_vars.append(samples.var(dim=-1))

        # [M, B, C, *domain]
        member_means = torch.stack(member_means, dim=0)
        member_vars = torch.stack(member_vars, dim=0)

        au_batch = domain_mean(member_vars.mean(dim=0))
        eu_batch = domain_mean(member_means.var(dim=0))
        mse_batch = domain_mean((member_means.mean(dim=0) - target) ** 2)

        au.append(au_batch.cpu().numpy())
        eu.append(eu_batch.cpu().numpy())
        mse.append(mse_batch.cpu().numpy())

        print(
            f"  batch {batch_idx + 1}/{len(test_loader)}  "
            f"mse={mse[-1].mean():.3e}  au={au[-1].mean():.3e}  eu={eu[-1].mean():.3e}",
            flush=True,
        )

    return (
        np.concatenate(au),
        np.concatenate(eu),
        np.concatenate(mse),
        np.concatenate(target_means),
    )


def retention_curve(uncertainty, error, retention_grid):
    """Mean error over the most-confident ``r`` fraction of examples, for each r.

    Examples are ranked by ``uncertainty`` ascending; at retention rate r the
    ``1 - r`` most uncertain examples are rejected.
    """
    order = np.argsort(uncertainty, kind="stable")
    error_sorted = error[order]
    n = len(error_sorted)
    return np.array(
        [error_sorted[: max(1, int(round(r * n)))].mean() for r in retention_grid]
    )


def prediction_rejection_ratio(curve, oracle, random_curve, retention_grid):
    """PRR summary: 1 = matches the oracle ranking, 0 = no better than random."""
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    ar_method = area(curve, retention_grid)
    ar_oracle = area(oracle, retention_grid)
    ar_random = area(random_curve, retention_grid)
    denom = ar_random - ar_oracle
    return float("nan") if denom == 0 else (ar_random - ar_method) / denom


def build_curves(au, eu, mse, retention_grid):
    """Retention curves for the three measures plus the oracle and random baselines."""
    curves = {
        "Total uncertainty": retention_curve(au + eu, mse, retention_grid),
        "Aleatoric uncertainty": retention_curve(au, mse, retention_grid),
        "Epistemic uncertainty": retention_curve(eu, mse, retention_grid),
    }
    # Oracle ranks by the realised error itself; random rejection leaves the
    # expected error unchanged at every retention rate.
    oracle = retention_curve(mse, mse, retention_grid)
    random_curve = np.full_like(retention_grid, mse.mean())
    return curves, oracle, random_curve


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def prepare_test_data(dataset, args):
    """Test loader plus the shapes and example count for one dataset."""
    cfg = DATASETS[dataset]
    test_dataset, target_dim, input_dim = build_dataset(dataset)
    if target_dim is None:
        target_dim = (1, *test_dataset.get_dimensions())

    n_used = len(test_dataset)
    if args.n_samples is not None and args.n_samples < n_used:
        n_used = args.n_samples
        test_dataset = Subset(test_dataset, range(n_used))

    # Only PDE1D has a stochastic __getitem__; freezing the others would just
    # cost memory (a whole T2M test set is several GB).
    if cfg["kind"] == "pde":
        test_dataset = materialize(test_dataset, args.seed)

    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, target_dim, input_dim, n_used, batch_size


def sampler_settings(model_cfg, model_kind=None):
    """Sampler kwargs for one model.

    ``model_kind`` names the distributional head and is passed on as the
    sampler's ``distributional_method``; it is omitted for the ensemble, which
    is sampled deterministically.
    """
    settings = dict(
        x_T_sampling_method=model_cfg["x_T_sampling_method"],
        beta_endpoints=model_cfg["beta_endpoints"],
        noise_schedule=NOISE_SCHEDULE,
    )
    if model_kind in DISTRIBUTIONAL_KINDS:
        settings["distributional_method"] = model_kind
    return settings


def run_distributional(
    dataset, model_kind, args, device, loader, target_dim, input_dim, n_used
):
    """One distributional head: sample-variance AU and predicted-noise-variance EU."""
    cfg = DATASETS[dataset]
    model_cfg = cfg[model_kind]

    ckpt = args.ckpt or model_cfg.get("ckpt")
    if ckpt is None:
        ckpt = resolve_checkpoints([model_cfg["ckpt_pattern"]], expected=1)[0]
    ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join(REPO_ROOT, ckpt)
    model = load_model(cfg, model_kind, ckpt_path, device, target_dim, input_dim)
    print(f"Loaded checkpoint {os.path.basename(ckpt_path)}.")

    regressor = None
    if model_cfg.get("regressor") == "orig_CARD_pretrain":
        regressor = load_uci_regressor(cfg, cfg["split"], device, target_dim, input_dim)

    au, mse, eu_profile, target_mean = collect_statistics(
        model, loader, args, device, sampler_settings(model_cfg, model_kind), regressor
    )
    eu = reduce_eu(eu_profile, args.eu_last_steps, args.eu_reduction)

    extra = dict(
        eu_profile=eu_profile,
        target_mean=target_mean,
        eu_last_steps=(-1 if args.eu_last_steps is None else args.eu_last_steps),
        eu_reduction=args.eu_reduction,
        checkpoint=os.path.basename(ckpt_path),
    )
    return au, eu, mse, extra


def run_ensemble(dataset, args, device, loader, target_dim, input_dim, n_used):
    cfg = DATASETS[dataset]
    model_cfg = cfg["ensemble"]

    ckpt_paths = resolve_checkpoints(
        model_cfg["ckpt_patterns"], expected=model_cfg.get("n_members")
    )
    models = [
        load_model(cfg, "ensemble", path, device, target_dim, input_dim)
        for path in ckpt_paths
    ]

    regressors = None
    if model_cfg.get("regressor") == "orig_CARD_pretrain":
        # Each member is sampled with the regressor of the split it was trained
        # on, which the checkpoint filename records; the members all sit on
        # split 0, so this is the same regressor the distributional model uses.
        regressors = [
            load_uci_regressor(
                cfg, split_of_checkpoint(path, cfg["dataset_name"]), device,
                target_dim, input_dim,
            )
            for path in ckpt_paths
        ]

    n_per_member = args.n_ensemble_per_member
    if n_per_member is None:
        # Match the distributional model's total sample budget.
        n_per_member = max(1, args.n_ensemble // len(models))

    print(
        f"Loaded {len(models)} ensemble members, {n_per_member} samples each "
        f"({len(models) * n_per_member} in total):"
    )
    for path in ckpt_paths:
        print(f"  {os.path.basename(path)}")

    au, eu, mse, target_mean = collect_ensemble_statistics(
        models,
        loader,
        args,
        device,
        sampler_settings(model_cfg, "ensemble"),
        regressors=regressors,
        n_per_member=n_per_member,
    )
    extra = dict(
        target_mean=target_mean,
        checkpoint=os.path.basename(ckpt_paths[0]),
        checkpoints=np.array([os.path.basename(p) for p in ckpt_paths]),
        n_members=len(models),
        n_samples_per_member=n_per_member,
    )
    return au, eu, mse, extra


def run_dataset(dataset, model_kind, args, device):
    cfg = DATASETS[dataset]
    print(f"\n=== {dataset} [{model_kind}] ===")

    if cfg.get(model_kind) is None:
        print(f"[{dataset}] no '{model_kind}' model configured -- skipping.")
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loader, target_dim, input_dim, n_used, batch_size = prepare_test_data(dataset, args)
    print(
        f"Evaluating {n_used} test examples with {args.n_ensemble} samples each "
        f"(batch {batch_size})."
    )

    if model_kind == "ensemble":
        au, eu, mse, extra = run_ensemble(
            dataset, args, device, loader, target_dim, input_dim, n_used
        )
    else:
        au, eu, mse, extra = run_distributional(
            dataset, model_kind, args, device, loader, target_dim, input_dim, n_used
        )
    tu = au + eu

    retention_grid = np.linspace(0.5, 1.0, 51)
    curves, oracle, random_curve = build_curves(au, eu, mse, retention_grid)

    print(f"\nMean MSE over {n_used} examples: {mse.mean():.3e}")
    print("PRR over retention in [0.5, 1] (1 = oracle, 0 = random):")
    for label, curve in curves.items():
        prr = prediction_rejection_ratio(curve, oracle, random_curve, retention_grid)
        print(f"  {label:<24s} {prr: .4f}")

    # The EU reduction only exists for the distributional models, so the
    # ensemble files never carry a variant suffix.
    suffix = (
        eu_variant_suffix(args.eu_last_steps, args.eu_reduction)
        if model_kind in DISTRIBUTIONAL_KINDS
        else ""
    )

    if args.save_npz:
        npz_dir = (
            args.save_npz
            if os.path.isabs(args.save_npz)
            else os.path.join(REPO_ROOT, args.save_npz)
        )
        pathlib.Path(npz_dir).mkdir(parents=True, exist_ok=True)
        npz_path = os.path.join(npz_dir, f"{dataset}_{model_kind}{suffix}.npz")
        # ``extra`` carries the model-specific fields: the checkpoint name(s),
        # ``target_mean`` (a per-example fingerprint of the evaluated targets, so
        # two runs can be checked to have seen the same examples before their
        # uncertainties are correlated) and, for the distributional model, the
        # full EU profile over the diffusion steps, from which other reductions
        # can be derived without re-running the sampler.
        np.savez(
            npz_path,
            tu=tu,
            au=au,
            eu=eu,
            mse=mse,
            retention_grid=retention_grid,
            dataset=dataset,
            model=model_kind,
            n_ensemble=args.n_ensemble,
            **extra,
        )
        print(f"-> saved per-example arrays to {npz_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help=f"Comma-separated dataset keys; default: {','.join(DATASETS)}.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_KINDS),
        help="Comma-separated model kinds to evaluate: normal, mixednormal and/or "
        "mvnormal (our distributional diffusion with the respective head) and/or "
        "ensemble (deterministic diffusion over several checkpoints); default: "
        f"{','.join(MODEL_KINDS)}.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of test examples to evaluate; default: the full test set.",
    )
    parser.add_argument(
        "--n-ensemble",
        type=int,
        default=100,
        help="Diffusion samples drawn per test example.",
    )
    parser.add_argument(
        "--n-ensemble-per-member",
        type=int,
        default=None,
        help="Samples drawn per ensemble member; default: --n-ensemble divided by "
        "the number of members, so both models get the same sample budget.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the per-dataset default batch size.",
    )
    parser.add_argument("--n-timesteps", type=int, default=50)
    parser.add_argument(
        "--eu-last-steps",
        type=int,
        default=None,
        help="Use only the last K reverse-diffusion steps (the low-noise ones) "
        "for EU; default: all steps. Distributional models only.",
    )
    parser.add_argument(
        "--eu-reduction",
        choices=["mean", "max"],
        default="mean",
        help="How to collapse EU over the diffusion steps, after averaging over "
        "the domain and the ensemble; default: mean. Distributional models only.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Override the distributional checkpoint; only valid with a single "
        "--datasets entry and a single distributional --models entry.",
    )
    parser.add_argument(
        "--save-npz",
        default=os.path.join("results", "selective_prediction"),
        help="Directory for the raw per-example arrays; empty string disables.",
    )
    args = parser.parse_args()

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in wanted if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Known: {list(DATASETS)}")

    model_kinds = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown_models = [m for m in model_kinds if m not in MODEL_KINDS]
    if unknown_models:
        raise SystemExit(
            f"Unknown model kind(s): {unknown_models}. Known: {list(MODEL_KINDS)}"
        )
    if args.ckpt and (
        len(wanted) != 1
        or len(model_kinds) != 1
        or model_kinds[0] not in DISTRIBUTIONAL_KINDS
    ):
        raise SystemExit(
            "--ckpt requires exactly one --datasets entry and exactly one "
            f"distributional --models entry ({', '.join(DISTRIBUTIONAL_KINDS)})."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}.")

    steps_desc = (
        "all steps" if args.eu_last_steps is None else f"last {args.eu_last_steps} steps"
    )
    print(f"EU reduction: {args.eu_reduction} over {steps_desc} (distributional models).")

    for dataset in wanted:
        for model_kind in model_kinds:
            try:
                run_dataset(dataset, model_kind, args, device)
            except Exception as e:
                print(f"[{dataset}/{model_kind}] FAILED: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()

import torch
import numpy as np
from scoringrules import crps_ensemble
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal


def reshape_to_x_sample(parameter, x):
    return parameter.view(*parameter.shape, *(1,) * (x.ndim - parameter.ndim)).expand(
        x.shape
    )


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        img_size=256,
        device="cuda",
        x_T_sampling_method="standard",
        ddim_churn=1.0,
        noise_schedule="linear",
        beta_endpoints=(1e-4, 0.02),
        tau = 1, # Scaling factor for learned covariance
        variance_method="fixed_ddim",
        gamma_t=None,
    ):
        self.device = device

        self.noise_steps = noise_steps
        self.noise_schedule = noise_schedule

        self.prepare_noise_schedule(beta_start=beta_endpoints[0], beta_end=beta_endpoints[1])

        self.img_size = img_size
        self.x_T_sampling_method = x_T_sampling_method
        self.ddim_churn = ddim_churn
        self.tau = tau
        # Reverse-process variance:
        #   "fixed_ddim"         — original DDIM sigma (default)
        #   "analytic_dpm"       — scalar Gamma_t per timestep (Bao et al. 2022)
        #   "analytic_dpm_diag"  — diagonal Gamma_t (per timestep, per dim)
        #   "ocm"                — per-input, per-dim variance from a learned head
        # gamma_t: cached E[||eps_theta||^2 / d] or E[eps_theta_i^2] estimator.
        #   Shape (T,) for "analytic_dpm", (T, d) for "analytic_dpm_diag".
        self.variance_method = variance_method
        if gamma_t is not None and not torch.is_tensor(gamma_t):
            gamma_t = torch.as_tensor(gamma_t, dtype=torch.float32)
        self.gamma_t = gamma_t.to(self.device) if gamma_t is not None else None
        # Precompute the DDPM posterior variance schedule beta_tilde_t once.
        # beta_tilde_1 = 0 by convention; we use t-1 index safely.
        alpha_hat_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alpha_hat[:-1]], dim=0
        )
        self.beta_tilde = (1.0 - alpha_hat_prev) / (1.0 - self.alpha_hat) * self.beta

    def _reverse_variance(self, t, x_shape, predicted_moment=None):
        """Return the reverse-process variance sigma^2 for the given timesteps.

        Returns a tensor shape-broadcastable to ``x_shape`` (i.e. same shape as x).

        Fallback behavior: "analytic_*" without a cached gamma_t falls back to
        beta_tilde (the DDPM lower bound), and logs no error — this matters
        during training/validation before the post-hoc Gamma_t estimation runs.
        """
        beta_t = reshape_to_x_sample(self.beta[t], torch.zeros(x_shape, device=self.device))
        beta_tilde_t = reshape_to_x_sample(self.beta_tilde[t], torch.zeros(x_shape, device=self.device))

        if self.variance_method == "fixed_ddim":
            raise RuntimeError("_reverse_variance called for fixed_ddim path")

        if self.variance_method in ("analytic_dpm", "analytic_dpm_diag"):
            if self.gamma_t is None:
                return beta_tilde_t  # fallback until estimator has run
            if self.variance_method == "analytic_dpm":
                gamma = self.gamma_t[t]  # (B,)
                gamma = reshape_to_x_sample(
                    gamma, torch.zeros(x_shape, device=self.device)
                )
            else:
                # gamma_t is (T, d_flat) — index by t then reshape to match x
                gamma = self.gamma_t[t]  # (B, d_flat)
                gamma = gamma.reshape(x_shape[0], *x_shape[1:])
            sigma_sq = beta_tilde_t + (beta_t - beta_tilde_t) * (1.0 - gamma).clamp(0.0, 1.0)
        elif self.variance_method == "ocm":
            assert predicted_moment is not None, (
                "OCM variance requires the model head's per-input moment prediction"
            )
            # predicted_moment estimates E[eps^2 | x_t] per dim (softplus output).
            # Analytic-DPM diagonal formula: sigma^2 = beta_tilde + (beta - beta_tilde) * (1 - m).
            sigma_sq = beta_tilde_t + (beta_t - beta_tilde_t) * (1.0 - predicted_moment).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown variance_method: {self.variance_method}")

        # Clip so DDIM's sqrt(1 - alpha_hat_{t-1} - sigma^2) stays real.
        alpha_hat_prev_t = reshape_to_x_sample(
            torch.cat([torch.ones(1, device=self.device), self.alpha_hat[:-1]], dim=0)[t],
            torch.zeros(x_shape, device=self.device),
        )
        upper = torch.minimum(beta_t, 1.0 - alpha_hat_prev_t - 1e-8)
        sigma_sq = torch.clamp(sigma_sq, min=beta_tilde_t, max=upper.clamp_min(0.0))
        return sigma_sq

    def sample_x_T(self, shape, pred, inference=True):
        if self.x_T_sampling_method in ["standard"]:
            x = torch.randn(shape).to(self.device)
        elif self.x_T_sampling_method == "naive-regressor-mean":
            x = torch.randn(shape).to(self.device) + pred
        elif self.x_T_sampling_method == "CARD":
            if inference:
                x = torch.randn(shape).to(self.device) + pred
            else:
                x = torch.randn(shape).to(self.device)
        else:
            raise NotImplementedError(
                f'Please choose as the x_T_sampling_method "standard", "CARD", or "naive-regressor-mean". You chose'
                f"{self.x_T_sampling_method}"
            )
        return x

    def sample_x_t_inference_DDIM(self, x, t, predicted_noise, pred, i, predicted_moment=None):
        alpha = reshape_to_x_sample(self.alpha[t], x)
        alpha_hat = reshape_to_x_sample(self.alpha_hat[t], x)
        if pred is None:
            pred = 0

        x_0_hat = (
            x
            - torch.sqrt(1 - alpha_hat) * predicted_noise
            - (1 - torch.sqrt(alpha_hat)) * pred  # CARD only
        ) / torch.sqrt(alpha_hat)  # DDIM eq. 9

        if i > 1:
            alpha_hat_t_minus_1 = reshape_to_x_sample(self.alpha_hat[t - 1], x)
            if self.variance_method == "fixed_ddim":
                ddim_sigma = (
                    self.ddim_churn
                    * torch.sqrt((1 - alpha_hat_t_minus_1) / (1 - alpha_hat))
                    * torch.sqrt(1 - alpha)
                )# * np.sqrt(self.tau)
            else:
                # Analytic-DPM / OCM: sigma^2 comes from the learned/estimated
                # optimal reverse variance. tau is baked into the sqrt below.
                sigma_sq = self._reverse_variance(t, x.shape, predicted_moment=predicted_moment)
                ddim_sigma = torch.sqrt(sigma_sq)

            if self.x_T_sampling_method == "standard":
                predicted_noise_ddim = predicted_noise
            elif self.x_T_sampling_method == "CARD":
                predicted_noise_ddim = (
                    predicted_noise
                    + (1 - torch.sqrt(alpha_hat)) / (torch.sqrt(1 - alpha_hat)) * pred
                )
            else:
                raise NotImplementedError(
                    f'Please choose as the x_T_sampling_method "standard" or "CARD". You chose'
                    f"{self.x_T_sampling_method}"
                )
            reverse_posterior_mean = (
                torch.sqrt(alpha_hat_t_minus_1) * x_0_hat
                + torch.sqrt(1 - alpha_hat_t_minus_1 - ddim_sigma**2)
                * predicted_noise_ddim
            )
        else:
            alpha_hat_t_minus_1 = torch.ones_like(alpha_hat)
            ddim_sigma = (
                self.ddim_churn
                * torch.sqrt((1 - alpha_hat_t_minus_1) / (1 - alpha_hat))
                * torch.sqrt(1 - alpha)
            )

            reverse_posterior_mean = x_0_hat

        noise = torch.randn_like(x)
        new_x = reverse_posterior_mean + np.sqrt(self.tau) * ddim_sigma * noise

        return new_x

    def sample_x_t_inference_DDPM(self, x, t, predicted_noise, pred, i):
        """
        Deprecated. Use sample_x_t_inference_DDIM instead.
        """
        alpha = self.alpha[t]
        alpha_hat = self.alpha_hat[t]
        beta = self.beta[t]
        # Reshape
        alpha = alpha.view(*alpha.shape, *(1,) * (x.ndim - alpha.ndim)).expand(x.shape)
        alpha_hat = alpha_hat.view(
            *alpha_hat.shape, *(1,) * (x.ndim - alpha_hat.ndim)
        ).expand(x.shape)
        beta = beta.view(*beta.shape, *(1,) * (x.ndim - beta.ndim)).expand(x.shape)
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        if self.x_T_sampling_method in ["standard", "naive-regressor-mean"]:
            x = (
                1
                / torch.sqrt(alpha)
                * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                + torch.sqrt(beta) * noise
            )
        elif self.x_T_sampling_method == "CARD":
            y_hat_0 = (
                1
                / torch.sqrt(alpha_hat)
                * (
                    x
                    - (1 - torch.sqrt(alpha_hat)) * pred
                    - torch.sqrt(1 - alpha_hat) * predicted_noise
                )
            )
            if i > 1:
                alpha_hat_t_minus_1 = self.alpha_hat[t - 1]
                alpha_hat_t_minus_1 = alpha_hat_t_minus_1.view(
                    *alpha_hat_t_minus_1.shape,
                    *(1,) * (x.ndim - alpha_hat_t_minus_1.ndim),
                ).expand(x.shape)

                gamma_0 = beta * torch.sqrt(alpha_hat_t_minus_1) / (1 - alpha_hat)
                gamma_1 = (
                    (1 - alpha_hat_t_minus_1) * torch.sqrt(alpha) / (1 - alpha_hat)
                )
                gamma_2 = 1 + (torch.sqrt(alpha_hat) - 1) * (
                    torch.sqrt(alpha) + torch.sqrt(alpha_hat_t_minus_1)
                ) / (1 - alpha_hat)

                beta_wiggle = (1 - alpha_hat_t_minus_1) / (1 - alpha_hat) * beta

                new_x = (
                    gamma_0 * y_hat_0
                    + gamma_1 * x
                    + gamma_2 * pred
                    + torch.sqrt(beta_wiggle) * noise
                )

                # new_x = (
                #     1 / torch.sqrt(alpha)
                #     * (
                #         x
                #         - (1 - torch.sqrt(alpha)) * pred
                #         -  ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                #     ) + torch.sqrt(beta_wiggle) * noise
                # )

                x = new_x

            else:
                x = y_hat_0
        else:
            raise NotImplementedError(
                f'Please choose as the x_T_sampling_method "standard", "CARD", or "naive-regressor-mean". You chose'
                f"{self.x_T_sampling_method}"
            )
        return x

    def sample_x_t_training(self, x, eps, t, pred=None):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        # Reshape
        sqrt_alpha_hat = sqrt_alpha_hat.view(
            *sqrt_alpha_hat.shape, *(1,) * (x.ndim - sqrt_alpha_hat.ndim)
        ).expand(x.shape)
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.view(
            *sqrt_one_minus_alpha_hat.shape,
            *(1,) * (x.ndim - sqrt_one_minus_alpha_hat.ndim),
        ).expand(x.shape)
        if self.x_T_sampling_method in ["standard", "naive-regressor-mean"]:
            target_training = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        elif self.x_T_sampling_method == "CARD":
            target_training = (
                sqrt_alpha_hat * x
                + sqrt_one_minus_alpha_hat * eps
                + (1 - sqrt_alpha_hat) * pred
            )
        return target_training

    def prepare_noise_schedule(self, beta_start = 1e-4, beta_end = 0.02):
        if self.noise_schedule == "linear":            
            self.beta = torch.linspace(beta_start, beta_end, self.noise_steps).to(
                self.device
            )
        elif self.noise_schedule == "cosine":
            t = torch.arange(0, self.noise_steps + 1, device=self.device)

            def f(t, T, s):
                return torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2

            T = self.noise_steps
            s = 0.008  # from improved DDPM paper, might try to adjust it, as the value is motivated by the pixel bin size
            alpha_bar = f(t, T, s) / f(torch.tensor([0], device=self.device), T, s)
            self.beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        else:
            raise ValueError(
                f'Noise schedule must be "linear" or "cosine". You chose "{self.noise_schedule}".'
            )
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_low_dimensional(self, x, t, pred=None):
        assert (self.x_T_sampling_method == "standard") or not (pred is None)

        # inference = False since this method is only used during training
        eps = self.sample_x_T(x.shape, pred, inference=False)
        x_t = self.sample_x_t_training(x, eps, t, pred)
        return x_t, eps

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @torch.no_grad()
    def estimate_gamma_t(
        self,
        model,
        dataloader,
        regressor=None,
        n_mc_batches=None,
        per_dim=False,
    ):
        """Monte-Carlo estimate of the Analytic-DPM confidence term Gamma_t.

        Gamma_t = E_{x_0, eps}[||eps_theta(x_t, t)||^2 / d]        (scalar per t)
              or E_{x_0, eps}[eps_theta_i^2] per dim i             (per_dim=True)

        Loops over the dataloader for each t and averages. For UCI (d small,
        T=50, N_train up to a few thousand) one pass over the training set per
        timestep is more than enough. If n_mc_batches is set, we cap the number
        of batches used per timestep.
        """
        model_was_training = model.training
        model.eval()

        # Peek at target dim from one batch (target is (B, 1, d) or (B, d)).
        target_dim = None
        for target, _ in dataloader:
            target_dim = target[0].numel()
            break
        if target_dim is None:
            raise ValueError("Dataloader is empty; cannot estimate gamma_t")

        T = self.noise_steps
        if per_dim:
            gamma = torch.zeros(T, target_dim, device=self.device)
        else:
            gamma = torch.zeros(T, device=self.device)

        # Cache all batches once
        batches = []
        for target, input_ in dataloader:
            batches.append((target.to(self.device), input_.to(self.device)))
            if n_mc_batches is not None and len(batches) >= n_mc_batches:
                break

        for t_val in range(1, T):
            sq_sum = None
            n_seen = 0
            for target, input_ in batches:
                B = target.shape[0]
                t = torch.full((B,), t_val, dtype=torch.long, device=self.device)
                pred = regressor(input_) if regressor is not None else None
                x_t, _eps = self.noise_low_dimensional(target, t, pred=pred)
                out = model(x_t, t, input_, pred=pred)
                # Distributional heads return (..., 2) with mu at [...,0]; for
                # deterministic backbone, out is the noise prediction directly.
                if out.dim() > target.dim() and out.shape[-1] in (2, 3):
                    eps_pred = out[..., 0]
                else:
                    eps_pred = out
                eps_flat = eps_pred.reshape(B, -1)
                if per_dim:
                    sq = (eps_flat ** 2).sum(dim=0)  # (d,)
                else:
                    sq = (eps_flat ** 2).sum() / eps_flat.shape[1]  # scalar
                sq_sum = sq if sq_sum is None else sq_sum + sq
                n_seen += B
            gamma[t_val] = sq_sum / n_seen

        # t=0 slot is unused by the sampler (i=1 branch uses fixed sigma=0);
        # leave it at 0. Clip to [0, 1] since 1 - gamma should be a valid weight.
        gamma = gamma.clamp(0.0, 1.0)
        self.gamma_t = gamma
        if model_was_training:
            model.train()
        return gamma

    # def sample_low_dimensional(
    #     self, model, n, conditioning=None, cfg_scale=3, pred=None
    # ):
    #     assert (self.x_T_sampling_method == "standard") or not (pred is None)

    #     model.eval()
    #     with torch.no_grad():
    #         x = self.sample_x_T((n, *self.img_size), pred, inference=True)
    #         for i in reversed(range(1, self.noise_steps)):
    #             t = (torch.ones(n) * i).long().to(self.device)
    #             predicted_noise = model(x, t, conditioning, pred=pred)
    #             if cfg_scale > 0:
    #                 uncond_predicted_noise = model(x, t, None, pred=pred)
    #                 predicted_noise = torch.lerp(
    #                     uncond_predicted_noise, predicted_noise, cfg_scale
    #                 )

    #             x = self.sample_x_t_inference_DDIM(x, t, predicted_noise, pred, i)

    #     model.train()
    #     return x

    def sample_low_dimensional(self, model, n, conditioning=None, cfg_scale=3, pred=None):
        assert (self.x_T_sampling_method == "standard") or not (pred is None)

        model.eval()
        device = next(model.parameters()).device  # DDP-safe device
        with torch.no_grad():
            # Make sure x starts on the correct device
            x = self.sample_x_T((n, *self.img_size), pred, inference=True).to(device)

            # Move conditioning and pred if provided
            if conditioning is not None:
                conditioning = conditioning.to(device)
            if pred is not None:
                pred = pred.to(device)

            for i in reversed(range(1, self.noise_steps)):
                t = torch.full((n,), i, dtype=torch.long, device=device)

                # Predict noise
                predicted_noise = model(x, t, conditioning, pred=pred)

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None, pred=pred)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                # Sample next step
                x = self.sample_x_t_inference_DDIM(x, t, predicted_noise, pred, i)

        model.train()
        return x


    def sample_images(self, model, n, labels=None, cfg_scale=3):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def generate_diffusion_samples_low_dimensional(
    model,
    input,
    n_timesteps,
    target_shape,
    n_samples,
    x_T_sampling_method,
    distributional_method="deterministic",
    closed_form=False,
    regressor=None,
    cfg_scale=3,
    gt_target=None,
    ddim_churn=1.0,
    noise_schedule=None,
    metrics_plots=False,
    beta_endpoints=(1e-4, 0.02),
    tau = 1,
    variance_method="fixed_ddim",
    gamma_t=None,
):
    # OCM requires the DistributionalDiffusion path so sample_low_dimensional
    # can dispatch to the OCM branch (deterministic Diffusion has no OCM branch).
    use_distributional = (distributional_method != "deterministic") or (variance_method == "ocm")
    if not use_distributional:
        diffusion = Diffusion(
            noise_steps=n_timesteps,
            img_size=target_shape[1:],
            device=input.device,
            x_T_sampling_method=x_T_sampling_method,
            ddim_churn=ddim_churn,
            noise_schedule=noise_schedule,
            beta_endpoints=beta_endpoints,
            tau = tau,
            variance_method=variance_method,
            gamma_t=gamma_t,
        )
    else:
        diffusion = DistributionalDiffusion(
            noise_steps=n_timesteps,
            img_size=target_shape[1:],
            device=input.device,
            distributional_method=distributional_method,
            closed_form=closed_form,
            x_T_sampling_method=x_T_sampling_method,
            ddim_churn=ddim_churn,
            noise_schedule=noise_schedule,
            beta_endpoints=beta_endpoints,
            tau = tau,
            variance_method=variance_method,
            gamma_t=gamma_t,
        )

    sampled_targets = torch.zeros(*target_shape, n_samples).to(input.device)
    if regressor is None:
        pred = None
    else:
        pred = regressor(input)

    if (
        metrics_plots
        and gt_target is not None
        and pred is not None
        and distributional_method != "deterministic"
    ):
        repeated_pred = pred.repeat_interleave(n_samples, dim=0)
        repeated_labels = input.repeat_interleave(n_samples, dim=0)
        sampled_targets, crps_over_time, rmse_over_time, distr_over_time = (
            diffusion.sample_low_dimensional(
                model,
                n=repeated_labels.shape[0],
                conditioning=repeated_labels,
                pred=repeated_pred,
                cfg_scale=cfg_scale,
                gt_target=gt_target,
            )
        )
        sampled_targets = sampled_targets.reshape(
            target_shape[0], n_samples, *target_shape[1:]
        ).moveaxis(1, -1)
        return sampled_targets, crps_over_time, rmse_over_time, distr_over_time
    else:
        for i in range(n_samples):
            with torch.no_grad():
                sampled_targets[..., i] = diffusion.sample_low_dimensional(
                    model,
                    n=input.shape[0],
                    conditioning=input,
                    pred=pred,
                    cfg_scale=cfg_scale,
                ).detach()
        return sampled_targets


class DistributionalDiffusion(Diffusion):
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
        tau = 1,
        variance_method="fixed_ddim",
        gamma_t=None,
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
            tau = tau,
            variance_method=variance_method,
            gamma_t=gamma_t,
        )
        self.distributional_method = distributional_method
        self.closed_form = closed_form
        self.tau = tau

    def sample_noise(self, model, x, t, conditioning=None, pred=None):
        if self.distributional_method in ["normal", "iDDPM"]:
            predicted_noise = model(x, t, conditioning, pred=pred)
            predicted_noise = predicted_noise[..., 0] + np.sqrt(self.tau) * predicted_noise[
                ..., 1
            ] * torch.randn_like(predicted_noise[..., 0], device=self.device)
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
            predicted_noise = mvnorm.sample()

        elif self.distributional_method == "sample":
            predicted_noise = model(x, t, conditioning, pred=pred, n_samples=1).squeeze(
                -1
            )
        elif self.distributional_method == "mixednormal":
            predicted_mixture = model(x, t, conditioning, pred=pred)
            mu = predicted_mixture[..., 0]
            sigma = np.sqrt(self.tau) * predicted_mixture[..., 1]
            weights = predicted_mixture[..., 2]
            sampled_weights = torch.distributions.Categorical(weights).sample()
            sampled_mu = torch.gather(mu, dim=-1, index=sampled_weights.unsqueeze(-1))
            sampled_sigma = torch.gather(
                sigma, dim=-1, index=sampled_weights.unsqueeze(-1)
            )
            predicted_noise = sampled_mu + sampled_sigma * torch.randn_like(
                sampled_mu, device=self.device
            )
            predicted_noise = predicted_noise.squeeze(-1)

        return predicted_noise

    def sample_x_t_closed_form(
        self,
        x,
        t,
        predicted_noise_distribution_params,
        pred,
        i,
        method,
    ):
        if method == "normal":
            predicted_noise_mu = predicted_noise_distribution_params[..., 0]
            predicted_noise_sigma = predicted_noise_distribution_params[..., 1]
            predicted_noise_covariance = torch.diag_embed(predicted_noise_sigma**2) * self.tau
        elif method == "mixednormal":
            mu = predicted_noise_distribution_params[..., 0]
            sigma = predicted_noise_distribution_params[..., 1]
            weights = predicted_noise_distribution_params[..., 2]
            sampled_weights = torch.distributions.Categorical(weights).sample()
            predicted_noise_mu = torch.gather(
                mu, dim=-1, index=sampled_weights.unsqueeze(-1)
            ).squeeze(-1)
            predicted_noise_sigma = torch.gather(
                sigma, dim=-1, index=sampled_weights.unsqueeze(-1)
            ).squeeze(-1)
            predicted_noise_covariance = torch.diag_embed(predicted_noise_sigma**2) *self.tau
        elif method == "mvnormal":
            if (
                predicted_noise_distribution_params.shape[-1]
                == predicted_noise_distribution_params.shape[-2] + 1
            ):  # Cholesky
                predicted_noise_mu = predicted_noise_distribution_params[..., 0]
                predicted_L = predicted_noise_distribution_params[..., 1:]
                predicted_noise_covariance = MultivariateNormal(
                    predicted_noise_mu, scale_tril=predicted_L
                ).covariance_matrix * self.tau
            else:  # LORA
                predicted_noise_mu = predicted_noise_distribution_params[..., 0]
                diag = predicted_noise_distribution_params[..., 1]
                lora = predicted_noise_distribution_params[..., 2:]
                predicted_noise_covariance = LowRankMultivariateNormal(
                    predicted_noise_mu, lora, diag
                ).covariance_matrix * self.tau
        else:
            raise Exception(f"Invalid method {method}")

        alpha = self.alpha[t[0]]  # reshape_to_x_sample(self.alpha[t], x)
        alpha_hat = self.alpha_hat[t[0]]  # reshape_to_x_sample(self.alpha_hat[t], x)

        if pred is None:
            pred = 0

        x_0_hat = (
            x
            - torch.sqrt(1 - alpha_hat) * predicted_noise_mu
            - (1 - torch.sqrt(alpha_hat)) * pred
        ) / torch.sqrt(alpha_hat)  # DDIM eq. 9

        if i > 1:
            alpha_hat_t_minus_1 = self.alpha_hat[
                t[0] - 1
            ]  # reshape_to_x_sample(self.alpha_hat[t - 1], x)
            ddim_sigma = (
                self.ddim_churn
                * torch.sqrt((1 - alpha_hat_t_minus_1) / (1 - alpha_hat))
                * torch.sqrt(1 - alpha)
            )

            if self.x_T_sampling_method == "standard":
                predicted_noise_ddim = predicted_noise_mu
            elif self.x_T_sampling_method == "CARD":
                predicted_noise_ddim = (
                    predicted_noise_mu
                    + (1 - torch.sqrt(alpha_hat)) / (torch.sqrt(1 - alpha_hat)) * pred
                )
            else:
                raise NotImplementedError(
                    f'Please choose as the x_T_sampling_method "standard" or "CARD". You chose'
                    f"{self.x_T_sampling_method}"
                )
            reverse_posterior_mean = (
                torch.sqrt(alpha_hat_t_minus_1) * x_0_hat
                + torch.sqrt(1 - alpha_hat_t_minus_1 - ddim_sigma**2)
                * predicted_noise_ddim
            )
            A = torch.sqrt(1 - alpha_hat_t_minus_1 - ddim_sigma**2) - torch.sqrt(
                (1 - alpha_hat) / alpha
            )
            covariance_matrix = A**2 * predicted_noise_covariance + torch.diag_embed(
                reshape_to_x_sample(ddim_sigma**2, x)
            )
        else:
            ddim_sigma = 0
            reverse_posterior_mean = x_0_hat
            A = -torch.sqrt((1 - alpha_hat) / alpha_hat)
            covariance_matrix = A**2 * predicted_noise_covariance

        # Sample from final closed form normal
        if reverse_posterior_mean.shape[-1] == 1: # We are in the 1D case
            noise = torch.randn_like(reverse_posterior_mean)
            new_x = reverse_posterior_mean + torch.sqrt(covariance_matrix.squeeze(-1)) * noise
        else:
            mvnormal = MultivariateNormal(loc = reverse_posterior_mean, covariance_matrix=covariance_matrix)
            new_x = mvnormal.sample()

        return new_x

    def sample_low_dimensional(
        self, model, n, conditioning=None, cfg_scale=3, pred=None, gt_target=None
    ):
        model.eval()
        if gt_target is not None:
            n_samples = n // gt_target.shape[0]

            crps_per_t = []
            rmse_per_t = []
            distr_per_t = []

        with torch.no_grad():
            x = self.sample_x_T((n, *self.img_size), pred, inference=True)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                if self.closed_form and self.distributional_method in [
                    "normal",
                    "mixednormal",
                    "mvnormal",
                ]:
                    predicted_noise_distribution_params = model(
                        x, t, conditioning, pred=pred
                    )
                    x = self.sample_x_t_closed_form(
                        x,
                        t,
                        predicted_noise_distribution_params,
                        pred,
                        i,
                        method=self.distributional_method,
                    )
                elif self.distributional_method == "OCM":
                    # OCM head returns (mu, m_pred) with shape (..., 2).
                    # Use mu as the noise prediction and m_pred to compute
                    # the per-input, per-dim reverse-process variance.
                    output = model(x, t, conditioning, pred=pred)
                    predicted_noise = output[..., 0].reshape(x.shape)
                    predicted_moment = output[..., 1].reshape(x.shape)
                    x = self.sample_x_t_inference_DDIM(
                        x, t, predicted_noise, pred, i,
                        predicted_moment=predicted_moment,
                    )
                else:
                    predicted_noise = self.sample_noise(model, x, t, conditioning, pred)
                    if cfg_scale > 0:
                        uncond_predicted_noise = self.sample_noise(
                            model, x, t, None, pred
                        )
                        predicted_noise = torch.lerp(
                            uncond_predicted_noise, predicted_noise, cfg_scale
                        )
                    x = self.sample_x_t_inference_DDIM(x, t, predicted_noise, pred, i)

                if gt_target is not None:
                    single_pred = pred.reshape(
                        (gt_target.shape[0], n_samples, *self.img_size)
                    ).mean(dim=1)
                    gt_images_t = self.noise_low_dimensional(
                        gt_target,
                        (torch.ones(gt_target.shape[0]) * i).long().to(self.device),
                        pred=single_pred,
                    )[0]
                    x_t = x.reshape(
                        gt_target.shape[0], n_samples, *self.img_size
                    ).moveaxis(1, -1)
                    crps_per_t.append(
                        crps_ensemble(
                            gt_images_t.cpu(),
                            x_t.cpu(),
                            backend="torch",
                        )
                        .sum()
                        .item()
                    )
                    rmse_per_t.append(
                        ((x_t.squeeze() - gt_images_t) ** 2).mean(axis=-1).sum().cpu()
                    )
                    distr_per_t.append(
                        (x_t.mean(axis=-1).cpu(), x_t.std(axis=-1).cpu())
                    )
        model.train()

        if gt_target is not None:
            return x, crps_per_t, rmse_per_t, distr_per_t
        else:
            return x

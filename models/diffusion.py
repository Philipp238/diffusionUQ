import torch
from scoringrules import crps_ensemble
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal

def reshape_to_x_sample(parameter, x):
    return parameter.view(*parameter.shape, *(1,) * (x.ndim - parameter.ndim)).expand(x.shape)

class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        img_size=256,
        device="cuda",
        x_T_sampling_method="standard",
        ddim_churn=1.0,
        noise_schedule="linear"
    ):
        self.device = device

        self.noise_steps = noise_steps
        self.noise_schedule = noise_schedule

        self.prepare_noise_schedule()

        self.img_size = img_size
        self.x_T_sampling_method = x_T_sampling_method
        self.ddim_churn = ddim_churn

    def sample_x_T(self, shape, pred, inference = True):
        if self.x_T_sampling_method in ["standard"]:
            x = torch.randn(shape).to(self.device)
        elif self.x_T_sampling_method == "naive-regressor-mean":
            x = torch.randn(shape).to(self.device) + pred
        elif self.x_T_sampling_method == 'CARD':
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

    def sample_x_t_inference_DDIM(self, x, t, predicted_noise, pred, i):
        alpha = reshape_to_x_sample(self.alpha[t], x)
        alpha_hat = reshape_to_x_sample(self.alpha_hat[t], x) 

        x_0_hat = (x 
                   - torch.sqrt(1 - alpha_hat) * predicted_noise
                   - (1 - torch.sqrt(alpha_hat)) * pred
                   ) / torch.sqrt(alpha_hat) # DDIM eq. 9

        if i > 1:
            alpha_hat_t_minus_1 = reshape_to_x_sample(self.alpha_hat[t - 1], x)
            ddim_sigma = self.ddim_churn * torch.sqrt((1 - alpha_hat_t_minus_1) / (1 - alpha_hat)) * torch.sqrt(1 - alpha)

            if self.x_T_sampling_method == "standard":
                predicted_noise_ddim = predicted_noise
            elif self.x_T_sampling_method == "CARD":
                predicted_noise_ddim = predicted_noise + (1 - torch.sqrt(alpha_hat))/(torch.sqrt(1 - alpha_hat)) * pred
            else:
                raise NotImplementedError(
                    f'Please choose as the x_T_sampling_method "standard" or "CARD". You chose'
                    f"{self.x_T_sampling_method}"
                )
            reverse_posterior_mean = (
                torch.sqrt(alpha_hat_t_minus_1) * x_0_hat + 
                torch.sqrt(1 - alpha_hat_t_minus_1 - ddim_sigma**2) * predicted_noise_ddim
            )
        else:
            alpha_hat_t_minus_1 = torch.ones_like(alpha_hat)
            ddim_sigma = self.ddim_churn * torch.sqrt((1 - alpha_hat_t_minus_1) / (1 - alpha_hat)) * torch.sqrt(1 - alpha)

            reverse_posterior_mean = x_0_hat

        noise = torch.randn_like(x)
        new_x = reverse_posterior_mean + ddim_sigma * noise

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

    def prepare_noise_schedule(self):
        if self.noise_schedule == 'linear':
            beta_start = 1e-4
            beta_end = 0.02
            self.beta = torch.linspace(beta_start, beta_end, self.noise_steps).to(self.device)
        elif self.noise_schedule == 'cosine':
            t = torch.arange(0, self.noise_steps+1, device=self.device)
            def f(t, T, s):
                return torch.cos((t/T+s)/(1+s) * torch.pi / 2)**2
            T=self.noise_steps
            s=0.008  # from improved DDPM paper, might try to adjust it, as the value is motivated by the pixel bin size
            alpha_bar = f(t, T, s) / f(torch.tensor([0], device=self.device), T, s)
            self.beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999) 
        else:
            raise ValueError(f'Noise schedule must be "linear" or "cosine". You chose "{self.noise_schedule}".')
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

    def sample_low_dimensional(
        self, model, n, conditioning=None, cfg_scale=3, pred=None
    ):
        assert (self.x_T_sampling_method == "standard") or not (pred is None)

        model.eval()
        with torch.no_grad():
            x = self.sample_x_T((n, *self.img_size), pred, inference=True)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, conditioning, pred = pred)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None, pred = pred)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )

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
    regressor=None,
    cfg_scale=3,
    gt_images=None,
    ddim_churn=1.0,
    noise_schedule=None,
    metrics_plots=False,

):
    if distributional_method == "deterministic":
        diffusion = Diffusion(
            noise_steps=n_timesteps,
            img_size=target_shape[1:],
            device=input.device,
            x_T_sampling_method=x_T_sampling_method,
            ddim_churn=ddim_churn,
            noise_schedule=noise_schedule
        )
    else:
        diffusion = DistributionalDiffusion(
            noise_steps=n_timesteps,
            img_size=target_shape[1:],
            device=input.device,
            distributional_method=distributional_method,
            x_T_sampling_method=x_T_sampling_method,
            ddim_churn=ddim_churn,
            noise_schedule=noise_schedule
        )

    sampled_images = torch.zeros(*target_shape, n_samples).to(input.device)
    if regressor is None:
        pred = None
    else:
        pred = regressor(input)

    if metrics_plots and gt_images is not None and pred is not None and distributional_method != "deterministic":
        repeated_pred = pred.repeat_interleave(n_samples, dim=0)
        repeated_labels = input.repeat_interleave(n_samples, dim=0)
        sampled_images, crps_over_time, rmse_over_time, distr_over_time = (
            diffusion.sample_low_dimensional(
                model,
                n=repeated_labels.shape[0],
                conditioning=repeated_labels,
                pred=repeated_pred,
                cfg_scale=cfg_scale,
                gt_images=gt_images,
            )
        )
        sampled_images = sampled_images.reshape(
            target_shape[0], n_samples, *target_shape[1:]
        ).moveaxis(1, -1)
        return sampled_images, crps_over_time, rmse_over_time, distr_over_time
    else:
        for i in range(n_samples):
            with torch.no_grad():
                sampled_images[..., i] = diffusion.sample_low_dimensional(
                    model,
                    n=input.shape[0],
                    conditioning=input,
                    pred=pred,
                    cfg_scale=cfg_scale,
                ).detach()
        return sampled_images


class DistributionalDiffusion(Diffusion):
    def __init__(
        self,
        noise_steps=1000,
        noise_schedule="linear",
        img_size=256,
        device="cuda",
        distributional_method="normal",
        x_T_sampling_method="standard",
        ddim_churn=1.0,
    ):
        super().__init__(
            noise_steps=noise_steps,
            noise_schedule=noise_schedule,
            img_size=img_size,
            device=device,
            x_T_sampling_method=x_T_sampling_method,
            ddim_churn=ddim_churn,
        )
        self.distributional_method = distributional_method

    def sample_noise(self, model, x, t, conditioning=None, pred=None):
        if self.distributional_method == "normal":
            predicted_noise = model(x, t, conditioning, pred = pred)
            predicted_noise = predicted_noise[..., 0] + predicted_noise[
                ..., 1
            ] * torch.randn_like(predicted_noise[..., 0], device=self.device)
        elif self.distributional_method == "mvnormal":
            predicted_noise = model(x, t, conditioning, pred = pred)
            if predicted_noise.shape[-1] == predicted_noise.shape[-2]+1:
                # Cholesky
                mu = predicted_noise[..., 0]
                L_full = predicted_noise[..., 1:]
                mvnorm = MultivariateNormal(loc = mu, scale_tril = L_full)
            else: # Lora
                mu = predicted_noise[...,0]
                diag = predicted_noise[...,1]
                lora = predicted_noise[...,2:]
                mvnorm = LowRankMultivariateNormal(mu, lora, diag)
            predicted_noise = mvnorm.sample()

        elif self.distributional_method == "sample":
            predicted_noise = model(x, t, conditioning, pred = pred, n_samples=1).squeeze(-1)
        elif self.distributional_method == "mixednormal":
            predicted_mixture = model(x, t, conditioning, pred= pred)
            mu = predicted_mixture[..., 0]
            sigma = predicted_mixture[..., 1]
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
        self, x, t, predicted_noise_distribution_params, pred, i, method,
    ):
        if method == "closed_form_normal":
            predicted_noise_mu = predicted_noise_distribution_params[..., 0]
            predicted_noise_sigma = predicted_noise_distribution_params[..., 1]
        elif method == "closed_form_mixednormal":
            mu = predicted_noise_distribution_params[..., 0]
            sigma = predicted_noise_distribution_params[..., 1]
            weights = predicted_noise_distribution_params[..., 2]
            sampled_weights = torch.distributions.Categorical(weights).sample()
            predicted_noise_mu = torch.gather(mu, dim=-1, index=sampled_weights.unsqueeze(-1)).squeeze(-1)
            predicted_noise_sigma = torch.gather(
                sigma, dim=-1, index=sampled_weights.unsqueeze(-1)
            ).squeeze(-1)
        else:
            raise Exception(f"Invalid method {method}")

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

        variance_factor_default = beta**2 / (alpha_hat * (1 - alpha_hat))

        if self.x_T_sampling_method in ["standard", "naive-regressor-mean"]:
            x = (
                1
                / torch.sqrt(alpha)
                * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise_mu)
                + torch.sqrt((variance_factor_default * predicted_noise_sigma**2 + beta))
                * noise
            )
        elif self.x_T_sampling_method == "CARD":
            y_hat_0 = (
                1
                / torch.sqrt(alpha_hat)
                * (
                    x
                    - (1 - torch.sqrt(alpha_hat)) * pred
                    - torch.sqrt(1 - alpha_hat) * predicted_noise_mu
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

                variance_factor_CARD = alpha_hat_t_minus_1 * variance_factor_default

                x = (gamma_0 * y_hat_0 + gamma_1 * x + gamma_2 * pred) + self.ddim_sigma * torch.sqrt(
                    variance_factor_CARD * predicted_noise_sigma**2 + beta_wiggle
                ) * noise

            else:
                x = y_hat_0
        else:
            raise NotImplementedError(
                f'Please choose as the x_T_sampling_method "standard", "CARD", or "naive-regressor-mean". You chose'
                f"{self.x_T_sampling_method}"
            )
        return x

    def sample_low_dimensional(
        self, model, n, conditioning=None, cfg_scale=3, pred=None, gt_images=None
    ):
        model.eval()
        if gt_images is not None:
            n_samples = n // gt_images.shape[0]

            crps_per_t = []
            rmse_per_t = []
            distr_per_t = []

        with torch.no_grad():
            x = self.sample_x_T((n, *self.img_size), pred, inference=True)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                if self.distributional_method.startswith("closed_form"):
                    predicted_noise_distribution_params = model(
                        x, t, conditioning, pred
                    )
                    x = self.sample_x_t_closed_form(
                        x, t, predicted_noise_distribution_params, pred, i, method=self.distributional_method,
                    )
                else:
                    predicted_noise = self.sample_noise(model, x, t, conditioning, pred)
                    if cfg_scale > 0:
                        uncond_predicted_noise = self.sample_noise(model, x, t, None, pred)
                        predicted_noise = torch.lerp(
                            uncond_predicted_noise, predicted_noise, cfg_scale
                        )
                    x = self.sample_x_t_inference_DDIM(x, t, predicted_noise, pred, i)

                if gt_images is not None:
                    single_pred = pred.reshape(
                        (gt_images.shape[0], n_samples, *self.img_size)
                    ).mean(dim=1)
                    gt_images_t = self.noise_low_dimensional(
                        gt_images,
                        (torch.ones(gt_images.shape[0]) * i).long().to(self.device),
                        pred=single_pred
                    )[0]
                    x_t = x.reshape(
                        gt_images.shape[0], n_samples, *self.img_size
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

        if gt_images is not None:
            return x, crps_per_t, rmse_per_t, distr_per_t
        else:
            return x

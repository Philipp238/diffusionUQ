import torch
from scoringrules import crps_ensemble


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
        x_T_sampling_method="standard",
        ddim_sigma=1,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device
        self.x_T_sampling_method = x_T_sampling_method
        self.ddim_sigma = ddim_sigma

    def sample_x_T(self, shape, pred):
        if self.x_T_sampling_method in ['standard', 'CARD']:
            x = torch.randn(shape).to(self.device)
        elif self.x_T_sampling_method == 'naive-regressor-mean':
            x = torch.randn(shape).to(self.device) + pred
        else:
            raise NotImplementedError(f'Please choose as the x_T_sampling_method "standard", "CARD", or "naive-regressor-mean". You chose'
                                      f'{self.x_T_sampling_method}')
        return x
    
    def sample_x_t_inference(self, x, t, predicted_noise, pred, i):
        alpha = self.alpha[t][:, None]
        alpha_hat = self.alpha_hat[t][:, None]
        beta = self.beta[t][:, None]
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        if self.x_T_sampling_method in ['standard', 'naive-regressor-mean']:
            x = (
                1
                / torch.sqrt(alpha)
                * (
                    x
                    - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                )
                + self.ddim_sigma * torch.sqrt(beta) * noise
            )
        elif self.x_T_sampling_method == 'CARD':
            y_hat_0 = 1 / torch.sqrt(alpha_hat) * (x - (1 - torch.sqrt(alpha_hat)) * pred - torch.sqrt(1 - alpha_hat) * predicted_noise)
            if i > 1:
                alpha_hat_t_minus_1 = self.alpha_hat[t-1][:, None]
                
                gamma_0 = beta * torch.sqrt(alpha_hat_t_minus_1) / (1 - alpha_hat)
                gamma_1 = (1 - alpha_hat_t_minus_1) * torch.sqrt(alpha) / (1 - alpha_hat)
                gamma_2 = (1 + (torch.sqrt(alpha_hat) - 1) * (torch.sqrt(alpha) + torch.sqrt(alpha_hat_t_minus_1)) / (1 - alpha_hat))
                
                beta_wiggle = (1 - alpha_hat_t_minus_1) / (1 - alpha_hat) * beta
                
                x = gamma_0 * y_hat_0 + gamma_1 * x + gamma_2 * pred + self.ddim_sigma * torch.sqrt(beta_wiggle) * noise
            
            else:
                 x = y_hat_0   
        else:
            raise NotImplementedError(f'Please choose as the x_T_sampling_method "standard", "CARD", or "naive-regressor-mean". You chose'
                                      f'{self.x_T_sampling_method}')
        return x
        

    def sample_x_t_training(self, x, eps, t, pred=None):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        if self.x_T_sampling_method in ['standard', 'naive-regressor-mean']:
            target_training = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        elif self.x_T_sampling_method == 'CARD':
            target_training = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps + (1 - sqrt_alpha_hat) * pred
        return target_training
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_low_dimensional(self, x, t, pred=None):
        assert (self.x_T_sampling_method == 'standard') or not (pred is None)

        eps = self.sample_x_T(x.shape, pred)
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

    def sample_low_dimensional(self, model, n, conditioning=None, cfg_scale=3, pred=None):
        assert (self.x_T_sampling_method == 'standard') or not (pred is None)
        
        model.eval()
        with torch.no_grad():
            x = self.sample_x_T((n, self.img_size), pred)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, conditioning, pred)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None, pred)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                
                x = self.sample_x_t_inference(x, t, predicted_noise, pred, i)

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
    labels,
    n_timesteps,
    images_shape,
    n_samples,
    x_T_sampling_method,
    distributional_method="deterministic",
    regressor=None,
    cfg_scale=3,
    log_variance=False,
    gt_images=None,
    ddim_sigma=1.0,
):
    if distributional_method == "deterministic":
        diffusion = Diffusion(
            noise_steps=n_timesteps,
            img_size=images_shape[1],
            device=labels.device,
            x_T_sampling_method=x_T_sampling_method,
            ddim_sigma=ddim_sigma,
        )
    else: 
        diffusion = DistributionalDiffusion(
            noise_steps=n_timesteps,
            img_size=images_shape[1],
            device=labels.device,
            distributional_method=distributional_method,
            x_T_sampling_method=x_T_sampling_method,
            ddim_sigma=ddim_sigma,
        )

    sampled_images = torch.zeros(*images_shape, n_samples).to(labels.device)
    if regressor is None:
        pred = None
    else:
        pred = regressor(labels)

    if gt_images is not None and distributional_method != "deterministic":
        repeated_pred = pred.repeat_interleave(n_samples, dim=0)
        repeated_labels = labels.repeat_interleave(n_samples, dim=0)
        sampled_images, crps_over_time, rmse_over_time, distr_over_time = diffusion.sample_low_dimensional(
            model,
            n=repeated_labels.shape[0],
            conditioning=repeated_labels,
            pred=repeated_pred,
            cfg_scale=cfg_scale,
            log_variance=log_variance,
            gt_images=gt_images,
            )
        sampled_images = sampled_images.reshape(images_shape[0], n_samples, images_shape[1]).permute(0,2,1)
    else:
        for i in range(n_samples):
            with torch.no_grad():
                sampled_images[..., i] = diffusion.sample_low_dimensional(
                    model, n=labels.shape[0], conditioning=labels, pred=pred, cfg_scale=cfg_scale
                ).detach()

    if gt_images is not None and distributional_method != "deterministic":
        return sampled_images, crps_over_time, rmse_over_time, distr_over_time
    else:
        return sampled_images

class DistributionalDiffusion(Diffusion):
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
        distributional_method="normal",
        x_T_sampling_method="standard",
        ddim_sigma=1.0
    ):
        super().__init__(
            noise_steps=noise_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            img_size=img_size,
            device=device,
            x_T_sampling_method=x_T_sampling_method,
            ddim_sigma=ddim_sigma,
        )
        self.distributional_method = distributional_method

    def sample_noise(self, model, x, t, conditioning=None, pred=None):
        if self.distributional_method == "normal":
            predicted_noise = model(x, t, conditioning, pred)
            predicted_noise = predicted_noise[..., 0] + predicted_noise[
                ..., 1
            ] * torch.randn_like(predicted_noise[..., 0], device=self.device)
        elif self.distributional_method == "sample":
            predicted_noise = model(x, t, conditioning, pred, n_samples = 1).squeeze(1)
        elif self.distributional_method == "mixednormal":
            predicted_mixture = model(x, t, conditioning, pred)
            mu = predicted_mixture[..., 0]
            sigma = predicted_mixture[..., 1]
            weights = predicted_mixture[..., 2]
            sampled_weights = torch.distributions.Categorical(weights).sample()
            sampled_mu = torch.gather(mu, dim=2, index=sampled_weights.unsqueeze(1))
            sampled_sigma = torch.gather(sigma, dim=2, index=sampled_weights.unsqueeze(1))
            predicted_noise = sampled_mu + sampled_sigma * torch.randn_like(
                sampled_mu, device=self.device
            )
            predicted_noise = predicted_noise.squeeze(1)

        return predicted_noise


    def sample_x_t_closed_form_normal(self, x, t, predicted_noise_distribution_params, pred, i):
        #assert self.x_T_sampling_method in ['standard', 'naive-regressor-mean'], f"Closed form normal sampling not supported for {self.x_T_sampling_method}"
        
        predicted_noise_mu = predicted_noise_distribution_params[..., 0]
        predicted_noise_sigma = predicted_noise_distribution_params[..., 1]
        
        alpha = self.alpha[t][:, None]
        alpha_hat = self.alpha_hat[t][:, None]
        beta = self.beta[t][:, None]
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        variance_factor_default = (beta**2 / (alpha * (1-alpha_hat)))
            
        if self.x_T_sampling_method in ['standard', 'naive-regressor-mean']:
            x = (
                1
                / torch.sqrt(alpha)
                * (
                    x
                    - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise_mu
                )
                + (variance_factor_default * predicted_noise_sigma + torch.sqrt(beta)) * noise
            )
        elif self.x_T_sampling_method == 'CARD':
            y_hat_0 = 1 / torch.sqrt(alpha_hat) * (x - (1 - torch.sqrt(alpha_hat)) * pred - torch.sqrt(1 - alpha_hat) * predicted_noise_mu)
            if i > 1:
                alpha_hat_t_minus_1 = self.alpha_hat[t-1][:, None]
                
                gamma_0 = beta * torch.sqrt(alpha_hat_t_minus_1) / (1 - alpha_hat)
                gamma_1 = (1 - alpha_hat_t_minus_1) * torch.sqrt(alpha) / (1 - alpha_hat)
                gamma_2 = (1 + (torch.sqrt(alpha_hat) - 1) * (torch.sqrt(alpha) + torch.sqrt(alpha_hat_t_minus_1)) / (1 - alpha_hat))
                
                beta_wiggle = (1 - alpha_hat_t_minus_1) / (1 - alpha_hat) * beta

                variance_factor_CARD = alpha_hat_t_minus_1 * variance_factor_default
                
                x = (
                    gamma_0 * y_hat_0 + gamma_1 * x + gamma_2 * pred
                    ) + (variance_factor_CARD * predicted_noise_sigma + self.ddim_sigma * torch.sqrt(beta_wiggle)) * noise
            
            else:
                 x = y_hat_0   
        else:
            raise NotImplementedError(f'Please choose as the x_T_sampling_method "standard", "CARD", or "naive-regressor-mean". You chose'
                                      f'{self.x_T_sampling_method}')
        return x


    def sample_low_dimensional(self, model, n, conditioning=None, cfg_scale=3, pred=None, log_variance=False, gt_images=None):
        model.eval()
        if gt_images is not None:
            n_samples = n // gt_images.shape[0]
            
            crps_per_t  = []
            rmse_per_t  = []
            distr_per_t = []

        with torch.no_grad():
            x = self.sample_x_T((n, self.img_size), pred)
            if self.x_T_sampling_method == "CARD":
                x += pred
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                if self.distributional_method == "closed_form_normal":
                    predicted_noise_distribution_params = model(x, t, conditioning, pred)
                else:
                    predicted_noise = self.sample_noise(model, x, t, conditioning, pred, log_variance=log_variance)
                if cfg_scale > 0:
                    uncond_predicted_noise = self.sample_noise(model, x, t, None, pred)
                    predicted_noise = torch.lerp(
                        uncond_predicted_noise, predicted_noise, cfg_scale
                    )
                if self.distributional_method == "closed_form_normal":
                    x = self.sample_x_t_closed_form_normal(x, t, predicted_noise_distribution_params, pred, i)
                else:
                    x = self.sample_x_t_inference(x, t, predicted_noise, pred, i)

                if gt_images is not None:
                    single_pred = pred.reshape((gt_images.shape[0], n_samples, self.img_size)).mean(dim=1)
                    gt_images_t = self.noise_low_dimensional(gt_images, (torch.ones(gt_images.shape[0]) * i).long().to(self.device), pred=single_pred)[0]
                    x_t = x.reshape(gt_images.shape[0], n_samples, self.img_size).permute(0,2,1)
                    crps_per_t.append(crps_ensemble(
                        gt_images_t.cpu(),
                        x_t.cpu(),
                        backend="torch",
                    ).sum().item())
                    rmse_per_t.append(((x_t.squeeze() - gt_images_t)**2).mean(axis=1).sum().cpu())
                    distr_per_t.append((x_t.mean(axis=2).cpu(), x_t.std(axis=2).cpu())) 
        model.train()

        if gt_images is not None:
            return x, crps_per_t, rmse_per_t, distr_per_t
        else:
            return x

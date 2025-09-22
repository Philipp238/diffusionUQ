# Implementation of U-Net based (distributional-)diffusion models for 1D and 2D data.

from __future__ import annotations

import torch
from torch import nn

from models.unet_layers import Conv1d, Conv2d, SongUNet

EPS = 1e-6


class UNetDiffusion(nn.Module):
    """U-Net based diffusion model for 1D and 2D data."""

    def __init__(
        self,
        d=1,
        conditioning_dim=3,
        hidden_channels=16,
        in_channels=1,
        out_channels=1,
        init_features=32,
        device="cuda",
        domain_dim=(128,),
    ):
        """U-Net based diffusion model for 1D and 2D data.

        Args:
            d (int, optional): The dimensionality of the input data (1 or 2). Defaults to 1.
            conditioning_dim (int, optional): The dimensionality of the conditioning input. Defaults to 3.
            hidden_channels (int, optional): The number of hidden channels in the U-Net. Defaults to 16.
            in_channels (int, optional): The number of input channels. Defaults to 1.
            out_channels (int, optional): The number of output channels. Defaults to 1.
            init_features (int, optional): The number of initial features. Defaults to 32.
            device (str, optional): The device to run the model on. Defaults to "cuda".
            domain_dim (tuple, optional): The dimensionality of the domain. Defaults to (128,).

        Raises:
            NotImplementedError: If the dimensionality is not 1 or 2.
        """
        super().__init__()
        self.d = d
        self.device = device
        self.hidden_dim = hidden_channels
        self.features = init_features
        if d == 1:
            self.unet = SongUNet(
                img_resolution=domain_dim,
                in_channels=(in_channels + conditioning_dim),
                out_channels=1,
                d=1,
                attn_resolutions=[16],
                model_channels=hidden_channels,
            )
            self.output_projection = nn.Conv1d(
                in_channels=init_features,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            )
        elif d == 2:
            self.unet = SongUNet(
                img_resolution=domain_dim,
                in_channels=(in_channels + conditioning_dim),
                out_channels=1,
                d=2,
                attn_resolutions=[16],
                model_channels=hidden_channels,
            )
            self.output_projection = nn.Conv2d(
                in_channels=init_features,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            )
        else:
            raise NotImplementedError(
                "Only 1D and 2D U-Nets are implemented at this stage."
            )

    def forward_body(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the U-Net body.

        Args:
            x_t (torch.Tensor): Noisy input at time t of shape [B, C, D1,..., DN].
            t (torch.Tensor): Timestep tensor
            condition_input (torch.Tensor): Conditioning input of shape [B, C, D1,..., DN].

        Returns:
            torch.Tensor: Predicted latent representation
        """
        t = t.unsqueeze(-1).type(torch.float32)
        x_latent = self.unet(x_t, t, condition_input)
        return x_latent

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the U-Net diffusion model.

        Args:
            x_t (torch.Tensor): Noisy input at time t of shape [B, C, D1,..., DN].
            t (torch.Tensor): Timestep tensor
            condition_input (torch.Tensor): Conditioning input of shape [B, C, D1,..., DN].

        Returns:
            torch.Tensor: Predicted noise
        """
        # Forward body
        latent_pred = self.forward_body(x_t, t, condition_input)
        # Reprojection to original data space
        eps_pred = self.output_projection(latent_pred)
        return eps_pred


class UNet_diffusion_normal(nn.Module):
    """Distributional diffusion U-Net with univariate normal parametrization."""

    def __init__(self, backbone: UNetDiffusion, d: int = 1, target_dim: int = 1):
        """Initializes the UNet_diffusion_normal model.

        Args:
            backbone (UNetDiffusion): Backbone diffusion model.
            d (int, optional): Dimensionality of the input/output. Defaults to 1.
            target_dim (int, optional): Number of output channels. Defaults to 1.
        """
        super(UNet_diffusion_normal, self).__init__()
        self.backbone = backbone
        hidden_dim = backbone.features
        if d == 1:
            self.mu_projection = Conv1d(
                in_channels=hidden_dim, out_channels=target_dim, kernel=3
            )
            self.sigma_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=target_dim,
                kernel=3,
                init_bias=1,
            )
        elif d == 2:
            self.mu_projection = Conv2d(
                in_channels=hidden_dim, out_channels=target_dim, kernel=3
            )
            self.sigma_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=target_dim,
                kernel=3,
                init_bias=1,
            )
        self.sofplus = nn.Softplus()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x_t (torch.Tensor): Noisy input at time t of shape [B, C, D1,..., DN].
            t (torch.Tensor): Timestep tensor
            condition_input (torch.Tensor): Conditioning input of shape [B, C, D1,..., DN].

        Returns:
            torch.Tensor: Noise prediction of shape [B, target_dim, D1,..., DN, 2] where the last dimension
            corresponds to the mean and standard deviation.
        """
        x_t = self.backbone.forward_body(x_t, t, condition_input)
        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        sigma = self.sofplus(sigma) + EPS
        output = torch.stack([mu, sigma], dim=-1)
        return output


class UNet_diffusion_mvnormal(nn.Module):
    """Distributional diffusion U-Net with multivariate normal parametrization."""

    def __init__(
        self,
        backbone: UNetDiffusion,
        d: int = 1,
        target_dim: int = 1,
        domain_dim: tuple = (128,),
        method: str = "lora",
        rank: int = 3,
    ):
        """Initialize model.

        Args:
            backbone (UNetDiffusion): Backbone diffusion model.
            d (int, optional): Dimensionality of the input/output. Defaults to 1.
            target_dim (int, optional): Number of output channels. Defaults to 1.
            domain_dim (tuple, optional): The dimensionality of the domain. Defaults to (128,).
            method (str, optional): Covariance approximation method, must be in ["lora", "cholesky"]. Defaults to "lora".
            rank (int, optional): Rank of the low-rank approximation. Defaults to 3.
        """
        super(UNet_diffusion_mvnormal, self).__init__()
        self.backbone = backbone
        hidden_dim = backbone.features
        self.method = method
        self.target_dim = target_dim
        self.domain_dim = domain_dim[-1]
        if method == "lora":
            sigma_out_channels = target_dim * (rank + 1)  # Rank + diagonal
        elif method == "cholesky":
            # Covariance buffer
            self.register_buffer(
                "tril_template",
                torch.zeros(self.domain_dim, self.domain_dim, dtype=torch.int64),
            )
            self.register_buffer(
                "tril_indices", torch.tril_indices(self.domain_dim, self.domain_dim)
            )
            self.tril_template[self.tril_indices.tolist()] = torch.arange(
                self.tril_indices.shape[1]
            )
            self.num_tril_params = self.tril_indices.shape[1]
            sigma_out_channels = (self.domain_dim) // 2 + 1
        if d == 1:
            self.mu_projection = Conv1d(
                in_channels=hidden_dim, out_channels=self.target_dim, kernel=3
            )
            self.sigma_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=sigma_out_channels,
                kernel=3,
                init_bias=1,
            )
        elif d == 2:
            self.mu_projection = Conv2d(
                in_channels=hidden_dim, out_channels=self.target_dim, kernel=3
            )
            self.sigma_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=sigma_out_channels,
                kernel=3,
                init_bias=1,
            )
        self.sofplus = nn.Softplus()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model.
        Args:
            x_t (torch.Tensor): Noisy input at time t of shape [B, C, D1,..., DN].
            t (torch.Tensor): Timestep tensor
            condition_input (torch.Tensor): Conditioning input of shape [B, C, D1,..., DN].

        Returns:
            torch.Tensor: Noise prediction
            For method == "lora" prediction is of shape [B, target_dim, D1,..., DN, target_dim * (rank + 1)]
            For method == "cholesky" prediction is of shape [B, target_dim, D1,..., DN, :] where the last dimension
            has the size of the lower triangular part of the covariance matrix.
        """
        x_t = self.backbone.forward_body(x_t, t, condition_input)
        mu = self.mu_projection(x_t).unsqueeze(-1)
        sigma = self.sigma_projection(x_t)
        if self.method == "lora":
            diag = self.sofplus(sigma[:, 0 : self.target_dim]).unsqueeze(-1) + EPS
            lora = sigma[:, self.target_dim :]
            lora = lora.reshape(
                lora.shape[0], self.target_dim, -1, *lora.shape[2:]
            ).moveaxis(2, -1)
            lora = lora / (torch.norm(lora, dim=(-2, -1), keepdim=True) + EPS)
            output = torch.cat([mu, diag, lora], dim=-1)

        elif self.method == "cholesky":
            # Initialize full zero matrix and fill lower triangle
            L_full = torch.zeros(
                mu.shape[0], self.domain_dim, self.domain_dim, device=x_t.device
            )
            L_full[:, self.tril_indices[0], self.tril_indices[1]] = sigma.flatten(
                start_dim=1
            )[:, 0 : self.tril_indices[0].shape[0]]

            # Enforce positive diagonal via softplus()
            diag = (
                nn.functional.softplus(torch.diagonal(L_full, dim1=-2, dim2=-1)) + EPS
            )
            L = torch.tril(L_full)
            L = L / (torch.norm(L, dim=-1, keepdim=True) + EPS)
            L[:, torch.arange(self.domain_dim), torch.arange(self.domain_dim)] = (
                diag.squeeze()
            )
            L = L.unsqueeze(1)
            output = torch.cat([mu, L], dim=-1)
        return output


class UNet_diffusion_sample(nn.Module):
    """Distributional diffusion U-Net with sample-based parametrization."""

    def __init__(self, backbone: UNetDiffusion, n_samples: int = 50):
        """Initialize the model.

        Args:
            backbone (UNetDiffusion): Backbone diffusion model.
            d (int, optional): Dimensionality of the input/output. Defaults to 1.
            target_dim (int, optional): Number of output channels. Defaults to 1.
            hidden_dim (int, optional): Hidden dim of the latent representation. Defaults to 32.
            n_samples (int, optional): Number of samples to generate. Defaults to 50.
        """
        super(UNet_diffusion_sample, self).__init__()
        self.backbone = backbone
        self.n_samples = n_samples

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_input: torch.Tensor,
        n_samples=None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model.
        Args:
            x_t (torch.Tensor): Noisy input at time t of shape [B, C, D1,..., DN].
            t (torch.Tensor): Timestep tensor
            condition_input (torch.Tensor): Conditioning input of shape [B, C, D1,..., DN].
            n_samples (int, optional): Number of samples to generate.

        Returns:
            torch.Tensor: Noise prediction of shape [B, target_dim, D1,..., DN, n_samples].
        """
        if n_samples is None:
            n_samples = self.n_samples

        x_t_expanded = torch.repeat_interleave(
            x_t.unsqueeze(1), n_samples, dim=1
        ).reshape(x_t.shape[0] * n_samples, *x_t.shape[1:])
        t_expanded = torch.repeat_interleave(
            t.unsqueeze(-1), n_samples, dim=-1
        ).reshape(t.shape[0] * n_samples)
        condition_input_expanded = torch.repeat_interleave(
            condition_input.unsqueeze(1), n_samples, dim=1
        ).reshape(condition_input.shape[0] * n_samples, *condition_input.shape[1:])

        # Concatenate noise
        noise = torch.randn_like(x_t_expanded)
        x_t_expanded = torch.cat([x_t_expanded, noise], dim=1).to(x_t.device)

        output = self.backbone.forward(
            x_t_expanded,
            t_expanded,
            condition_input_expanded,
        )
        output = output.reshape(x_t.shape[0], n_samples, *output.shape[1:])
        return torch.moveaxis(output, 1, -1)  # Move sample dimension to last position


class UNet_diffusion_mixednormal(nn.Module):
    """Distributional diffusion U-Net with univariate normal mixture parametrization."""

    def __init__(
        self,
        backbone: UNetDiffusion,
        d: int = 1,
        target_dim: int = 1,
        n_components: int = 3,
    ):
        """Initialize model.

        Args:
            backbone (UNetDiffusion): Backbone diffusion model.
            d (int, optional): Dimensionality of the input/output. Defaults to 1.
            target_dim (int, optional): Number of output channels. Defaults to 1.
            n_components (int, optional): Number of mixture components. Defaults to 3.
        """
        super(UNet_diffusion_mixednormal, self).__init__()
        self.backbone = backbone
        hidden_dim = backbone.features
        self.n_components = n_components
        self.target_dim = target_dim

        if d == 1:
            self.mu_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel=3,
            )
            self.sigma_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel=3,
                init_bias=1,
            )
            self.weights_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel=3,
            )
        elif d == 2:
            self.mu_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel=3,
            )
            self.sigma_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel=3,
                init_bias=1,
            )
            self.weights_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel=3,
            )

        self.sofplus = nn.Softplus()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model.
        Args:
            x_t (torch.Tensor): Noisy input at time t of shape [B, C, D1,..., DN].
            t (torch.Tensor): Timestep tensor
            condition_input (torch.Tensor): Conditioning input of shape [B, C, D1,..., DN].

        Returns:
            torch.Tensor: Noise prediction of shape [B, target_dim, D1,..., DN, n_components, 3].
        """
        x_t = self.backbone.forward_body(x_t, t, condition_input)

        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        weights = self.weights_projection(x_t)
        # Reshape
        mu = mu.reshape(
            mu.shape[0], self.target_dim, self.n_components, *mu.shape[2:]
        ).moveaxis(2, -1)
        sigma = sigma.reshape(
            sigma.shape[0], self.target_dim, self.n_components, *sigma.shape[2:]
        ).moveaxis(2, -1)
        weights = weights.reshape(
            weights.shape[0], self.target_dim, self.n_components, *weights.shape[2:]
        ).moveaxis(2, -1)

        # Apply postprocessing
        sigma = self.sofplus(torch.clamp(sigma, min=-15)) + EPS

        # Take global maxima to obtain global mixture weights
        if len(weights.shape) == 4:
            weights = torch.amax(weights, dim=(-2), keepdims=True).repeat(
                1, 1, mu.shape[-2], 1
            )
        elif len(weights.shape) == 5:
            weights = torch.amax(weights, dim=(-2, -3), keepdims=True).repeat(
                1, 1, mu.shape[-3], mu.shape[-2], 1
            )
        weights = torch.softmax(torch.clamp(weights, min=-15, max=15), dim=-1)

        output = torch.stack([mu, sigma, weights], dim=-1)
        return output


if __name__ == "__main__":
    input = torch.randn(8, 1, 128)
    condition = torch.randn(8, 3, 128)
    output = torch.randn(8, 1, 128)
    t = torch.ones(8) * 0.5

    backbone = UNetDiffusion(
        d=1,
        conditioning_dim=3,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        init_features=32,
        device="cpu",
        domain_dim=(1, 128),
    )
    unet = UNet_diffusion_mixednormal(backbone, d=1, target_dim=1, n_components=3)
    test = unet.forward(input, t, condition)
    print(test.shape)

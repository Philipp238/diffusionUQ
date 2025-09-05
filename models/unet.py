"""
U-Net. Implementation taken and modified from
https://github.com/mateuszbuda/brain-segmentation-pytorch

MIT License

Copyright (c) 2019 mateuszbuda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from models.unet_layers import SongUNet, Conv1d, Conv2d

EPS = 1e-6


class UNetDiffusion(nn.Module):
    def __init__(
        self,
        d=1,
        conditioning_dim=3,
        hidden_channels=16,
        in_channels=1,
        out_channels=1,
        init_features=32,
        device="cuda",
        domain_dim=128,
    ):
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
            raise NotImplementedError("Only 1D U-Net is implemented in this example.")


    def forward_body(self, x_t, t, condition_input, **kwargs):
        # x_t and condition input have shape [B, C, D1,..., DN]
        t = t.unsqueeze(-1).type(torch.float32)
        x_t = self.unet(x_t, t, condition_input)
        return x_t

    def forward(self, x_t, t, condition_input, **kwargs):
        # x_t and condition_input have shape [B, C, D1,..., DN]
        # Forward body
        eps_pred = self.forward_body(x_t, t, condition_input)
        # Reprojection
        output = self.output_projection(eps_pred)
        return output


class UNet_diffusion_normal(nn.Module):
    def __init__(self, backbone, d=1, target_dim=1):
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

    def forward(self, x_t, t, condition_input, **kwargs):
        x_t = self.backbone.forward_body(x_t, t, condition_input)

        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        sigma = self.sofplus(sigma) + EPS
        output = torch.stack([mu, sigma], dim=-1)
        return output


class UNet_diffusion_mvnormal(nn.Module):
    def __init__(self, backbone, d=1, target_dim=1, domain_dim=128, method="lora", rank=3):
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
            sigma_out_channels = (self.domain_dim)//2 +1
        if d == 1:
            self.mu_projection = Conv1d(
                in_channels=hidden_dim, out_channels=self.target_dim, kernel = 3
            )
            self.sigma_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=sigma_out_channels,
                kernel = 3,
                init_bias=1,
            )
        elif d == 2:
            self.mu_projection = Conv2d(
                in_channels=hidden_dim, out_channels=self.target_dim, kernel = 3
            )
            self.sigma_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=sigma_out_channels,
                kernel = 3,
                init_bias=1,
            )
        self.sofplus = nn.Softplus()

    def forward(self, x_t, t, condition_input, **kwargs):
        x_t = self.backbone.forward_body(x_t, t, condition_input)

        mu = self.mu_projection(x_t).unsqueeze(-1)
        sigma = self.sigma_projection(x_t)
        if self.method == "lora":
            diag = self.sofplus(sigma[:, 0 : self.target_dim]).unsqueeze(-1) + EPS
            lora = sigma[:, self.target_dim :]
            lora = lora.reshape(
                lora.shape[0], self.target_dim, -1, *lora.shape[2:]
            ).moveaxis(2, -1)
            lora = lora/(torch.norm(lora, dim=(-2,-1), keepdim=True) + EPS)
            output = torch.cat([mu, diag, lora], dim=-1)

        elif self.method == "cholesky":
            # Initialize full zero matrix and fill lower triangle
            L_full = torch.zeros(mu.shape[0], self.domain_dim, self.domain_dim, device=x_t.device)
            L_full[:, self.tril_indices[0], self.tril_indices[1]] = sigma.flatten(start_dim = 1)[:,0:self.tril_indices[0].shape[0]]

            # Enforce positive diagonal via softplus()
            diag = nn.functional.softplus(torch.diagonal(L_full, dim1=-2, dim2=-1)) + EPS
            L = torch.tril(L_full)
            L = L/(torch.norm(L, dim=-1, keepdim=True) + EPS)
            L[:, torch.arange(self.domain_dim), torch.arange(self.domain_dim)] = diag.squeeze()
            L = L.unsqueeze(1)
            output = torch.cat([mu, L], dim=-1)
        return output


class UNet_diffusion_sample(nn.Module):
    def __init__(self, backbone, d=1, target_dim=1, hidden_dim=32, n_samples=50):
        super(UNet_diffusion_sample, self).__init__()
        self.backbone = backbone
        self.n_samples = n_samples

    def forward(self, x_t, t, condition_input, n_samples=None, **kwargs):
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
    def __init__(self, backbone, d=1, target_dim=1, n_components=3):
        super(UNet_diffusion_mixednormal, self).__init__()
        self.backbone = backbone
        hidden_dim = backbone.features
        self.n_components = n_components
        self.target_dim = target_dim

        if d == 1:
            self.mu_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel = 3,
            )
            self.sigma_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel = 3,
                init_bias=1,
            )
            self.weights_projection = Conv1d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel = 3,
            )
        elif d == 2:
            self.mu_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel = 3,
            )
            self.sigma_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel = 3,
                init_bias=1,
            )
            self.weights_projection = Conv2d(
                in_channels=hidden_dim,
                out_channels=target_dim * n_components,
                kernel = 3,
            )

        self.sofplus = nn.Softplus()

    def forward(self, x_t, t, condition_input, **kwargs):
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
            weights = torch.amax(weights, dim = (-2), keepdims = True).repeat(1,1,mu.shape[-2],1)
        elif len(weights.shape) == 5:
            weights = torch.amax(weights, dim = (-2,-3), keepdims = True).repeat(1,1,mu.shape[-3],mu.shape[-2],1)
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
        domain_dim=(1,128),
    )
    unet = UNet_diffusion_mixednormal(backbone, d=1, target_dim=1, n_components=3)
    test = unet.forward(input, t, condition)
    print(test.shape)

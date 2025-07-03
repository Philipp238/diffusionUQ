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

EPS = 1e-9


class UNet1d(nn.Module):
    def __init__(self, in_channels=2, init_features=32):
        super().__init__()

        features = init_features
        self.encoder1 = UNet1d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1d._block(features * 2, features, name="dec1")

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class UNet2d(nn.Module):
    def __init__(self, in_channels=2, init_features=32):
        super().__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class UNetDiffusion(nn.Module):
    def __init__(
        self,
        d=1,
        conditioning_dim=2,
        hidden_channels=16,
        in_channels=1,
        out_channels=1,
        init_features=32,
        device="cuda",
    ):
        super().__init__()
        self.d = d
        self.device = device
        self.hidden_dim = hidden_channels
        self.features = init_features
        self.input_projection = nn.Linear(
            in_channels, hidden_channels
        )  # the dimension of the target, is the dimension of the input of this MLP
        self.time_projection = nn.Linear(hidden_channels, hidden_channels)
        if d == 1:
            self.unet = UNet1d(3 * hidden_channels, init_features)
            self.output_projection = nn.Conv1d(
                in_channels=init_features, out_channels=out_channels, kernel_size=1
            )
        elif d == 2:
            self.unet = UNet2d(3 * hidden_channels, init_features)
            self.output_projection = nn.Conv2d(
                in_channels=init_features, out_channels=out_channels, kernel_size=1
            )
        else:
            raise NotImplementedError("Only 1D U-Net is implemented in this example.")

        if conditioning_dim:
            self.conditioning_projection = nn.Linear(conditioning_dim, hidden_channels)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat_interleave(channels // 2, dim=-1) * inv_freq)
        pos_enc_b = torch.cos(t.repeat_interleave(channels // 2, dim=-1) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward_body(self, x_t, t, condition_input, **kwargs):
        # x_t and condition input have shape [B, C, D1,..., DN]
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.hidden_dim)
        t = self.time_projection(t)

        # Reorder channel dimensions to last dimension
        x_t = torch.swapaxes(x_t, 1, -1)
        condition_input = torch.swapaxes(condition_input, 1, -1)

        # Projection
        x_t = self.input_projection(x_t)
        condition_input = self.conditioning_projection(condition_input)

        if self.d == 1:
            t = torch.repeat_interleave(t.unsqueeze(1), x_t.shape[1], dim=1)
        elif self.d == 2:
            t = t.unsqueeze(1).unsqueeze(1).repeat((1, x_t.shape[1], x_t.shape[2], 1))

        # Concatenate
        x_t = torch.cat([x_t, condition_input, t], dim=-1).to(x_t.device)
        # Reorder back to [B, C, D]
        x_t = torch.swapaxes(x_t, 1, -1)
        x_t = self.unet(x_t)
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
            self.mu_projection = nn.Conv1d(
                in_channels=hidden_dim, out_channels=target_dim, kernel_size=1
            )
            self.sigma_projection = nn.Conv1d(
                in_channels=hidden_dim, out_channels=target_dim, kernel_size=1
            )
        elif d == 2:
            self.mu_projection = nn.Conv2d(
                in_channels=hidden_dim, out_channels=target_dim, kernel_size=1
            )
            self.sigma_projection = nn.Conv2d(
                in_channels=hidden_dim, out_channels=target_dim, kernel_size=1
            )
        self.sofplus = nn.Softplus()

    def forward(self, x_t, t, condition_input, **kwargs):
        x_t = self.backbone.forward_body(x_t, t, condition_input)

        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        sigma = self.sofplus(sigma) + EPS
        output = torch.stack([mu, sigma], dim=-1)
        return output
    

class UNet_diffusion_sample(nn.Module):
    def __init__(self, backbone, d=1, target_dim = 1, hidden_dim=32, n_samples = 50):
        super(UNet_diffusion_sample, self).__init__()
        self.backbone = backbone

        # Concatenate noise with channel
        self.backbone.input_projection = nn.Linear(target_dim+1, hidden_dim)

        self.n_samples = n_samples

    def forward(self, x_t, t,condition_input, n_samples = None, **kwargs):
        if n_samples is None:
            n_samples = self.n_samples

        x_t_expanded = torch.repeat_interleave(x_t.unsqueeze(1), n_samples, dim=1).reshape(
            x_t.shape[0]* n_samples,*x_t.shape[1:]
        )
        t_expanded = torch.repeat_interleave(t.unsqueeze(-1), n_samples, dim=-1).reshape(t.shape[0]*n_samples)
        condition_input_expanded = torch.repeat_interleave(
            condition_input.unsqueeze(1), n_samples, dim=1
        ).reshape(
            condition_input.shape[0]* n_samples,*condition_input.shape[1:]
        )

        # Concatenate noise
        noise = torch.randn_like(x_t_expanded)
        x_t_expanded = torch.cat([x_t_expanded, noise], dim=1).to(x_t.device)

        output =  self.backbone.forward(
            x_t_expanded,            
            t_expanded,
            condition_input_expanded,
        )
        output =  output.reshape(
            x_t.shape[0], n_samples, *output.shape[1:]
        )
        return torch.moveaxis(output, 1, -1)  # Move sample dimension to last position
    

class UNet_diffusion_mixednormal(nn.Module):
    def __init__(self, backbone, d=1, target_dim = 1, n_components = 3):
        super(UNet_diffusion_mixednormal, self).__init__()
        self.backbone = backbone
        hidden_dim = backbone.features
        self.n_components = n_components
        self.target_dim = target_dim

        if d == 1:
            self.mu_projection = nn.Conv1d(
                in_channels=hidden_dim, out_channels=target_dim*n_components, kernel_size=1
            )
            self.sigma_projection = nn.Conv1d(
                in_channels=hidden_dim, out_channels=target_dim*n_components, kernel_size=1
            )
            self.weights_projection = nn.Conv1d(
                in_channels=hidden_dim, out_channels=target_dim*n_components, kernel_size=1
            )
        elif d == 2:
            self.mu_projection = nn.Conv2d(
                in_channels=hidden_dim, out_channels=target_dim*n_components, kernel_size=1
            )
            self.sigma_projection = nn.Conv2d(
                in_channels=hidden_dim, out_channels=target_dim*n_components, kernel_size=1
            )
            self.weights_projection = nn.Conv2d(
                in_channels=hidden_dim, out_channels=target_dim*n_components, kernel_size=1
            )

        self.sofplus = nn.Softplus()

    def forward(self, x_t, t, condition_input, **kwargs):
        x_t = self.backbone.forward_body(x_t, t, condition_input)

        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        weights = self.weights_projection(x_t)
        # Reshape
        mu = mu.reshape(mu.shape[0], self.target_dim, self.n_components, *mu.shape[2:]).moveaxis(2,-1)
        sigma = sigma.reshape(sigma.shape[0], self.target_dim, self.n_components, *sigma.shape[2:]).moveaxis(2,-1)
        weights = weights.reshape(weights.shape[0], self.target_dim, self.n_components, *weights.shape[2:]).moveaxis(2,-1)

        # Apply postprocessing
        sigma = self.sofplus(torch.clamp(sigma, min = -15)) + EPS
        weights = torch.softmax(torch.clamp(weights, min = -15, max = 15), dim=-1)

        output = torch.stack([mu, sigma, weights], dim=-1)
        return output


if __name__ == "__main__":
    input = torch.randn(8, 1, 128,128)
    condition = torch.randn(8, 2, 128,128)
    output = torch.randn(8, 1, 128,128)
    t = torch.ones(8) * 0.5

    backbone = UNetDiffusion(
        d=2,
        conditioning_dim=2,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        init_features=32,
        device="cpu",
    )
    unet = UNet_diffusion_normal(backbone, d=2, target_dim=1)
    test = unet.forward(input, t, condition)
    print(test.shape)
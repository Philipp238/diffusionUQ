import math

import torch
import torch.nn as nn

from models.mlp import MLP
from models.unet import UNetDiffusion


class MLP_crps_ensemble(nn.Module):
    """MLP with noise injection at the pre-final layer, trained with CRPS loss.

    Produces an ensemble of predictions by injecting different noise realizations
    into the hidden representation before the output projection.
    """

    def __init__(
        self,
        target_dim=1,
        conditioning_dim=None,
        hidden_dim=128,
        layers=5,
        dropout=0.1,
        noise_dim=10,
        n_samples=20,
    ):
        super().__init__()
        if isinstance(target_dim, tuple):
            target_dim = math.prod(target_dim)
        if isinstance(conditioning_dim, tuple):
            conditioning_dim = math.prod(conditioning_dim)

        self.noise_dim = noise_dim
        self.n_samples = n_samples

        self.input_projection = nn.Linear(conditioning_dim, hidden_dim)
        self.act = nn.ReLU()
        self.blocks = nn.Sequential(
            *[MLP.MLPBlock(hidden_dim, dropout) for _ in range(layers)]
        )
        self.output_projection = nn.Linear(hidden_dim + noise_dim, target_dim)

    def forward(self, x, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples

        h = self.input_projection(x)
        h = self.act(h)
        h = self.blocks(h)  # (*batch_dims, hidden_dim)

        # Expand for n_samples
        h = h.unsqueeze(-1).expand(*h.shape, n_samples)  # (*batch_dims, hidden_dim, n_samples)

        # Inject noise
        noise = torch.randn(
            *h.shape[:-2], self.noise_dim, n_samples, device=h.device
        )  # (*batch_dims, noise_dim, n_samples)
        h = torch.cat([h, noise], dim=-2)  # (*batch_dims, hidden_dim + noise_dim, n_samples)

        # Project each sample
        h = torch.movedim(h, -1, -2)  # (*batch_dims, n_samples, hidden_dim + noise_dim)
        output = self.output_projection(h)  # (*batch_dims, n_samples, target_dim)
        output = torch.movedim(output, -2, -1)  # (*batch_dims, target_dim, n_samples)

        return output


class UNet_crps_ensemble(nn.Module):
    """UNet with noise injection at the pre-final layer, trained with CRPS loss.

    Wraps a UNetDiffusion backbone. Passes dummy x_t and t internally,
    injects noise into the feature map before the output conv projection.
    """

    def __init__(
        self,
        backbone: UNetDiffusion,
        noise_dim=10,
        n_samples=20,
        d=1,
    ):
        super().__init__()
        self.backbone = backbone
        self.noise_dim = noise_dim
        self.n_samples = n_samples
        self.d = d

        features = backbone.features
        if d == 1:
            self.output_projection = nn.Conv1d(
                in_channels=features + noise_dim,
                out_channels=1,
                kernel_size=3,
                padding="same",
            )
        elif d == 2:
            self.output_projection = nn.Conv2d(
                in_channels=features + noise_dim,
                out_channels=1,
                kernel_size=3,
                padding="same",
            )

    def forward(self, condition_input, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples

        B = condition_input.shape[0]
        spatial = condition_input.shape[2:]  # spatial dimensions (D1, ..., DN)

        # Create dummy x_t and t for the backbone
        x_t = torch.zeros(B, 1, *spatial, device=condition_input.device)
        t = torch.zeros(B, device=condition_input.device)

        # Get hidden features from backbone
        h = self.backbone.forward_body(x_t, t, condition_input)  # (B, features, *spatial)

        # Expand for n_samples
        h_expanded = h.unsqueeze(1).expand(
            B, n_samples, -1, *spatial
        )  # (B, n_samples, features, *spatial)

        # Generate noise matching spatial dims
        noise = torch.randn(
            B, n_samples, self.noise_dim, *spatial, device=h.device
        )  # (B, n_samples, noise_dim, *spatial)

        # Concatenate noise to features
        h_noisy = torch.cat(
            [h_expanded, noise], dim=2
        )  # (B, n_samples, features+noise_dim, *spatial)

        # Reshape for conv: merge batch and samples
        h_flat = h_noisy.reshape(
            B * n_samples, -1, *spatial
        )  # (B*n_samples, features+noise_dim, *spatial)
        output = self.output_projection(h_flat)  # (B*n_samples, 1, *spatial)

        # Reshape back: (B, n_samples, 1, *spatial) -> (B, 1, *spatial, n_samples)
        output = output.reshape(B, n_samples, 1, *spatial)
        output = torch.movedim(output, 1, -1)  # (B, 1, *spatial, n_samples)

        return output


def generate_crps_samples(model, x, n_samples):
    """Generate samples from a CRPS ensemble model for evaluation.

    Args:
        model: CRPS ensemble model (MLP_crps_ensemble or UNet_crps_ensemble).
        x: Conditioning input tensor.
        n_samples: Number of ensemble samples to generate.

    Returns:
        Tensor of shape (B, *target_shape, n_samples).
    """
    model.eval()
    with torch.no_grad():
        samples = model(x, n_samples=n_samples)
    return samples

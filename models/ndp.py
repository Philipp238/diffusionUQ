import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding matching NDP's timestep_embedding."""
    device = t.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, dim)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class NDPAttentionBlock(nn.Module):
    """
    Attention block adapted from BiDimensionalAttentionBlock in NDP.

    Attention is computed over the token sequence (d_x feature tokens for tabular,
    or N_spatial point tokens for PDE tasks). Produces residual + skip outputs
    following the original NDP architecture (2H output split into two H halves).
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.time_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, s: torch.Tensor, t_emb: torch.Tensor):
        """
        Args:
            s: (B, N_tokens, H) token sequence
            t_emb: (B, H) timestep embedding
        Returns:
            s_out: (B, N_tokens, H) updated token sequence
            skip: (B, N_tokens, H) skip connection for output aggregation
        """
        t_proj = self.time_linear(t_emb).unsqueeze(1)  # (B, 1, H)
        y = s + t_proj  # (B, N_tokens, H)

        # Self-attention over token axis
        y_att, _ = self.attn(y, y, y)  # (B, N_tokens, H)
        y_att = self.out_proj(y_att)   # (B, N_tokens, 2H)

        # Split into residual and skip following NDP BiDimensionalAttentionBlock
        residual, skip = y_att.chunk(2, dim=-1)  # each (B, N_tokens, H)
        residual = F.gelu(residual)
        skip = F.gelu(skip)

        return (s + residual) / math.sqrt(2.0), skip


class NDP_diffusion(nn.Module):
    """
    Score network adapted from Neural Diffusion Processes
    (Dutordoir et al., ICML 2023, arXiv:2206.03992).

    Supports two operating modes:

    **Tabular mode** (UCI regression, target_dim scalar or (1,1)):
        Attention over input feature dimensions. Token_d = [feature_d, x_t].
        Skip connections are mean-pooled over the feature axis.
        Output: (B, 1, d_y).

    **Spatial mode** (1D PDE tasks, target_dim = (C, N_spatial) with N_spatial > 1):
        Attention over spatial points. Token_i = [cond_channels_i, x_t_channels_i].
        Per-token output projection produces a noise field.
        Output: (B, C_target, N_spatial).
        forward_body() returns (B, H) via spatial mean-pooling for MLP wrapper
        compatibility (normal, iDDPM distributional heads).
    """

    def __init__(
        self,
        target_dim: int | tuple = 1,
        conditioning_dim: int | tuple = None,
        hidden_dim: int = 128,
        n_heads: int = 4,
        layers: int = 4,
        dropout: float = 0.0,
        use_regressor_pred: bool = False,
        init_zero: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Detect spatial (PDE) vs. tabular (UCI) mode
        # Spatial mode: target_dim is a 2-tuple (C, N) with N > 1
        # ------------------------------------------------------------------
        if isinstance(target_dim, tuple) and len(target_dim) == 2 and target_dim[1] > 1:
            self.spatial_mode = True
            self.n_target_channels = target_dim[0]   # e.g., 1 for Burgers/KS
            self.n_spatial = target_dim[1]            # e.g., 128 spatial points
            n_cond = (
                conditioning_dim[0]
                if isinstance(conditioning_dim, tuple)
                else (conditioning_dim or 0)
            )
            self.n_cond_channels = n_cond
            token_input_dim = n_cond + self.n_target_channels
            # Flat dim used by external MLP wrapper heads: C * N
            self.target_dim = self.n_target_channels * self.n_spatial
            self.conditioning_dim = (
                n_cond * self.n_spatial
                if isinstance(conditioning_dim, tuple)
                else (conditioning_dim or 0)
            )
        else:
            # Tabular mode (original UCI behavior)
            self.spatial_mode = False
            self.n_spatial = None
            if isinstance(target_dim, tuple):
                target_dim_flat = math.prod(target_dim)
            else:
                target_dim_flat = target_dim if target_dim else 1
            if isinstance(conditioning_dim, tuple):
                cond_flat = math.prod(conditioning_dim)
            else:
                cond_flat = conditioning_dim or 0
            self.target_dim = target_dim_flat
            self.conditioning_dim = cond_flat
            self.n_target_channels = target_dim_flat
            self.n_cond_channels = cond_flat
            token_input_dim = 1 + target_dim_flat
            if use_regressor_pred:
                token_input_dim += target_dim_flat

        self.hidden_dim = hidden_dim
        self.n_layers = layers
        self.use_regressor_pred = use_regressor_pred

        # Token embedding: raw token features -> H
        self.token_embed = nn.Linear(token_input_dim, hidden_dim)

        # Timestep MLP: sinusoidal embedding -> H
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # NDP-style attention blocks
        self.blocks = nn.ModuleList([
            NDPAttentionBlock(hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(layers)
        ])

        # Output head
        # Tabular: global (mean-pooled) projection to d_y
        # Spatial: per-token projection to C_target (preserves spatial resolution)
        self.output_norm = nn.Linear(hidden_dim, hidden_dim)
        out_dim = self.n_target_channels if self.spatial_mode else self.target_dim
        self.output_head = nn.Linear(hidden_dim, out_dim)
        if init_zero:
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _tokenize(self, x_t: torch.Tensor, t: torch.Tensor, c=None, pred=None):
        """
        Tabular-mode tokenization (UCI regression).

        Follows NDP's process_inputs(): for each feature dimension d,
            token_d = [feature_d, x_t_0, ..., x_t_{dy-1}]
            d_y = 1 for scalar regression,

        Args:
            x_t: (B, 1, d_y) or (B, d_y) noisy target
            t: (B,) timestep
            c: (B, 1, d_x) or (B, d_x) conditioning features
            pred: (B, 1, d_y) or (B, d_y) regressor prediction (optional)

        Returns:
            tokens: (B, d_x, H) embedded token sequence
            t_emb: (B, H) timestep embedding
        """
        B = x_t.shape[0]
        x_t_flat = x_t.reshape(B, self.target_dim)  # (B, d_y)

        t_emb = sinusoidal_embedding(t, self.hidden_dim)  # (B, H)
        t_emb = self.time_mlp(t_emb)

        if self.conditioning_dim > 0:
            if c is not None:
                y_flat = c.reshape(B, self.conditioning_dim)
            else:
                y_flat = torch.zeros(B, self.conditioning_dim, device=x_t.device, dtype=x_t.dtype)

            # token_d = [y_d, x_t_0, ..., x_t_{dy-1}] for each feature dim d
            feature_vals = y_flat.unsqueeze(-1)  # (B, d_x, 1)
            x_t_rep = x_t_flat.unsqueeze(1).expand(B, self.conditioning_dim, self.target_dim)

            if pred is not None and self.use_regressor_pred:
                pred_flat = pred.reshape(B, self.target_dim)
                pred_rep = pred_flat.unsqueeze(1).expand(B, self.conditioning_dim, self.target_dim)
                token_inputs = torch.cat([feature_vals, x_t_rep, pred_rep], dim=-1)
            else:
                token_inputs = torch.cat([feature_vals, x_t_rep], dim=-1)  # (B, d_x, 1+d_y)
        else:
            pad_size = self.token_embed.in_features - self.target_dim
            token_inputs = x_t_flat.unsqueeze(1)
            if pad_size > 0:
                pad = torch.zeros(B, 1, pad_size, device=x_t.device, dtype=x_t.dtype)
                token_inputs = torch.cat([token_inputs, pad], dim=-1)

        tokens = F.gelu(self.token_embed(token_inputs))  # (B, d_x, H)
        return tokens, t_emb

    def _tokenize_spatial(self, x_t: torch.Tensor, t: torch.Tensor, c=None):
        """
        Spatial-mode tokenization (1D PDE tasks).

        One token per spatial point i:
            token_i = [cond_channels_i, x_t_channels_i]

        Args:
            x_t: (B, C_target, N_spatial)
            t: (B,) timestep
            c: (B, C_cond, N_spatial) conditioning (2 past states + spatial grid)

        Returns:
            tokens: (B, N_spatial, H)
            t_emb: (B, H)
        """
        B = x_t.shape[0]
        x_t_sp = x_t.permute(0, 2, 1)  # (B, N_spatial, C_target)

        t_emb = sinusoidal_embedding(t, self.hidden_dim)  # (B, H)
        t_emb = self.time_mlp(t_emb)

        if self.n_cond_channels > 0:
            if c is not None:
                y_sp = c.permute(0, 2, 1)  # (B, N_spatial, C_cond)
            else:
                y_sp = torch.zeros(
                    B, self.n_spatial, self.n_cond_channels,
                    device=x_t.device, dtype=x_t.dtype
                )
            token_inputs = torch.cat([y_sp, x_t_sp], dim=-1)  # (B, N_spatial, C_cond+C_target)
        else:
            token_inputs = x_t_sp  # (B, N_spatial, C_target)

        tokens = F.gelu(self.token_embed(token_inputs))  # (B, N_spatial, H)
        return tokens, t_emb

    # ------------------------------------------------------------------
    # Shared attention loop
    # ------------------------------------------------------------------

    def _run_blocks(self, tokens, t_emb):
        """Run all attention blocks and return (updated_tokens, skip_agg)."""
        skip_sum = None
        for block in self.blocks:
            tokens, skip = block(tokens, t_emb)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        return tokens, skip_sum / math.sqrt(self.n_layers)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward_body(self, x_t, t, y=None, pred=None):
        """
        Compute mean-pooled hidden representation: (B, H).

        Used by distributional wrapper heads (MLP_diffusion_normal, MLP_diffusion_iDDPM).
        In both tabular and spatial mode, tokens are mean-pooled over the token axis.
        """
        if self.spatial_mode:
            tokens, t_emb = self._tokenize_spatial(x_t, t, y)
        else:
            tokens, t_emb = self._tokenize(x_t, t, y, pred)

        _, skip_agg = self._run_blocks(tokens, t_emb)
        return skip_agg.mean(dim=1)  # (B, H)

    def forward(self, x_t, t, y=None, pred=None):
        """
        Predict noise.

        Tabular mode:
            x_t: (B, 1, d_y) -> returns (B, 1, d_y)
        Spatial mode:
            x_t: (B, C_target, N_spatial) -> returns (B, C_target, N_spatial)
        """
        if self.spatial_mode:
            tokens, t_emb = self._tokenize_spatial(x_t, t, y)
        else:
            tokens, t_emb = self._tokenize(x_t, t, y, pred)

        _, skip_agg = self._run_blocks(tokens, t_emb)

        if self.spatial_mode:
            # Per-token projection: (B, N_spatial, H) -> (B, N_spatial, C_target)
            per_token = F.gelu(self.output_norm(skip_agg))
            noise_pred = self.output_head(per_token)         # (B, N_spatial, C_target)
            return noise_pred.permute(0, 2, 1)               # (B, C_target, N_spatial)
        else:
            # Global: mean-pool then project
            pooled = F.gelu(self.output_norm(skip_agg.mean(dim=1)))  # (B, H)
            noise_pred = self.output_head(pooled)                     # (B, d_y)
            return noise_pred.unsqueeze(1)                            # (B, 1, d_y)

import torch
import torch.nn as nn

import torch.nn.functional as F

EPS = 1e-9

class Sequential2Inputs(nn.Sequential):
    def __init__(self, modules, concat=False):
        super().__init__(*modules)
        self.concat = concat
    
    def forward(self, x_t, t):
        if self.concat:
            x_t = torch.cat([x_t, t], dim=-1).to(x_t.device)
        else:
            x_t = x_t + t
        return super().forward(x_t)

class MLP_diffusion(nn.Module):
    
    class MLPBlock(nn.Module):
        def __init__(self, hidden_dim=128, concat=False, dropout=0.1):
            super().__init__()
            self.concat = concat
            if self.concat:
                hidden_dim = 2 * hidden_dim
            self.ff = nn.Linear(hidden_dim, hidden_dim)
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x_t):
            return self.dropout(self.act(self.ff(x_t)))
    
    def __init__(self, target_dim=1, conditioning_dim=None, concat=False, use_regressor_pred=False, hidden_dim=128, layers=5, dropout=0.1, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(target_dim, hidden_dim)  # the dimension of the target, is the dimension of the input of this MLP
        self.time_projection = nn.Linear(hidden_dim, hidden_dim)
        if conditioning_dim:
            self.conditioning_projection = nn.Linear(conditioning_dim, hidden_dim)

        if use_regressor_pred:
            self.regressor_pred_projection = nn.Linear(target_dim, hidden_dim)
        
        self.act = nn.ReLU()
        
        self.blocks = Sequential2Inputs([MLP_diffusion.MLPBlock(hidden_dim, concat=concat, dropout=dropout) for _ in range(layers)], concat=concat)
        
        if concat:
            self.output_projection = nn.Linear(2 * hidden_dim, target_dim)
        else:
            self.output_projection = nn.Linear(hidden_dim, target_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward_body(self, x_t, t, y=None, pred=None):
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.hidden_dim)
        t = self.time_projection(t)
        if y is not None:
            t += self.conditioning_projection(y)

        if pred is not None:
            t += self.regressor_pred_projection(pred)

        x_t = self.input_projection(x_t)    
        x_t = self.act(x_t)
        x_t = self.blocks(x_t, t)
        
        return x_t

    def forward(self, x_t, t, y=None, pred=None):
        output = self.forward_body(x_t, t, y, pred)
        eps_pred = self.output_projection(output)
        return eps_pred


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return F.softplus(out)
    

class MLP_diffusion_CARD(nn.Module):
    def __init__(self, target_dim=1, conditioning_dim=0, hidden_dim=128, layers=2, diffusion_timesteps=1000, use_regressor_guidance=False, device="cuda"):
        super(MLP_diffusion_CARD, self).__init__()
        self.device = device
        n_steps = diffusion_timesteps + 1
        self.conditioning_dim = conditioning_dim
        self.use_regressor_guidance = use_regressor_guidance

        data_dim = target_dim + conditioning_dim
        if use_regressor_guidance:
            data_dim += target_dim

        self.input_projection = ConditionalLinear(data_dim, hidden_dim, n_steps)
        hidden_layers = [
            ConditionalLinear(hidden_dim, hidden_dim, n_steps) for _ in range(layers)
        ]

        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_projection = nn.Linear(hidden_dim, 1)


    def forward_body(self, x_t, t, y=None, x_0_hat=None): 
        assert not self.use_regressor_guidance or (
            self.use_regressor_guidance and x_0_hat is not None
        )

        if self.conditioning_dim > 0:
            if y is None:
                y = torch.zeros((x_t.shape[0], self.conditioning_dim)).to(x_t.device)

            if x_0_hat is not None:
                eps_pred = torch.cat((x_t, y, x_0_hat), dim=1)
            else:
                eps_pred = torch.cat((x_t, y), dim=1)
        else:
            if x_0_hat is not None:
                eps_pred = torch.cat((x_t, x_0_hat), dim=1)
            else:
                eps_pred = x_t

        eps_pred = self.input_projection(eps_pred, t)

        for hidden_layer in self.hidden_layers:
            eps_pred = hidden_layer(eps_pred, t)
        
        return eps_pred
    
    def forward(self, x_t, t, y=None, x_0_hat=None): 
        eps_pred = self.forward_body(x_t, t, y, x_0_hat)
        return self.output_projection(eps_pred)
    
    
class MLP_diffusion_normal(nn.Module):
    def __init__(self, backbone, target_dim = 1, concat=False, hidden_dim=128):
        super(MLP_diffusion_normal, self).__init__()
        self.backbone = backbone
        if concat:
            self.mu_projection = nn.Linear(2 * hidden_dim, target_dim)
            self.sigma_projection = nn.Linear(2 * hidden_dim, target_dim)
        else:
            self.mu_projection = nn.Linear(hidden_dim, target_dim)
            self.sigma_projection = nn.Linear(hidden_dim, target_dim)
        self.sofplus = nn.Softplus()

    def forward(self, x_t, t, y=None, pred=None):
        x_t = self.backbone.forward_body(x_t, t, y, pred)

        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        sigma = self.sofplus(sigma) + EPS
        output = torch.stack([mu, sigma], dim=-1)
        return output
    

class MLP_diffusion_sample(MLP_diffusion):
    def __init__(self, target_dim = 1, conditioning_dim=None, concat=False, hidden_dim=128, layers=5, dropout=0.1, device="cuda", n_samples = 50):
        super().__init__(target_dim=target_dim, conditioning_dim=conditioning_dim, concat=concat, hidden_dim=hidden_dim, layers=layers, dropout=dropout, device=device)
        self.input_projection = nn.Linear(target_dim+1, hidden_dim)  # Concatenate noise with channel
        self.n_samples = n_samples

    def forward(self, x_t, t, y=None, pred=None, n_samples = None):
        # TODO: Implement regressor prediction guidance
        if n_samples is None:
            n_samples = self.n_samples
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.hidden_dim)
        t = self.time_projection(t)
        if y is not None:
            t += self.conditioning_projection(y)
        # Concatenate noise
        noise = torch.randn(*x_t.shape[:-1], n_samples, x_t.shape[-1]).to(x_t.device)
        x_t_expanded = torch.repeat_interleave(x_t.unsqueeze(-2), n_samples, dim=-2)
        t_expanded = torch.repeat_interleave(t.unsqueeze(-2), n_samples, dim=-2)
        x_t = torch.cat([x_t_expanded, noise], dim=-1).to(x_t.device)
        x_t = self.input_projection(x_t)
        x_t = self.act(x_t)
        x_t = self.blocks(x_t, t_expanded)

        output = self.output_projection(x_t)
        return output

class MLP_diffusion_mixednormal(MLP_diffusion):
    def __init__(self, backbone, target_dim = 1, concat=False, hidden_dim=128, n_components = 3):
        super(MLP_diffusion_mixednormal, self).__init__()
        self.backbone = backbone
        self.n_components = n_components
        self.target_dim = target_dim

        if concat:
            self.mu_projection = nn.Linear(2 * hidden_dim, target_dim*self.n_components)
            self.sigma_projection = nn.Linear(2 * hidden_dim, target_dim*self.n_components)
            self.weights_projection = nn.Linear(2 * hidden_dim, target_dim*self.n_components)
        else:
            self.mu_projection = nn.Linear(hidden_dim, target_dim*self.n_components)
            self.sigma_projection = nn.Linear(hidden_dim, target_dim*self.n_components)
            self.weights_projection = nn.Linear(hidden_dim, target_dim*self.n_components)
        self.sofplus = nn.Softplus()

    def forward(self, x_t, t, y=None, pred=None):
        x_t = self.backbone.forward_body(x_t, t, y, pred)

        mu = self.mu_projection(x_t)
        sigma = self.sigma_projection(x_t)
        weights = self.weights_projection(x_t)
        # Reshape
        mu = mu.reshape(mu.shape[0], self.target_dim, self.n_components)
        sigma = sigma.reshape(sigma.shape[0], self.target_dim, self.n_components)
        weights = weights.reshape(weights.shape[0], self.target_dim, self.n_components)

        # Apply postprocessing
        sigma = self.sofplus(sigma) + EPS
        weights = torch.softmax(weights, dim=-1)

        output = torch.stack([mu, sigma, weights], dim=-1)
        return output
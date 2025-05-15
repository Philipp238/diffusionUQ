import torch
import torch.nn as nn

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
    
    def __init__(self, target_dim=1, conditioning_dim=None, concat=False, hidden_dim=128, layers=5, dropout=0.1, device="cuda"):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(target_dim, hidden_dim)  # the dimension of the target, is the dimension of the input of this MLP
        self.time_projection = nn.Linear(hidden_dim, hidden_dim)
        if conditioning_dim:
            self.conditioning_projection = nn.Linear(conditioning_dim, hidden_dim)
        
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

    def forward(self, x_t, t, y=None):
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.hidden_dim)
        t = self.time_projection(t)
        if y is not None:
            t += self.conditioning_projection(y)
        x_t = self.input_projection(x_t)    
        x_t = self.act(x_t)
        x_t = self.blocks(x_t, t)

        output = self.output_projection(x_t)
        return output
    

class MLP_diffusion_normal(MLP_diffusion):
    def __init__(self, target_dim = 1, conditioning_dim=None, concat=False, hidden_dim=128, layers=5, dropout=0.1, device="cuda"):
        super().__init__(target_dim=target_dim, conditioning_dim=conditioning_dim, concat=concat, hidden_dim=hidden_dim, layers=layers, dropout=dropout, device=device)
        
        if concat:
            self.mu_projection = nn.Linear(2 * hidden_dim, target_dim)
            self.sigma_projection = nn.Linear(2 * hidden_dim, target_dim)
        else:
            self.mu_projection = nn.Linear(hidden_dim, target_dim)
            self.sigma_projection = nn.Linear(hidden_dim, target_dim)
        self.sofplus = nn.Softplus()

    def forward(self, x_t, t, y=None):
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.hidden_dim)
        t = self.time_projection(t)
        if y is not None:
            t += self.conditioning_projection(y)
        x_t = self.input_projection(x_t)    
        x_t = self.act(x_t)
        x_t = self.blocks(x_t, t)

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

    def forward(self, x_t, t, y=None, n_samples = None):
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
    def __init__(self, target_dim = 1, conditioning_dim=None, concat=False, hidden_dim=128, layers=5, dropout=0.1, device="cuda", n_components = 3):
        super().__init__(target_dim=target_dim, conditioning_dim=conditioning_dim, concat=concat, hidden_dim=hidden_dim, layers=layers, dropout=dropout, device=device)
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

    def forward(self, x_t, t, y=None):
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.hidden_dim)
        t = self.time_projection(t)
        if y is not None:
            t += self.conditioning_projection(y)
        x_t = self.input_projection(x_t)    
        x_t = self.act(x_t)
        x_t = self.blocks(x_t, t)

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
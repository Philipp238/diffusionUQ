import torch
import torch.nn as nn

class Sequential2Inputs(nn.Sequential):
    def __init__(self, modules, concat=False):
        super().__init__(*modules)
        self.concat = concat
    
    def forward(self, x_t, t):
        if self.concat:
            x_t = torch.cat([x_t, t], dim=1).to(x_t.device)
        else:
            x_t = x_t + t
        return super().forward(x_t)

class MLP_diffusion(nn.Module):
    
    class MLPBlock(nn.Module):
        def __init__(self, hidden_dim=128, concat=False, dropout=0.1):
            super().__init__()
            self.concat = concat
            if self.concat:
                input_dim = 2 * hidden_dim
            else:
                input_dim = hidden_dim
            self.ff = nn.Linear(input_dim, hidden_dim)
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
        
        self.blocks = Sequential2Inputs([MLP_diffusion.MLPBlock(hidden_dim, concat=concat, dropout=dropout) for _ in range(layers)])
        
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
    

class MLP_CRPS_diffusion(nn.Module):
    
    class MLPBlock(nn.Module):
        def __init__(self, hidden_dim=128, concat=False, dropout=0.1):
            super().__init__()
            self.concat = concat
            if self.concat:
                input_dim = 2 * hidden_dim
            else:
                input_dim = hidden_dim
            self.ff = nn.Linear(input_dim, hidden_dim)
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
        
        self.blocks = Sequential2Inputs([MLP_diffusion.MLPBlock(hidden_dim, concat=concat, dropout=dropout) for _ in range(layers)])
        
        self.output_projection_mu = nn.Linear(hidden_dim, target_dim)
        self.output_projection_sigma = nn.Linear(hidden_dim, target_dim)

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

        output_mu = self.output_projection_mu(x_t)
        output_sigma = nn.functional.softplus(self.output_projection_sigma(x_t))+1e-9
        output = torch.cat([output_mu, output_sigma], dim = 1)
        return output
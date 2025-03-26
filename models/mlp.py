from torch import nn

class MLP(nn.Module):
    
    class MLPBlock(nn.Module):
        def __init__(self, hidden_dim=128, dropout=0.1):
            super().__init__()
            self.ff = nn.Linear(hidden_dim, hidden_dim)
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            return self.dropout(self.act(self.ff(x)))
    
    def __init__(self, target_dim=1, conditioning_dim=None, hidden_dim=128, layers=5, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(conditioning_dim, hidden_dim)  # the dimension of the target, is the dimension of the input of this MLP
        self.act = nn.ReLU()
        
        self.blocks = nn.Sequential(*[MLP.MLPBlock(hidden_dim, dropout) for _ in range(layers)])
                
        self.output_projection = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.output_projection(x)
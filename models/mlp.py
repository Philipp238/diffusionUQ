from torch import nn
import math

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
        if isinstance(target_dim, tuple):
            target_dim = math.prod(target_dim)

        if isinstance(conditioning_dim, tuple):
            conditioning_dim = math.prod(conditioning_dim) 

        self.input_projection = nn.Linear(conditioning_dim, hidden_dim)  # the dimension of the target, is the dimension of the input of this MLP
        self.act = nn.ReLU()
        
        self.blocks = nn.Sequential(*[MLP.MLPBlock(hidden_dim, dropout) for _ in range(layers)])
                
        self.output_projection = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.output_projection(x)

class MLP_CARD(nn.Module):
    def __init__(self, input_dim, target_dim, hid_layers,
                 use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
        super(MLP_CARD, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)

        if isinstance(target_dim, tuple):
            target_dim = math.prod(target_dim) 
        self.input_dim = input_dim  # dimension of nn input
        self.target_dim = target_dim  # dimension of nn output
        self.hid_layers = hid_layers  # nn hidden layer architecture
        self.nn_layers = [self.input_dim] + self.hid_layers  # nn hidden layer architecture, except output layer
        self.use_batchnorm = use_batchnorm  # whether apply batch norm
        self.negative_slope = negative_slope  # negative slope for LeakyReLU
        self.dropout_rate = dropout_rate
        layers = self.create_nn_layers()
        self.network = nn.Sequential(*layers)

    def create_nn_layers(self):
        layers = []
        for idx in range(len(self.nn_layers) - 1):
            layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.nn_layers[-1], self.target_dim))
        return layers

    def forward(self, x):
        return self.network(x)
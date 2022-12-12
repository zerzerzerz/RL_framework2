import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout:float=0.0, acti:str='leaky_relu', norm:str='bn1d') -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.acti = acti
        self.norm = norm
        self.model = []

        self.add_layer(input_dim, hidden_dim)
        for _ in range(num_layers):
            self.add_layer(hidden_dim, hidden_dim)
        self.model.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*self.model)
        self.init()

    
    def add_layer(self, input_dim, output_dim):
        self.model.append(nn.Linear(input_dim, output_dim))

        if self.acti == 'leaky_relu':
            self.model.append(nn.LeakyReLU(0.2))
        else:
            raise NotImplementedError

        if self.norm is None:
            pass
        elif self.norm == 'bn1d':
            self.model.append(nn.BatchNorm1d(output_dim))
        else:
            raise NotImplementedError

        self.model.append(nn.Dropout(self.dropout))
    

    def init(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
    
    
    def forward(self, x):
        return self.model(x)
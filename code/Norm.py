import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class Norm(nn.Module):
    def __init__(self, hidden_dim=300, print_info=None):
        super(Norm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(hidden_dim))
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.__eps = 10e-10 #Numerical Stability


    def forward(self, x, batch):        
        per_graph_mean = scatter_mean(x, index=batch, dim=0) 
        shifted = x - self.alpha * per_graph_mean[batch]
        sigma_2 = scatter_mean(torch.pow(shifted, 2), index=batch, dim=0) + self.__eps
        return self.weight * shifted / torch.sqrt(sigma_2[batch]) + self.bias

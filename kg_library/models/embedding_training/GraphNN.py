import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, Linear

class GraphNN(nn.Module):
    def __init__(self, metadata, input_dim, hidden_dim, output_dim):
        super().__init__()

    def forward(self, x_dict, edge_index_dict):
        pass
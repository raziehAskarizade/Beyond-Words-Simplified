# Fardin Rastakhiz @ 2023



from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm

class HeteroGat(nn.Module):
    
    def __init__(self, in_feature, out_feature, dropout = 0.2, num_heads: int = 1, use_norm=True, use_relu=True, use_dropout=True) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(in_feature, int(out_feature/num_heads), heads=num_heads, edge_dim=1, add_self_loops=False)
        self.batch_norm = BatchNorm(out_feature)
        self.dropout= nn.Dropout(dropout)
        self.use_norm = use_norm
        self.use_relu = use_relu
        self.use_dropout = use_dropout

    def forward(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor) -> Tensor:

        x = self.conv1(x, edge_index, edge_weights)
        if self.use_norm: 
            x = self.batch_norm(x)
        if self.use_relu: 
            x = F.leaky_relu(x)
        if self.use_dropout: 
            x = self.dropout(x)
        return x
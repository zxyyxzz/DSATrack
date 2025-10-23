import numpy as np
import scipy.sparse as s
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class DynamicGraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, feature_size):
        super(DynamicGraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.size = feature_size*feature_size
        self.dynamic_adjacency_matrix_pred = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 2),
            nn.LogSoftmax(dim=-1)
        )
    def forward(self, features, features_mean):
        B, S, L = features_mean.shape

        corr = torch.bmm(features_mean, features_mean.permute(0, 2, 1)).flatten(1) # [B, S * S]
        dynamic_adj = self.dynamic_adjacency_matrix_pred(corr.unsqueeze(-1)) # [B, S * S, 2]
        dynamic_adj = F.gumbel_softmax(dynamic_adj, hard=True)[:, :, 0].view(-1, S, S) # [B, S, S]

        eye = torch.eye(S, dtype=dynamic_adj.dtype, device=dynamic_adj.device).view(1, S, S) # [1, S, S]
        dynamic_adj = dynamic_adj + (1.0 - dynamic_adj) * eye

        # Compute the degree matrix and its inverse
        degree_matrix_inv = torch.reciprocal(torch.sum(dynamic_adj, dim=-1))

        # Compute the normalized adjacency matrix
        dynamic_normalized_adjacency = degree_matrix_inv.unsqueeze(-1) * dynamic_adj  # [B, S, S]

        x = torch.matmul(dynamic_normalized_adjacency.unsqueeze(1), features)
        x = self.linear(x)
        return x

class Gnn4d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 feature_size_z,
                 feature_size_x,
                 ):
        super().__init__()
        self.supp_conv = DynamicGraphConvolution(in_channels, out_channels, feature_size_z)
    def forward(self, x):
        """
        x: Hyper correlation map.
            shape: B L H_q W_q H_s W_s
        """
        # Query : Search
        # Support: Template
        B, L, NUM_S, S, Q = x.size()

        supp_x = x.clone()
        supp_x_mean = rearrange(supp_x, 'B L NUM_S S Q -> B (NUM_S S) L Q').mean(-1)
        supp_x = rearrange(supp_x, 'B L NUM_S S Q -> B Q (NUM_S S) L')
        supp_x = self.supp_conv(supp_x, supp_x_mean)
        supp_x = rearrange(supp_x, 'B Q (NUM_S S) L ->B L NUM_S S Q', NUM_S=NUM_S)
        return supp_x


class Encoder4Dgnn(nn.Module):
    def __init__(self,
                 corr_levels,
                 layer_num,
                 feature_size_z,
                 feature_size_x,
                 residual=True,
                 ):
        super().__init__()
        self.gnn4d = nn.ModuleList([])
        for i in range(layer_num):
            gnn4d = nn.Sequential(
                Gnn4d(corr_levels[i], corr_levels[i + 1], feature_size_z, feature_size_x),
                nn.GroupNorm(corr_levels[i], corr_levels[i + 1]),
                #nn.ReLU()  # No inplace for residual
            )
            self.gnn4d.append(gnn4d)

        self.residual = residual

    def forward(self, x):
        """
        x: Hyper correlation. B L H_q W_q H_s W_s
        """

        residuals = x.clone()
        for gnn in self.gnn4d:
            x = gnn(x)
        if self.residual:
            x = x + residuals
        return x
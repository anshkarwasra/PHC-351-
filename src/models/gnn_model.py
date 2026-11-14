"""Residual GraphSAGE encoder with a classification head and a masked regression head."""
import math

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class ResSAGEBlock(nn.Module):
    """Pre-norm residual block: SAGE message passing, then a feed-forward sub-block."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.norm_conv = nn.LayerNorm(dim)
        self.conv = SAGEConv(dim, dim, aggr="mean")
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index):
        h = h + self.drop(self.conv(self.norm_conv(h), edge_index))
        h = h + self.drop(self.ff(self.norm_ff(h)))
        return h


def _head(dim, dropout):
    return nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(dim // 2, 1))


class MeshQualityGNN(nn.Module):
    """
    Element dual graph in, two per-element outputs out:
      logit: needs-correction logit (apply sigmoid for probability)
      reg:   suggested smoothing weight in [0, 1]; only meaningful where flagged
    Default hidden_dim=64, n_blocks=3 gives roughly 85k parameters.
    """

    def __init__(self, in_channels=8, hidden_dim=64, n_blocks=3, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResSAGEBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.cls_head = _head(hidden_dim, dropout)
        self.reg_head = _head(hidden_dim, dropout)

    def set_cls_prior(self, pos_rate):
        """Initialise the classifier bias to the training positive rate.
        Without this, the untrained model starts at p=0.5 and wastes early epochs."""
        p = min(max(pos_rate, 1e-4), 1 - 1e-4)
        nn.init.constant_(self.cls_head[-1].bias, math.log(p / (1 - p)))

    def forward(self, x, edge_index):
        h = self.encoder(x)
        for blk in self.blocks:
            h = blk(h, edge_index)
        h = self.final_norm(h)
        logit = self.cls_head(h).squeeze(-1)
        reg = torch.sigmoid(self.reg_head(h)).squeeze(-1)
        return logit, reg
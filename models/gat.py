import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GATConv
from models.base import ModelBase


class GAT(ModelBase):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):
        super(GAT, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)
        self.num_layers = config['num_layers']
        self.num_hidden = config['num_hidden']
        self.num_head = config['num_head']
        self.negative_slope = config['negative_slope']
        self.dropout = config['dropout']

        self.layers = nn.ModuleList()
        self.layers.append(GATConv(
            in_channels=self.pyg_data.num_features, out_channels=self.num_hidden,
            heads=self.num_head, negative_slope=self.negative_slope,
            dropout=self.dropout, bias=self.with_bias))
        for _ in range(self.num_layers-2):
            self.layers.append(GATConv(
                in_channels=self.num_hidden*self.num_head, out_channels=self.num_hidden,
                heads=self.num_head, negative_slope=self.negative_slope,
                dropout=self.dropout, bias=self.with_bias))
        self.layers.append(GATConv(
            in_channels=self.num_hidden*self.num_head, out_channels=self.num_hidden,
            heads=1, negative_slope=self.negative_slope, concat=False,
            dropout=self.dropout, bias=self.with_bias))

        if self.with_bn:
            self.bn_layers = nn.ModuleList()
            self.bn_layers.append(nn.BatchNorm1d(self.num_hidden))
            for _ in range(self.num_layers-2):
                self.bn_layers.append(nn.BatchNorm1d(self.num_hidden*self.num_head))


    def initialize(self):
        for conv in self.layers:
            conv.reset_parameters()
        if self.with_bn:
            for bn in self.bn_layers:
                bn.reset_parameters()

    def _forward(self, x, mat, weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers - 1):
            x = self.layers[i](x, mat, weight)
            if self.with_bn:
                x = self.bn_layers[i](x)
            if self.with_relu:
                x = F.elu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, mat, weight)
        return x

    def forward(self, x, mat, weight=None):
        x = self._forward(x, mat, weight)
        return F.log_softmax(x, dim=1)
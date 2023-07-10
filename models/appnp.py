
from models.base import ModelBase
from torch_geometric.nn.conv import APPNP as APPNPConv
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F


class APPNP(ModelBase):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False, dense_conv=False):

        super(APPNP, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)

        self.K = config['K']
        self.alpha = config['alpha']
        self.n_hidden = config['num_hidden']

        nfeat = self.pyg_data.num_features
        nclass = self.pyg_data.num_classes

        self.ln1 = Linear(nfeat, self.n_hidden)
        if self.with_bn:
            self.bn1 = nn.BatchNorm1d(self.n_hidden)
            self.bn2 = nn.BatchNorm1d(nclass)

        self.ln2 = Linear(self.n_hidden, nclass)
        self.prop1 = APPNPConv(self.K, self.alpha)



    def initialize(self):
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        if self.with_bn:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

    def _forward(self, x, mat, weight=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ln1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ln2(x)
        if self.with_bn:
            x = self.bn2(x)
        x = self.prop1(x, mat, weight)
        return x

    def forward(self, x, mat, weight=None):
        x = self._forward(x, mat, weight)
        return F.log_softmax(x, dim=1)
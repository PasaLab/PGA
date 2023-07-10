import torch.nn.functional as F

from torch_geometric.nn.conv import SGConv
from models.base import ModelBase


class SGC(ModelBase):

    def __init__(self,
                 config, pyg_data, device, logger, with_relu=True, with_bias=True, with_bns=False):
        super(SGC, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bns)
        self.K = self.config['K']
        self.layer = SGConv(self.pyg_data.num_features, self.pyg_data.num_classes, K=self.K, bias=self.with_bias)

    def initialize(self):
        self.layer.reset_parameters()

    def _forward(self, x, mat, weight=None):
        return self.layer(x, mat, weight)

    def forward(self, x, mat, weight=None):
        x = self._forward(x, mat, weight)
        return F.log_softmax(x, dim=1)
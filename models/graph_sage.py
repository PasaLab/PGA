
import torch.nn as nn
from models.gcn import GCN
from torch_geometric.nn.conv import SAGEConv



class GraphSage(GCN):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):
        super(GraphSage, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(self.pyg_data.num_features, self.config['num_hidden'], bias=self.with_bias))
        for _ in range(self.num_layers - 2):
            self.layers.append(SAGEConv(self.config['num_hidden'], self.config['num_hidden'], bias=self.with_bias))
        self.layers.append(SAGEConv(self.config['num_hidden'], self.pyg_data.num_classes, bias=self.with_bias))


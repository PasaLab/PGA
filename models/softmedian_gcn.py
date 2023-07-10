from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv import MessagePassing

# This works for higher version of torch_gometric, e.g., 2.0.
from torch_geometric.nn.dense.linear import Linear
# from torch.nn import Linear

from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import remove_self_loops, add_self_loops

from models.gcn import GCN



class SoftMedianConv(MessagePassing):

    def __init__(self, in_channels, out_channels, add_self_loops=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SoftMedianConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_weight):
        # apply the exponential function to edge weights
        # which yields a "softmin" instead of a sum of activations
        if edge_weight is not None:
            edge_weight = torch.exp(-edge_weight)

        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, x_j, index, ptr, dim_size):
        # calculate the softmedian
        x_j = x_j.view(-1, self.out_channels)
        softmin = torch.softmax(-x_j, dim=0)
        softmedian = torch.sum(x_j * softmin, dim=0)

        # convert the softmedian to a dense tensor
        out = torch.zeros((dim_size, self.out_channels), device=x_j.device)
        out.index_add_(0, index, softmedian[ptr[1:] - 1])

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class SoftMedianGCN(GCN):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):
        super(SoftMedianGCN, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)
        self.layers = torch.nn.ModuleList()
        self.layers.append(SoftMedianConv(self.pyg_data.num_features, self.config['num_hidden'], bias=self.with_bias))
        for _ in range(self.num_layers - 2):
            self.layers.append(SoftMedianConv(self.config['num_hidden'], self.config['num_hidden'], bias=self.with_bias))
        self.layers.append(SoftMedianConv(self.config['num_hidden'], self.pyg_data.num_classes, bias=self.with_bias))

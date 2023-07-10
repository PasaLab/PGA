import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear


class MlpBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5, bias=True):
        super(MlpBlock, self).__init__()
        self.fc1 = Linear(in_channels, hidden_channels, bias=bias)
        self.gelu = nn.GELU()
        self.fc2 = Linear(hidden_channels, in_channels, bias=bias)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.dropout(x, self.dropout, train=self.training)
        x = self.gelu(self.fc2(x))
        return x


class MixBlock(nn.Module):
    def __init__(self, batch_size, dimension_size, batch_hidden_dim, dimension_hidden_dim, dropout=0.5):
        super(MixBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(dimension_size)
        self.batch_mlp_block = MlpBlock(
            in_channels=batch_size,
            hidden_channels=batch_hidden_dim,
            dropout=dropout)
        self.dimension_mlp_block = MlpBlock(
            in_channels=dimension_size,
            hidden_channels=dimension_hidden_dim,
            dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.batch_mlp_block.reset_parameters()
        self.dimension_mlp_block.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, feature_mat):
        x = self.layer_norm(feature_mat)
        x = x.transpose(0, 1)
        x = self.batch_mlp_block(x)
        x = x.transpose(0, 1)
        out = feature_mat + x
        y = self.layer_norm(out)
        y = self.dimension_mlp_block(y)
        y = out + y
        return y



class MixMLP(nn.Module):
    """
    batch_size: the maximum number of neighbors
    dimension_size: feature dimension
    """
    def __init__(self,
                 batch_size, dimension_size,
                 batch_hidden_dim, dimension_hidden_dim,
                 num_blocks=2, dropout=0.5):
        super(MixMLP, self).__init__()
        self.num_blocks = num_blocks
        self.layer_norm = nn.LayerNorm(dimension_size)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(MixBlock(
                batch_size, dimension_size,
                batch_hidden_dim, dimension_hidden_dim,
                dropout,
            ))
        self.reset_parameters()

    def reset_parameters(self):
        self.layer_norm.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()

    def forward(self, feature_mat):
        emb = feature_mat
        for block in self.blocks:
            emb = block(emb)
        emb = self.layer_norm(emb)
        return emb








class DemoConv(MessagePassing):

    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(DemoConv, self).__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels,
                                   bias=False,
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
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        # normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.propagate(edge_index=edge_index, x=x, norm=norm)
        if self.bias is not None:
            x += self.bias
        return x


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


from models.base import ModelBase
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix


class DemoModel(ModelBase):

    def __init__(self, config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):
        super(DemoModel, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)
        self.num_layers = self.config['num_layers']

        node_deg = calculate_degree(pyg_data.adj_t)
        max_node_deg = node_deg.max(dim=0).values.item()
        print(max_node_deg)

        conv = DemoConv
        self.layers = nn.ModuleList()
        self.layers.append(conv(self.pyg_data.num_features, self.config['num_hidden'], bias=with_bias))
        for _ in range(self.num_layers - 2):
            self.layers.append(conv(self.config['num_hidden'], self.config['num_hidden'], bias=with_bias))
        self.layers.append(conv(self.config['num_hidden'], self.pyg_data.num_classes, bias=with_bias))

        if with_bn:
            self.bn_layers = nn.ModuleList()
            self.bn_layers.append(nn.BatchNorm1d(self.config['num_hidden']))
            for _ in range(self.num_layers - 2):
                self.bn_layers.append(nn.BatchNorm1d(self.config['num_hidden']))

    def initialize(self):
        for conv in self.layers:
            conv.reset_parameters()
        if self.with_bn:
            for bn in self.bn_layers:
                bn.reset_parameters()

    def _forward(self, x, mat, weight=None):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x, mat, weight)
            if self.with_bn:
                x = self.bn_layers[i](x)
            if self.with_relu:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, mat, weight)
        return x

    def forward(self, x, mat, weight=None):
        x = self._forward(x, mat, weight)
        return F.log_softmax(x, dim=1)

    def predict(self, x=None, edge_index=None):

        self.eval()
        if x is None and edge_index is None:
            return self.forward(self.x, self.edge_index).detach()
        else:
            assert (type(edge_index) is torch.Tensor and edge_index.size(0) == 2) or (type(edge_index) is SparseTensor)
            assert type(x) is torch.Tensor
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            return self.forward(x, edge_index).detach()



if __name__ == '__main__':
    import os.path as osp
    import yaml
    from yaml import SafeLoader
    from common.utils import *
    config_file = osp.join(osp.expanduser('../victims/configs'), 'demo.yaml')
    config = yaml.load(open(config_file), Loader=SafeLoader)['cora']
    logger_filename = 'demo_test.log'
    logger_name = 'demo_test'
    logger = get_logger(logger_filename, 0, logger_name)

    device = get_device(0)
    seed = config['seed']
    freeze_seed(seed)

    pyg_data = load_data('cora')
    model = DemoModel(config=config, pyg_data=pyg_data, device=device, logger=logger)
    model = model.to(device)

    model.fit(pyg_data, adj_t=pyg_data.edge_index, verbose=True)
    model.test(pyg_data.test_mask)

    perturbed = load_perturbed_adj('cora', 'pga', 0.05, path='../attack/perturbed_adjs/')
    perturbed = perturbed['modified_adj_list'][0]
    check_undirected(perturbed)
    ptb_rate = calc_modified_rate(pyg_data.adj_t, perturbed, pyg_data.num_edges)
    assert ptb_rate <= 0.05

    perturbed, _ = from_scipy_sparse_matrix(perturbed.to_scipy(layout='coo'))
    attack_acc = evaluate_attack_performance(model, pyg_data.x, perturbed, pyg_data.y, pyg_data.test_mask)
    logger.debug(f"attacked accuracy: {attack_acc.item()}")
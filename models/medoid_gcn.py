

import torch
from torch import Tensor
from torch_geometric.nn.conv import GCNConv
from torch_sparse import SparseTensor
from models.gcn import GCN
from models._aggregations import ROBUST_MEANS



class MedoidConv(GCNConv):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(MedoidConv, self).__init__(in_channels, out_channels, **kwargs)
        self._mean = ROBUST_MEANS['k_medoid']

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return NotImplemented

    def propagate(self, edge_index: torch.Tensor, size=None, **kwargs) -> torch.Tensor:
        x = kwargs['x']
        return self._mean(edge_index, x)



def _distance_matrix(x: torch.Tensor, eps_factor=1e2) -> torch.Tensor:
    x_norm = (x ** 2).sum(1).view(-1, 1)
    x_norm_t = x_norm.transpose(0, 1)
    squared = x_norm + x_norm_t - (2 * (x @ x.transpose(0, 1)))
    # For "save" sqrt
    eps = eps_factor * torch.finfo(x.dtype).eps
    return torch.sqrt(torch.abs(squared) + eps)


def weighted_medoid(x, A):
    N, D = x.shape
    l2 = _distance_matrix(x)
    A_cpu_dense = A.cpu().to_dense()
    l2_cpu = l2.cpu()
    distances = A_cpu_dense[:, None, :].expand(N, N, N) * l2_cpu
    distances[A_cpu_dense == 0] = torch.finfo(distances.dtype).max
    distances = distances.sum(-1).to(x.device)
    distances[~torch.isfinite(distances)] = torch.finfo(distances.dtype).max
    row_sum = A_cpu_dense.sum(-1)[:, None].to(x.device)
    return row_sum * x[distances.argmin(-1)]


class MedoidGCN(GCN):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):
        super(MedoidGCN, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)
        self.layers = torch.nn.ModuleList()
        self.layers.append(MedoidConv(self.pyg_data.num_features, self.config['num_hidden'], bias=self.with_bias))
        for _ in range(self.num_layers - 2):
            self.layers.append(MedoidConv(self.config['num_hidden'], self.config['num_hidden'], bias=self.with_bias))
        self.layers.append(MedoidConv(self.config['num_hidden'], self.pyg_data.num_classes, bias=self.with_bias))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from numba import njit

from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_sparse import SparseTensor
from models.base import ModelBase

import deeprobust.graph.utils as _utils

from models.naotan_func import *



class DenseGCNConv(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(DenseGCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std_v = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std_v, std_v)
        if self.bias is not None:
            self.bias.data.uniform_(-std_v, std_v)

    def forward(self, x, adj, weight=None):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(ModelBase):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False, dense_conv=False):

        super(GCN, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)

        self.num_layers = self.config['num_layers']
        self.hidden_sizes = []

        conv = GCNConv if not dense_conv else DenseGCNConv
        self.layers = nn.ModuleList()
        self.layers.append(conv(self.pyg_data.num_features, self.config['num_hidden'], bias=with_bias))
        self.hidden_sizes.append(self.config['num_hidden'])
        for _ in range(self.num_layers - 2):
            self.layers.append(conv(self.config['num_hidden'], self.config['num_hidden'], bias=with_bias))
            self.hidden_sizes.append(self.config['num_hidden'])
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


class DenseGCN(GCN):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False, dense_conv=True):
        super(DenseGCN, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn, dense_conv)

    def fit(self, pyg_data, adj_t=None, train_iters=None, patience=None, initialize=True, verbose=False):

        if initialize:
            self.initialize()

        if train_iters is None:
            train_iters = self.config['num_epochs']
        if patience is None:
            patience = self.config['patience']

        # self.logger.debug(f"total training epochs: {train_iters}, patience: {patience}")

        self.x = pyg_data.x.to(self.device)
        if adj_t is None:
            self.edge_index = pyg_data.adj_t.to_dense().to(self.device)
        else:
            self.edge_index = adj_t.to_dense().to(self.device)

        self.edge_index = _utils.normalize_adj_tensor(self.edge_index)


        if patience < train_iters:
            self._train_with_early_stopping(train_iters, patience, verbose)
        else:
            self._train_with_val(train_iters, verbose)

    def predict(self, x=None, edge_index=None):

        self.eval()
        if x is None and edge_index is None:
            adj = self.edge_index
            if type(self.edge_index) is SparseTensor:
                adj = adj.to_dense()
            return self.forward(self.x, adj).detach()
        else:
            assert (type(edge_index) is torch.Tensor and edge_index.size[0] == 2) or (type(edge_index) is SparseTensor)
            assert type(x) is torch.Tensor
            x = x.to(self.device)
            adj = edge_index
            if type(self.edge_index) is not SparseTensor:
                adj = SparseTensor.from_edge_index(edge_index)
            adj = adj.to_dense()
            adj = adj.to(self.device)
            return self.forward(x, adj).detach()


class GNNGuard(GCN):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False, dense_conv=True):
        super(GNNGuard, self).__init__(
            config, pyg_data, device, logger,
            with_relu, with_bias, with_bn, dense_conv=dense_conv)
        self.pre_beta = nn.Parameter(torch.randn(1))
        self.prune_edges = False
        self.mimic_ref_impl = False
        self.dense_conv = dense_conv

    def _forward(self, x, mat, weight=None):
        beta = self.pre_beta.sigmoid()
        for idx, conv in enumerate(self.layers):
            Alpha = self._edge_weights(mat.to_dense(), x)
            if idx == 0:
                W = Alpha
            else:
                W = beta * W + (1 - beta) * Alpha
            del Alpha
            if not self.dense_conv:
                x = conv(x, SparseTensor.from_dense(W))
            else:
                x = conv(x, W)
            if idx != len(self.layers) - 1 and self.with_relu:
                x = F.relu(x)
            if idx != len(self.layers) - 1:
                x = F.dropout(x, self.dropout, training=self.training)
        return x

    def _edge_weights(self, A, X):
        # No gradients pass through the paper authors' implementation of this method.
        # Also, if we passed gradients through this, we would get unstable gradients due to the cosine distance.
        X = X.detach()
        # Build attention matrix from the pairwise cosine similarity matrix.
        cos = pairwise_cosine(X)
        if self.mimic_ref_impl:
            cos[cos < 0.1] = 0
        S = torch.mul(A, cos)
        del cos  # free memory
        # Normalize S, yielding Alpha as defined in the paper respectively the reference implementation.

        if not self.mimic_ref_impl:
            N = Sum(Neq0(S).int(), dim=-1, dense=True)
            S_sums = Sum(S, dim=-1, dense=True)
            S_sums[S_sums.abs() < 1e-8] = 1
            Alpha = torch.mul(S, (N / ((N + 1) * S_sums))[..., None])
        else:
            S_sums = Sum(torch.abs(S), dim=-1, dense=True)
            # Note: Taken from sklearn's normalize().
            S_sums[S_sums < 10 * torch.finfo(S_sums.dtype).eps] = 1
            Alpha = torch.div(S, S_sums[..., None])

        del S, S_sums  # free memory

        # Edge pruning
        if self.prune_edges:
            edges = Alpha.nonzero()
            char_vec = torch.vstack([Alpha[edges[:, 0], edges[:, 1]], Alpha[edges[:, 1], edges[:, 0]]])
            drop_score = (self.pruning_weight @ char_vec).sigmoid()
            Alpha[tuple(edges[drop_score <= 0.5].T)] = 0
        if self.mimic_ref_impl:
            N = Sum(Neq0(Alpha).int(), dim=-1, dense=True)
        Alpha = Alpha + Sp_diag(1 / (N + 1))
        del N  # free memory
        return Alpha


def truncatedSVD(data, k=50):
    if sp.issparse(data):
        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        diag_S = np.diag(S)

    return U @ diag_S @ V


class GCNSVD(GCN):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):
        super(GCNSVD, self).__init__(
            config, pyg_data, device, logger,
            with_relu, with_bias, with_bn, dense_conv=config['use_dense'])
        self.k = config['k']
        self.use_dense = config['use_dense']

    @staticmethod
    def truncatedSVD(adj, k=50):
        if type(adj) is not SparseTensor and adj.size(0) == adj.size(1):
            adj = SparseTensor.from_dense(adj)
        elif type(adj) is not SparseTensor and adj.size(0) == 2:
            assert True, "EdgeIndex is not supported now."

        row, col, values = adj.cpu().coo()
        if values is None:
            values = torch.ones(row.size(0), dtype=torch.float32)
        N = adj.size(0)

        low_rank_adj = sp.coo_matrix((values, (row, col)), (N, N))
        low_rank_adj = truncatedSVD(low_rank_adj, k)
        low_rank_adj = torch.from_numpy(low_rank_adj).to(adj.device(), adj.dtype())
        return low_rank_adj

    def fit(self,
            pyg_data, adj_t=None,
            train_iters=None, patience=None,
            initialize=True, verbose=False):
        adj = pyg_data.adj_t
        if adj_t is not None:
            adj = adj_t
        mod_adj = self.get_modified_adj(adj)
        super().fit(
            pyg_data, adj_t=mod_adj,
            train_iters=train_iters, patience=patience,
            initialize=initialize, verbose=verbose)

    def get_modified_adj(self, adj_t):
        mod_adj = GCNSVD.truncatedSVD(adj_t, k=self.k)
        if not self.use_dense:
            mod_adj = SparseTensor.from_dense(mod_adj)
        else:
            mod_adj = _utils.normalize_adj_tensor(mod_adj)
        return mod_adj

    def predict(self, x=None, edge_index=None):

        self.eval()
        if x is None and edge_index is None:
            return self.forward(self.x, self.get_modified_adj(self.edge_index)).detach()
        else:
            assert (type(edge_index) is torch.Tensor and edge_index.size[0] == 2) or (type(edge_index) is SparseTensor)
            assert type(x) is torch.Tensor
            x = x.to(self.device)
            adj = self.get_modified_adj(edge_index).to(self.device)
            return self.forward(x, adj).detach()


class GCNJaccard(GCN):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):
        super(GCNJaccard, self).__init__(
            config, pyg_data, device, logger,
            with_relu, with_bias, with_bn, dense_conv=False)
        self.binary_feature = config['binary_feature']
        self.threshold = config['threshold']

    def fit(self,
            pyg_data, adj_t=None,
            train_iters=None, patience=None,
            initialize=True, verbose=False):
        adj = pyg_data.adj_t
        if adj_t is not None:
            adj = adj_t
        mod_adj = self.get_modified_adj(adj)
        super().fit(
            pyg_data, adj_t=mod_adj,
            train_iters=train_iters, patience=patience,
            initialize=initialize, verbose=verbose)

    def predict(self, x=None, edge_index=None):
        self.eval()
        if x is None and edge_index is None:
            return self.forward(self.x, self.edge_index).detach()
        else:
            assert (type(edge_index) is torch.Tensor and edge_index.size[0] == 2) or (type(edge_index) is SparseTensor)
            assert type(x) is torch.Tensor
            x = x.to(self.device)
            adj = self.get_modified_adj(edge_index).to(self.device)
            return self.forward(x, adj).detach()

    def get_modified_adj(self, adj_t):
        mod_adj = self.drop_dissimilar_edges(adj_t)
        edge_index, edge_weight = from_scipy_sparse_matrix(mod_adj)
        mod_adj = SparseTensor.from_edge_index(
            edge_index=edge_index, edge_attr=edge_weight,
            sparse_sizes=(self.pyg_data.num_nodes, self.pyg_data.num_nodes))
        return mod_adj

    def drop_dissimilar_edges(self, adj, metric='similarity'):
        """Drop dissimilar edges.(Faster version using numba)
        """
        adj = adj.to_scipy(layout='csr')
        adj_triu = sp.triu(adj, format='csr')
        features = self.pyg_data.x.cpu().numpy()

        if metric == 'distance':
            removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                       threshold=self.threshold)
        else:
            if self.binary_feature:
                removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                               threshold=self.threshold)
            else:
                removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                              threshold=self.threshold)
        # print('removed %s edges in the original graph' % removed_cnt)
        modified_adj = adj_triu + adj_triu.transpose()
        return modified_adj

    # def _drop_dissimilar_edges(self, features, adj):
    #     """Drop dissimilar edges. (Slower version)
    #     """
    #     if not sp.issparse(adj):
    #         adj = sp.csr_matrix(adj)
    #     modified_adj = adj.copy().tolil()
    #
    #     # preprocessing based on features
    #     print('=== GCN-Jaccrad ===')
    #     edges = np.array(modified_adj.nonzero()).T
    #     removed_cnt = 0
    #     for edge in edges:
    #         n1 = edge[0]
    #         n2 = edge[1]
    #         if n1 > n2:
    #             continue
    #
    #         if self.binary_feature:
    #             J = _jaccard_similarity(features[n1], features[n2])
    #
    #             if J < self.threshold:
    #                 modified_adj[n1, n2] = 0
    #                 modified_adj[n2, n1] = 0
    #                 removed_cnt += 1
    #         else:
    #             # For not binary feature, use cosine similarity
    #             C = _cosine_similarity(features[n1], features[n2])
    #             if C < self.threshold:
    #                 modified_adj[n1, n2] = 0
    #                 modified_adj[n2, n1] = 0
    #                 removed_cnt += 1
    #     print('removed %s edges in the original graph' % removed_cnt)
    #     return modified_adj


def _jaccard_similarity(a, b):
    intersection = a.multiply(b).count_nonzero()
    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
    return J


def _cosine_similarity(a, b):
    inner_product = (a * b).sum()
    C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-10)
    return C


def _dropedge_jaccard(A, iA, jA, features, threshold):
    # deprecated: for sparse feature matrix...
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]

            intersection = a.multiply(b).count_nonzero()
            J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a * b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)

            if C < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


@njit
def dropedge_both(A, iA, jA, features, threshold1=2.5, threshold2=0.01):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C1 = np.linalg.norm(features[n1] - features[n2])

            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C2 = inner_product / (np.sqrt(np.square(a).sum() + np.square(b).sum()) + 1e-6)
            if C1 > threshold1 or threshold2 < 0:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


def pairwise_cosine(X):
    pairwise_feat_dot_prods = X @ X.transpose(-2, -1)  # pfdp_ij = <X_i|X_j>
    range_ = torch.arange(pairwise_feat_dot_prods.shape[-1])
    feat_norms = pairwise_feat_dot_prods[..., range_, range_].sqrt()  # fn_i = ||X_i||_2
    feat_norms = torch.where(feat_norms < 1e-8, torch.tensor(1.0, device=X.device), feat_norms)
    return pairwise_feat_dot_prods / feat_norms[..., :, None] / feat_norms[..., None, :]

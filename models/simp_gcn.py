
import os
import math
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

from models.base import ModelBase
from models.gcn import DenseGCNConv

import deeprobust.graph.utils as _utils
from sklearn.metrics.pairwise import cosine_similarity
from torch_sparse import SparseTensor


def noaug_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj_noloop(adj, device):
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = _utils.sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj


class SimpGCN(ModelBase):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):

        super(SimpGCN, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)

        self.gamma = config['gamma']
        self.lambda_ = config['lambda_']
        self.bias_init = config['bias_init']

        nfeat = self.pyg_data.num_features
        nhid = self.config['num_hidden']
        nnodes = self.pyg_data.num_nodes
        nclass = self.pyg_data.num_classes

        self.gc1 = DenseGCNConv(nfeat, nhid, bias=with_bias)
        self.gc2 = DenseGCNConv(nhid, nclass, bias=with_bias)

        self.scores = nn.ParameterList()
        self.D_k = nn.ParameterList()
        self.D_bias = nn.ParameterList()

        self.scores.append(nn.Parameter(torch.FloatTensor(nfeat, 1)))
        self.D_k.append(nn.Parameter(torch.FloatTensor(nfeat, 1)))
        self.D_bias.append(nn.Parameter(torch.FloatTensor(1)))

        for i in range(1):
            self.scores.append(nn.Parameter(torch.FloatTensor(nhid, 1)))
            self.D_k.append(nn.Parameter(torch.FloatTensor(nhid, 1)))
            self.D_bias.append(nn.Parameter(torch.FloatTensor(1)))

        self.bias = nn.ParameterList()
        self.bias.append(nn.Parameter(torch.FloatTensor(1)))
        for i in range(1):
            self.bias.append(nn.Parameter(torch.FloatTensor(1)))

        self.linear = nn.Linear(nhid, 1).to(device)
        self.identity = _utils.sparse_mx_to_torch_sparse_tensor(
            sp.eye(nnodes)).to(self.device)

        self.adj_knn = None
        self.pseudo_labels = None
        self.adj_normed = None

    def get_knn_graph(self, features, k=20):
        features[features != 0] = 1
        sims = cosine_similarity(features)
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        self.sims = sims
        adj_knn = sp.csr_matrix(sims)
        return preprocess_adj_noloop(adj_knn, self.device)
        # if not os.path.exists('saved_knn/'):
        #     os.mkdir('saved_knn')
        # if not os.path.exists(f'saved_knn/knn_graph_{features.shape[0]}_{features.shape[1]}.npz'):
        #     features[features != 0] = 1
        #     sims = cosine_similarity(features)
        #     np.save(f'saved_knn/cosine_sims_{features.shape[0]}_{features.shape[1]}.npy', sims)
        #
        #     sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        #     for i in range(len(sims)):
        #         indices_argsort = np.argsort(sims[i])
        #         sims[i, indices_argsort[: -k]] = 0
        #
        #     adj_knn = sp.csr_matrix(sims)
        #     sp.save_npz(f'saved_knn/knn_graph_{features.shape[0]}_{features.shape[1]}.npz', adj_knn)
        # else:
        #     print(f'loading saved_knn/knn_graph_{features.shape[0]}_{features.shape[1]}.npz...')
        #     adj_knn = sp.load_npz(f'saved_knn/knn_graph_{features.shape[0]}_{features.shape[1]}.npz')
        # return preprocess_adj_noloop(adj_knn, self.device)

    def regression_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = AttrSim(self.x)
            self.pseudo_labels = agent.get_label(sims=self.sims).to(self.device)
            node_pairs = agent.node_pairs
            self.node_pairs = node_pairs

        k = 10000
        node_pairs = self.node_pairs
        if len(self.node_pairs[0]) > k:
            sampled = np.random.choice(len(self.node_pairs[0]), k, replace=False)

            embeddings0 = embeddings[node_pairs[0][sampled]]
            embeddings1 = embeddings[node_pairs[1][sampled]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels[sampled], reduction='mean')
        else:
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')
        # print(loss)
        return loss


    def initialize(self):
        for s in self.scores:
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
        for b in self.bias:
            # fill in b with postive value to make
            # score s closer to 1 at the beginning
            b.data.fill_(self.bias_init)
        for Dk in self.D_k:
            stdv = 1. / math.sqrt(Dk.size(1))
            Dk.data.uniform_(-stdv, stdv)
        for b in self.D_bias:
            b.data.fill_(0)

        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def _forward(self, fea, adj, weight=None):
        if self.adj_knn is None:
            self.adj_knn = self.get_knn_graph(fea.detach().cpu().numpy())

        adj_knn = self.adj_knn
        gamma = self.gamma

        s_i = torch.sigmoid(fea @ self.scores[0] + self.bias[0])

        Dk_i = (fea @ self.D_k[0] + self.D_bias[0])
        x = (s_i * self.gc1(fea, adj) + (1 - s_i) * self.gc1(fea, adj_knn)) + (gamma) * Dk_i * self.gc1(fea, self.identity)

        x = F.dropout(x, self.dropout, training=self.training)
        embedding = x.clone()

        # output, no relu and dropput here.
        s_o = torch.sigmoid(x @ self.scores[-1] + self.bias[-1])
        Dk_o = (x @ self.D_k[-1] + self.D_bias[-1])
        x = (s_o * self.gc2(x, adj) + (1 - s_o) * self.gc2(x, adj_knn)) + (gamma) * Dk_o * self.gc2(x, self.identity)

        # x = F.log_softmax(x, dim=1)

        self.ss = torch.cat((s_i.view(1, -1), s_o.view(1, -1), gamma * Dk_i.view(1, -1), gamma * Dk_o.view(1, -1)),
                            dim=0)
        return x, embedding

    def forward(self, x, mat, weight=None):
        x, _ = self._forward(x, mat, weight)
        return F.log_softmax(x, dim=1)



    def _train_with_early_stopping(self, train_iters, patience, verbose):
        if verbose:
            self.logger.debug(f'=== training {self.__class__.__name__} model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100
        best_weight = None

        train_mask = self.pyg_data.train_mask
        val_mask = self.pyg_data.val_mask

        self.adj_normed = _utils.normalize_adj_tensor(self.edge_index.to_dense())

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, embeddings = self._forward(self.x, self.adj_normed)
            output = F.log_softmax(output, dim=1)
            loss_train = F.nll_loss(output[train_mask], self.labels[train_mask])
            loss_ssl = self.lambda_ * self.regression_loss(embeddings)
            loss_total = loss_train + loss_ssl
            loss_total.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                self.logger.debug(f'Epoch {i:04d}, training loss: {loss_train.item():.5f}')

            self.eval()
            output = self.forward(self.x, self.adj_normed)

            loss_val = F.nll_loss(output[val_mask], self.labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                best_weight = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             self.logger.debug(f'=== early stopping at {i:04d}, loss_val = {best_loss_val:.5f} ===')
        if best_weight is not None:
            self.load_state_dict(best_weight)


    def predict(self, x=None, edge_index=None):
        self.eval()
        if x is None and edge_index is None:
            return self.forward(self.x, self.adj_normed).detach()
        else:
            assert (type(edge_index) is torch.Tensor and edge_index.size[0] == 2) or (type(edge_index) is SparseTensor)
            assert type(x) is torch.Tensor
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            self.adj_normed = _utils.normalize_adj_tensor(edge_index.to_dense())
            return self.forward(x, self.adj_normed).detach()


from itertools import product


class AttrSim:

    def __init__(self, features):
        self.features = features.cpu().numpy()
        self.features[self.features!=0] = 1

    def get_label(self, sims, k=5):
        try:
            indices_sorted = sims.argsort(1)
            idx = np.arange(k, sims.shape[0] - k)
            selected = np.hstack((indices_sorted[:, :k],
                                  indices_sorted[:, -k - 1:]))

            selected_set = set()
            for i in range(len(sims)):
                for pair in product([i], selected[i]):
                    if pair[0] > pair[1]:
                        pair = (pair[1], pair[0])
                    if pair[0] == pair[1]:
                        continue
                    selected_set.add(pair)

        except MemoryError:
            selected_set = set()
            for ii, row in enumerate(sims):
                row = row.argsort()
                idx = np.arange(k, sims.shape[0] - k)
                sampled = np.random.choice(idx, k, replace=False)
                for node in np.hstack((row[:k], row[-k - 1:], row[sampled])):
                    if ii > node:
                        pair = (node, ii)
                    else:
                        pair = (ii, node)
                    selected_set.add(pair)

        sampled = np.array(list(selected_set)).transpose()
        # np.save('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape), sampled)
        print('number of sampled:', len(sampled[0]))
        self.node_pairs = (sampled[0], sampled[1])
        self.sims = sims
        return torch.FloatTensor(sims[self.node_pairs]).reshape(-1,1)

        # features = self.features
        # if not os.path.exists('saved_knn/cosine_sims_{}.npy'.format(features.shape)):
        #     sims = cosine_similarity(features)
        #     np.save('saved_knn/cosine_sims_{}.npy'.format(features.shape), sims)
        # else:
        #     print('loading saved_knn/cosine_sims_{}.npy'.format(features.shape))
        #     sims = np.load('saved_knn/cosine_sims_{}.npy'.format(features.shape))

        # if not os.path.exists('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape)):
        #     try:
        #         indices_sorted = sims.argsort(1)
        #         idx = np.arange(k, sims.shape[0]-k)
        #         selected = np.hstack((indices_sorted[:, :k],
        #             indices_sorted[:, -k-1:]))
        #
        #         selected_set = set()
        #         for i in range(len(sims)):
        #             for pair in product([i], selected[i]):
        #                 if pair[0] > pair[1]:
        #                     pair = (pair[1], pair[0])
        #                 if  pair[0] == pair[1]:
        #                     continue
        #                 selected_set.add(pair)
        #
        #     except MemoryError:
        #         selected_set = set()
        #         for ii, row in enumerate(sims):
        #             row = row.argsort()
        #             idx = np.arange(k, sims.shape[0]-k)
        #             sampled = np.random.choice(idx, k, replace=False)
        #             for node in np.hstack((row[:k], row[-k-1:], row[sampled])):
        #                 if ii > node:
        #                     pair = (node, ii)
        #                 else:
        #                     pair = (ii, node)
        #                 selected_set.add(pair)
        #
        #     sampled = np.array(list(selected_set)).transpose()
        #     np.save('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape), sampled)
        # else:
        #     print('loading saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape))
        #     sampled = np.load('saved_knn/attrsim_sampled_idx_{}.npy'.format(features.shape))
        # print('number of sampled:', len(sampled[0]))
        # self.node_pairs = (sampled[0], sampled[1])
        # self.sims = sims
        # return torch.FloatTensor(sims[self.node_pairs]).reshape(-1,1)

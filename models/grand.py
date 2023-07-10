
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch_sparse import SparseTensor
import scipy.sparse as sp

from models.base import ModelBase

import deeprobust.graph.utils as _utils


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        return x


class Grand(ModelBase):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):

        super(Grand, self).__init__(config, pyg_data, device, logger, with_relu, with_bias, with_bn)

        self.mlp = MLP(
            nfeat=pyg_data.num_features,
            nhid=config['hidden'],
            nclass=pyg_data.num_classes,
            input_droprate=config['input_droprate'],
            hidden_droprate=config['hidden_droprate'],
            use_bn=config['use_bn'])
        self.dropnode_rate = config['dropnode_rate']
        self.sample = config['sample']
        self.order = config['order']
        self.temperate = config['temperate']
        self.lam = config['lam']
        # self.mlp = self.mlp.to(self.device)
        self.A = None


    def initialize(self):
        self.mlp.reset_parameters()

    def _construct_A(self, edge_index):
        adj = edge_index.to_scipy(layout='csr')
        adj = adj + sp.eye(adj.shape[0])
        D1_ = np.array(adj.sum(axis=1)) ** (-0.5)
        D2_ = np.array(adj.sum(axis=0)) ** (-0.5)
        D1_ = sp.diags(D1_[:, 0], format='csr')
        D2_ = sp.diags(D2_[0, :], format='csr')
        A_ = adj.dot(D1_)
        A_ = D2_.dot(A_)

        D1 = np.array(adj.sum(axis=1)) ** (-0.5)
        D2 = np.array(adj.sum(axis=0)) ** (-0.5)
        D1 = sp.diags(D1[:, 0], format='csr')
        D2 = sp.diags(D2[0, :], format='csr')

        A = adj.dot(D1)
        A = D2.dot(A)
        self.A = _utils.sparse_mx_to_torch_sparse_tensor(A)
        self.A = self.A.to(self.device)


    def _propagate(self, feature, mat):
        x = feature
        y = feature
        for i in range(self.order):
            x = torch.spmm(mat, x).detach_()
            y.add_(x)
        return y.div_(self.order+1.0).detach_()

    def _rand_prop(self, features, mat):
        n = features.shape[0]
        drop_rates = torch.FloatTensor(np.ones(n) * self.dropnode_rate)
        if self.training:
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).to(features.device)
            features = masks * features
        else:
            features = features * (1 - drop_rates).unsqueeze(1).to(features.device)

        features = self._propagate(features, mat)
        return features

    def _consis_loss(self, logps):
        ps = [torch.exp(p) for p in logps]
        sum_p = 0
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p / len(ps)
        sharp_p = (
                torch.pow(avg_p, 1./self.temperate) / torch.sum(torch.pow(avg_p, 1./self.temperate), dim=1, keepdim=True)
        ).detach()
        loss = 0.
        for p in ps:
            loss += torch.mean((p - sharp_p).pow(2).sum(1))
        loss = loss / len(ps)
        return self.lam * loss


    def _forward(self, x, mat, weight=None):

        if self.training:
            X = x
            X_list = []
            K = self.sample
            for k in range(K):
                X_list.append(self._rand_prop(X, mat=mat))

            output_list = []
            for k in range(K):
                output_list.append(
                    torch.log_softmax(self.mlp(X_list[k]), dim=-1))
            return output_list
        else:
            return self.mlp(self._rand_prop(x, mat=mat))


    def forward(self, x, mat, weight=None):
        return self._forward(x, mat)


    def _train_with_early_stopping(self, train_iters, patience, verbose):
        if verbose:
            self.logger.debug(f'=== training {self.__class__.__name__} model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100
        best_weight = None

        train_mask = self.pyg_data.train_mask
        val_mask = self.pyg_data.val_mask

        self._construct_A(self.edge_index)

        for i in range(train_iters):


            optimizer.zero_grad()
            self.train()
            output_list = self.forward(self.x, self.A)
            # X = self.x
            # X_list = []
            # K = self.sample
            # for k in range(K):
            #     X_list.append(self._rand_prop(X, mat=self.edge_index))
            #
            # output_list = []
            # for k in range(K):
            #     output_list.append(
            #         torch.log_softmax(self.mlp(X_list[k]), dim=-1))

            loss_train = 0.
            for k in range(self.sample):
                loss_train += F.nll_loss(output_list[k][train_mask], self.labels[train_mask])
            loss_train = loss_train/self.sample
            loss_consis = self._consis_loss(output_list)

            loss_train = loss_train + loss_consis
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                self.logger.debug(f'Epoch {i:04d}, training loss: {loss_train.item():.5f}')

            self.eval()
            output = self.forward(self.x, self.A)
            output = torch.log_softmax(output, dim=-1)

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

    def test(self, test_mask, verbose=True):
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[test_mask], self.labels[test_mask])
        acc_test = _utils.accuracy(output[test_mask], self.labels[test_mask])
        if verbose:
            self.logger.debug(f"Test set results: loss= {loss_test.item():.4f}, accuracy= {float(acc_test):.4f}")
        return float(acc_test)

    def predict(self, x=None, edge_index=None):

        self.eval()
        if x is None and edge_index is None:
            self._construct_A(self.edge_index)
            return F.log_softmax(self.forward(self.x, self.A), dim=-1).detach()
        else:
            assert (type(edge_index) is torch.Tensor and edge_index.size[0] == 2) or (type(edge_index) is SparseTensor)
            assert type(x) is torch.Tensor
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            self._construct_A(edge_index)
            return F.log_softmax(self.forward(x, self.A), dim=-1).detach()




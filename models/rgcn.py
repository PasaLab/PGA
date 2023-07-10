from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor

from deeprobust.graph import utils as _utils
from deeprobust.graph.defense.r_gcn import GGCL_F, GGCL_D, MultivariateNormal



class RGCN(nn.Module):

    def __init__(self,
                 config, pyg_data, device, logger):
        super(RGCN, self).__init__()
        self.config = config
        self.pyg_data = pyg_data
        self.device = device
        self.logger = logger

        self.gamma = self.config['gamma']
        self.beta1 = self.config['beta1']
        self.beta2 = self.config['beta2']
        self.num_hidden = self.config['num_hidden']

        self.dropout = self.config['dropout']
        self.gc1 = GGCL_F(self.pyg_data.num_features, self.num_hidden//2, dropout=self.dropout)
        self.gc2 = GGCL_D(self.num_hidden//2, self.pyg_data.num_classes, dropout=self.dropout)
        self.gaussian = MultivariateNormal(
            torch.zeros(self.pyg_data.num_nodes, self.pyg_data.num_classes),
            torch.diag_embed(torch.ones(self.pyg_data.num_nodes, self.pyg_data.num_classes)),
        )

        self.x = pyg_data.x.to(self.device)
        if hasattr(pyg_data, 'adj_t') and pyg_data.adj_t is not None:
            dense_adj = pyg_data.adj_t.to_dense()
        else:
            dense_adj = to_dense_adj(pyg_data.edge_index).squeeze(0)
        self.adj_norm1 = RGCN._normalize_adj(dense_adj, power=-1 / 2).to(self.device)
        self.adj_norm2 = RGCN._normalize_adj(dense_adj, power=-1).to(self.device)
        self.labels = self.pyg_data.y.to(self.device)


        self.lr = self.config['learning_rate']
        self.weight_decay = self.config['weight_decay']

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, adj_norm1, adj_norm2):
        x = self.x
        miu, sigma = self.gc1(x, adj_norm1, adj_norm2, self.gamma)
        miu, sigma = self.gc2(miu, sigma, adj_norm1, adj_norm2, self.gamma)
        output = miu + self.gaussian.sample().to(self.device) * torch.sqrt(sigma + 1e-8)
        return F.log_softmax(output, dim=1)


    def fit(self, pyg_data, train_iters=None, patience=None, initialize=True, verbose=False):

        if initialize:
            self.initialize()

        if train_iters is None:
            train_iters = self.config['num_epochs']
        if patience is None:
            patience = self.config['patience']

        self.logger.debug(f"total training epochs: {train_iters}, patience: {patience}")

        self.x = pyg_data.x.to(self.device)

        if hasattr(pyg_data, 'adj_t') and pyg_data.adj_t is not None:
            dense_adj = pyg_data.adj_t.to_dense()
        else:
            dense_adj = to_dense_adj(pyg_data.edge_index).squeeze(0)
        self.adj_norm1 = RGCN._normalize_adj(dense_adj, power=-1 / 2).to(self.device)
        self.adj_norm2 = RGCN._normalize_adj(dense_adj, power=-1).to(self.device)


        if patience < train_iters:
            self._train_with_early_stopping(train_iters, patience, verbose)
        else:
            self._train_with_val(train_iters, verbose)

    def _train_with_val(self, train_iters, verbose):
        if verbose:
            self.logger.debug('=== training gcn model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        best_weight = None

        train_mask = self.pyg_data.train_mask
        val_mask = self.pyg_data.val_mask

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(adj_norm1=self.adj_norm1, adj_norm2=self.adj_norm2)
            loss_train = F.nll_loss(output[train_mask], self.labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                self.logger.debug(f'Epoch {i:04d}, training loss: {loss_train.item():.5f}')

            self.eval()
            output = self.forward(adj_norm1=self.adj_norm1, adj_norm2=self.adj_norm2)
            loss_val = F.nll_loss(output[val_mask], self.labels[val_mask])
            acc_val = _utils.accuracy(output[val_mask], self.labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                best_weight = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                best_weight = deepcopy(self.state_dict())

        if verbose:
            self.logger.debug('=== picking the best model according to the performance on validation ===')
        if best_weight is not None:
            self.load_state_dict(best_weight)

    def _train_with_early_stopping(self, train_iters, patience, verbose):
        if verbose:
            self.logger.debug('=== training gcn model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100
        best_weight = None

        train_mask = self.pyg_data.train_mask
        val_mask = self.pyg_data.val_mask

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(adj_norm1=self.adj_norm1, adj_norm2=self.adj_norm2)
            loss_train = F.nll_loss(output[train_mask], self.labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                self.logger.debug(f'Epoch {i:04d}, training loss: {loss_train.item():.5f}')

            self.eval()
            output = self.forward(adj_norm1=self.adj_norm1, adj_norm2=self.adj_norm2)

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
            return self.forward(adj_norm1=self.adj_norm1, adj_norm2=self.adj_norm2).detach()
        else:
            assert type(edge_index) is torch.Tensor or type(edge_index is SparseTensor)
            assert type(x) is torch.Tensor
            # x = x.to(self.device)
            if edge_index.size(0) == 2:  # [2, num_edges]
                dense_adj = to_dense_adj(edge_index).squeeze(0).to(self.device)
            elif type(edge_index) is SparseTensor:
                dense_adj = edge_index.to_dense().to(self.device)
            elif edge_index.size(0) == edge_index.size(1):  # [n, n]
                dense_adj = edge_index.to(self.device)

            adj_norm1 = RGCN._normalize_adj(dense_adj, power=-1 / 2)
            adj_norm2 = RGCN._normalize_adj(dense_adj, power=-1)
            return self.forward(adj_norm1=adj_norm1, adj_norm2=adj_norm2).detach()


    def _loss(self, input, labels):
        loss = F.nll_loss(input, labels)
        miu1 = self.gc1.miu
        sigma1 = self.gc1.sigma
        kl_loss = 0.5 * (miu1.pow(2) + sigma1 - torch.log(1e-8 + sigma1)).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + \
                torch.norm(self.gc1.weight_sigma, 2).pow(2)
        return loss + self.beta1 * kl_loss + self.beta2 * norm2

    @staticmethod
    def _normalize_adj(adj, power=-1/2):
        device = adj.device
        A = adj + torch.eye(len(adj)).to(device)
        D_power = (A.sum(1)).pow(power)
        D_power[torch.isinf(D_power)] = 0.
        D_power = torch.diag(D_power)
        return D_power @ A @ D_power






from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_sparse import SparseTensor
from deeprobust.graph import utils as _utils

from abc import abstractmethod


class ModelBase(nn.Module):

    def __init__(self,
                 config, pyg_data, device, logger,
                 with_relu=True, with_bias=True, with_bn=False):

        super(ModelBase, self).__init__()
        self.config = config
        self.pyg_data = pyg_data
        self.device = device
        self.logger = logger

        self.with_relu = with_relu
        self.with_bias = with_bias
        self.with_bn = with_bn

        self.x = pyg_data.x.to(self.device)
        self.edge_index = pyg_data.adj_t.to(self.device)
        self.labels = self.pyg_data.y.to(self.device)

        if 'learning_rate' in self.config:
            self.lr = self.config['learning_rate']
        if 'weight_decay' in self.config:
            self.weight_decay = self.config['weight_decay']
        if 'dropout' in self.config:
            self.dropout = self.config['dropout']


    def initialize(self):
        pass

    @abstractmethod
    def _forward(self, x, mat, weight=None):
        pass

    @abstractmethod
    def forward(self, x, mat, weight=None):
        pass


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
            self.edge_index = pyg_data.adj_t.to(self.device)
        else:
            self.edge_index = adj_t.to(self.device)


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
            output = self.forward(self.x, self.edge_index)
            loss_train = F.nll_loss(output[train_mask], self.labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                self.logger.debug(f'Epoch {i:04d}, training loss: {loss_train.item():.5f}')

            self.eval()
            output = self.forward(self.x, self.edge_index)
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
            self.logger.debug(f'=== training {self.__class__.__name__} model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100
        best_weight = None

        train_mask = self.pyg_data.train_mask
        val_mask = self.pyg_data.val_mask

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.x, self.edge_index)
            loss_train = F.nll_loss(output[train_mask], self.labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                self.logger.debug(f'Epoch {i:04d}, training loss: {loss_train.item():.5f}')

            self.eval()
            output = self.forward(self.x, self.edge_index)

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
            return self.forward(self.x, self.edge_index).detach()
        else:
            assert (type(edge_index) is torch.Tensor and edge_index.size(0) == 2) or (type(edge_index) is SparseTensor)
            assert type(x) is torch.Tensor
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            return self.forward(x, edge_index).detach()

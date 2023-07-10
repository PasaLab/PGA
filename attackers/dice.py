
import torch
import numpy as np
import scipy.sparse as sp
from attackers.base import AttackABC
from torch_sparse import SparseTensor


class DICE(AttackABC):

    def __init__(self,
                 attack_config, pyg_data,
                 model, device, logger, **kwargs):
        super(DICE, self).__init__(attack_config, pyg_data, model, device, logger)
        self.add_ratio = attack_config['add_ratio']

    def _attack(self, n_perturbations):
        ori_adj = self.pyg_data.adj_t.to_scipy(layout='csr')
        labels = self.pyg_data.y
        modified_adj = ori_adj.tolil()
        # remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = int(n_perturbations * (1 - self.add_ratio))

        # nonzero = set(zip(*ori_adj.nonzero()))
        indices = sp.triu(modified_adj).nonzero()
        possible_indices = [x for x in zip(indices[0], indices[1])
                            if labels[x[0]] == labels[x[1]]]

        remove_indices = np.random.permutation(possible_indices)[: n_remove]
        modified_adj[remove_indices[:, 0], remove_indices[:, 1]] = 0
        modified_adj[remove_indices[:, 1], remove_indices[:, 0]] = 0

        n_insert = n_perturbations - n_remove

        # sample edges to add
        added_edges = 0
        while added_edges < n_insert:
            n_remaining = n_insert - added_edges

            # sample random pairs
            candidate_edges = np.array([np.random.choice(ori_adj.shape[0], n_remaining),
                                        np.random.choice(ori_adj.shape[0], n_remaining)]).T

            # filter out existing edges, and pairs with the different labels
            candidate_edges = set([(u, v) for u, v in candidate_edges if labels[u] != labels[v]
                                   and modified_adj[u, v] == 0 and modified_adj[v, u] == 0])
            candidate_edges = np.array(list(candidate_edges))

            # if none is found, try again
            if len(candidate_edges) == 0:
                continue

            # add all found edges to your modified adjacency matrix
            modified_adj[candidate_edges[:, 0], candidate_edges[:, 1]] = 1
            modified_adj[candidate_edges[:, 1], candidate_edges[:, 0]] = 1
            added_edges += candidate_edges.shape[0]

        self.adj_adversary = SparseTensor.from_dense(torch.tensor(modified_adj.todense()))

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from deeprobust.graph import utils
from attackers.base import AttackABC
from torch_sparse import SparseTensor


def filter_potential_singletons(modified_adj):

    degrees = modified_adj.sum(0)
    degree_one = (degrees == 1)
    resh = degree_one.repeat(modified_adj.shape[0], 1).float()
    l_and = resh * modified_adj
    l_and = l_and + l_and.t()
    flat_mask = 1 - l_and
    return flat_mask


def get_adj_score(adj_grad, modified_adj):
    adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
    # Make sure that the minimum entry is 0.
    adj_meta_grad -= adj_meta_grad.min()
    # Filter self-loops
    adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
    # # Set entries to 0 that could lead to singleton nodes.
    singleton_mask = filter_potential_singletons(modified_adj)
    adj_meta_grad = adj_meta_grad * singleton_mask

    return adj_meta_grad


class Greedy(AttackABC):
    def __init__(self,
                 attack_config, pyg_data,
                 model, device, logger, **kwargs):
        super(Greedy, self).__init__(attack_config, pyg_data, model, device, logger)
        nnodes = self.pyg_data.num_nodes
        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes)).to(self.device)
        self.adj_changes.data.fill_(0)

    def initialize(self):
        self.adj_changes.data.fill_(0)

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj

    def get_meta_grad(self, features, adj_norm, idx_train, idx, labels_self_training):
        self.surrogate.eval()
        output = self.surrogate(features, adj_norm)
        output = F.log_softmax(output, dim=1)
        train_loss = F.nll_loss(output[idx_train], labels_self_training[idx_train])
        test_loss = F.nll_loss(output[idx], labels_self_training[idx])

        alpha = 0.0
        attack_loss = train_loss * alpha + test_loss * (1-alpha)

        adj_grad = torch.autograd.grad(attack_loss, self.adj_changes)[0]
        return adj_grad


    def _attack(self, n_perturbations):

        victim = self.attacked_model
        ori_features = self.pyg_data.x
        ori_adj = self.pyg_data.adj_t.to_dense()
        labels = self.pyg_data.y
        idx_train = self.pyg_data.train_mask
        idx_test = self.pyg_data.test_mask

        mod_adj = self.private_attack(
            victim, ori_features, ori_adj,
            labels, idx_train, idx_test, n_perturbations
        )

        self.adj_adversary = SparseTensor.from_dense(mod_adj).coalesce().detach()


    def private_attack(self, victim, ori_features, ori_adj, labels, idx_train, idx_test, n_perturbations, initialize=True):

        self.surrogate = victim
        victim.eval()
        if initialize:
            self.initialize()

        labels_self_training = labels

        for _ in tqdm(range(n_perturbations), desc="Perturbing graph"):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            adj_grad = self.get_meta_grad(ori_features, adj_norm, idx_train, idx_test, labels_self_training)
            adj_meta_score = get_adj_score(adj_grad, modified_adj.detach())

            adj_meta_argmax = torch.argmax(adj_meta_score).detach()
            row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

        return self.get_modified_adj(ori_adj).detach().cpu()


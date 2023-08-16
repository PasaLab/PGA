import numpy as np
import torch
import torch.nn.functional as F

from attackers.base import AttackABC
from tqdm import tqdm
from common.utils import *

from torch_geometric.utils import k_hop_subgraph, coalesce as pyg_coalesce
from torch_sparse import coalesce



def select_attacking_targets(logits, labels, idx_attack, adj_t, policy, pre_ratio, select_ratio):

    correct_mask = logits.argmax(1).cpu() == labels.cpu()
    index_attack = torch.nonzero(correct_mask & idx_attack.cpu()).flatten().detach().cpu()

    """
    'degree_centrality'
    'pagerank'
    'clustering_coefficient'
    'eigenvector_centrality'
    """
    margins = classification_margin(logits, labels)[index_attack]
    degrees = calculate_degree(adj_t)[index_attack]
    # degree_cty = stat_data['degree_centrality'][index_attack]
    # pagerank = stat_data['pagerank'][index_attack]
    # cluster_coeff = stat_data['clustering_coefficient'][index_attack]
    # eigen_cty = stat_data['eigenvector_centrality'][index_attack]
    # surr_logit = stat_data['logits']
    # if type(surr_logit) is list:
    #     surr_logit = surr_logit[0]
    # surr_logit = surr_logit[index_attack]
    # entropy = -(surr_logit * surr_logit.log()).sum(1)

    n_nodes = index_attack.size(0)

    sorted_margins, _ = margins.sort()
    margin_threshold = sorted_margins[int(pre_ratio * n_nodes)]
    pre_mask = torch.where(margins > margin_threshold, True, False)

    margins = margins[pre_mask]
    degrees = degrees[pre_mask]
    # degree_cty = degree_cty[pre_mask]
    # pagerank = pagerank[pre_mask]
    # cluster_coeff = cluster_coeff[pre_mask]
    # eigen_cty = eigen_cty[pre_mask]
    # entropy = entropy[pre_mask]

    index_attack = index_attack[pre_mask]

    n_nodes = index_attack.size(0)
    selected_idx = torch.arange(0, n_nodes)
    n_selected = int(select_ratio * n_nodes)

    indicators = [
        # entropy,
        degrees,
        # degree_cty,
        # pagerank,
        # eigen_cty,
        # cluster_coeff,
        margins,
    ]



    for item in indicators:
        _, sorted_index = item.sort()
        item_selected = sorted_index[:n_selected]
        selected_idx = torch.tensor(np.intersect1d(item_selected, selected_idx), dtype=torch.long)

    # selected_idx = torch.randperm(n_nodes)[:n_selected]


    return index_attack[selected_idx]



class PGA(AttackABC):

    def __init__(self,
                 attack_config, pyg_data,
                 model, device, logger, **kwargs):
        super(PGA, self).__init__(attack_config, pyg_data, model, device, logger)

        self.greedy_steps = self.attack_config['greedy_steps']
        self.pre_ratio = self.attack_config['pre_ratio']
        self.select_ratio = self.attack_config['select_ratio']
        self.influ_ratio = self.attack_config['influ_ratio']
        self.select_policy = self.attack_config['select_policy']
        self.attacked_model.with_relu = False

        self.loss_type = self.attack_config['loss_type']
        self.x = self.pyg_data.x
        self.adj = self.pyg_data.adj_t
        self.labels = self.pyg_data.y
        self.n = self.pyg_data.num_nodes

        self.edge_index = deepcopy(self.pyg_data.edge_index)

        self.attacking_targets = None
        self.anchor_labels = None

        self.idx_attack = self.pyg_data.test_mask

        # self.graph_statistics = torch.load(f"../analysis/{kwargs['dataset_name']}-gcn.pth")


    def _attack(self, n_perturbations, **kwargs):

        logits = self.attacked_model.predict().cpu()

        self.attacking_targets = select_attacking_targets(
            logits, self.labels, self.idx_attack, self.adj,
            self.select_policy, self.pre_ratio, self.select_ratio)

        # torch.save(
        #     obj=self.attacking_targets.detach().cpu(),
        #     f=f"{kwargs['dataset']}_selected_targets_"
        #       f"{self.attack_config['pre_ratio']}_"
        #       f"{self.attack_config['select_ratio']}_"
        #       f"{self.attack_config['influ_ratio']}.pt",
        # )

        self.anchor_labels = kth_best_wrong_label(
            logits[self.attacking_targets],
            self.labels[self.attacking_targets]
        ).to(self.device)

        (
            edge_index, non_edge_index, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight
        ) = self._construct_modified_edges()

        steps = [self.greedy_steps] * (n_perturbations // self.greedy_steps)
        for i in range(n_perturbations % self.greedy_steps):
            steps[i] += 1

        added_set = set()
        removed_set = set()

        for step_size in tqdm(steps, desc='Attacking...'):
            weights = torch.cat([edge_weight, edge_weight,
                                 non_edge_weight, non_edge_weight,
                                 self_loop_weight], dim=-1)
            edge_grad, non_edge_grad = self._compute_gradient(edges_all, weights, (edge_weight, non_edge_weight))
            i = 0
            while i < step_size:

                max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
                max_non_edge_grad, max_non_edge_idx = torch.max(non_edge_grad, dim=0)
                if max_edge_grad.max().item() == 0.0 and max_non_edge_grad.max().item() == 0:
                    break

                if max_edge_grad > max_non_edge_grad:  # 删除边
                    edge_grad.data[max_edge_idx] = 0.0
                    best_edge = edge_index[:, max_edge_idx]
                    best_edge_reverse = best_edge[[1, 0]]
                    edge_weight.data[max_edge_idx] = 0.0
                    mask = ~((self.edge_index == best_edge.reshape(-1, 1)).all(0) | (self.edge_index == best_edge_reverse.reshape(-1, 1)).all(0))
                    self.edge_index = self.edge_index[:, mask]

                    u, v = best_edge.tolist()
                    if (u, v) in removed_set or (v, u) in removed_set:
                        continue
                    removed_set.add((u, v))

                else:  # 加边
                    non_edge_grad.data[max_non_edge_idx] = 0.0
                    best_edge = non_edge_index[:, max_non_edge_idx]
                    best_edge_reverse = best_edge[[1, 0]]
                    non_edge_weight.data[max_non_edge_idx] = 1.0
                    self.edge_index = pyg_coalesce(
                        torch.cat([
                            self.edge_index, best_edge.reshape(-1, 1), best_edge_reverse.reshape(-1, 1),
                        ], dim=1)
                    )
                    u, v = best_edge.tolist()
                    if (u, v) in added_set or (v, u) in added_set:
                        continue
                    added_set.add((u, v))

                i += 1

        self.adj_adversary = SparseTensor.from_edge_index(edge_index=self.edge_index, sparse_sizes=(self.n, self.n)).coalesce().detach()


    def _construct_modified_edges(self):

        sub_nodes, sub_edges, *_ = k_hop_subgraph(self.attacking_targets, 2, self.edge_index)
        sub_edges = sub_edges[:, sub_edges[0] < sub_edges[1]]
        neighbors = torch.as_tensor(self.adj[self.attacking_targets].to_scipy(layout='csr').nonzero()[1], dtype=torch.long)

        anchors = []
        type_set = set(self.anchor_labels.tolist())
        for node_type in type_set:
            anchors.extend(torch.where(self.labels == node_type)[0].tolist())
        anchors = torch.as_tensor(anchors, dtype=torch.long)
        anchors = np.setdiff1d(anchors, neighbors)

        (
            edge_index, non_edge_index, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight
        ) = self._gen_edges(anchors, sub_nodes, sub_edges)

        weights = torch.cat([edge_weight, edge_weight,
                             non_edge_weight, non_edge_weight,
                             self_loop_weight], dim=-1)

        k = int(self.influ_ratio * len(anchors))

        add_gradient = self._compute_gradient(edges_all, weights, (non_edge_weight, ))[0]
        _, topk_nodes = torch.topk(add_gradient, k=k, sorted=False)

        anchors = np.unique(non_edge_index[1][topk_nodes.cpu()].cpu().numpy())

        del edge_index, non_edge_index, self_loop, edges_all
        del edge_weight, non_edge_weight, self_loop_weight
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        (
            edge_index, non_edge_index, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight
        ) = self._gen_edges(anchors, sub_nodes, sub_edges)

        return (
            edge_index, non_edge_index, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight
        )

    def get_modified_adj(self):
        return SparseTensor.from_edge_index(edge_index=self.edge_index, sparse_sizes=(self.n, self.n)).coalesce().detach()

    def _targets_subgraph(self):
        sub_nodes, sub_edges, *_ = k_hop_subgraph(self.attacking_targets, 2, self.edge_index)
        sub_edges = sub_edges[:, sub_edges[0] < sub_edges[1]]
        return sub_nodes.cpu(), sub_edges.cpu()

    def _targets_neighbors(self):
        return torch.as_tensor(self.adj[self.attacking_targets].to_scipy(layout='csr').nonzero()[1], dtype=torch.long)

    def _select_anchors(self, neighbors):
        anchors = []
        type_set = set(self.anchor_labels.tolist())
        for node_type in type_set:
            anchors.extend(torch.where(self.labels == node_type)[0].tolist())
        anchors = torch.as_tensor(anchors, dtype=torch.long)
        anchors = np.setdiff1d(anchors, neighbors)
        return torch.as_tensor(anchors, dtype=torch.long)


    def _gen_edges(self, anchors, sub_nodes, sub_edges):
        row = np.repeat(self.attacking_targets, len(anchors))
        col = np.tile(anchors, self.attacking_targets.size(0))

        non_edges = np.row_stack([row, col])
        unique_nodes = np.union1d(sub_nodes.tolist(), anchors.tolist())

        non_edges = torch.as_tensor(non_edges, device=self.device)
        unique_nodes = torch.as_tensor(unique_nodes, dtype=torch.long, device=self.device)

        self_loop = unique_nodes.repeat((2, 1))
        edges_all = torch.cat([
            sub_edges, sub_edges[[1, 0]],
            non_edges, non_edges[[1, 0]],
            self_loop,
        ], dim=1)

        edge_weight = torch.ones(sub_edges.size(1), device=self.device).requires_grad_(True)
        non_edge_weight = torch.zeros(non_edges.size(1), device=self.device).requires_grad_(True)
        self_loop_weight = torch.ones(self_loop.size(1), device=self.device)

        return (
            sub_edges, non_edges, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight
        )

    def _compute_gradient(self, edges_all, weights, inputs, target_weight=None):
        inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
        if target_weight is not None:
            inputs = inputs + (target_weight, )
        logit = self._calc_logit(edges_all, weights)
        loss = self._calc_loss(logit, target_weight=target_weight)
        gradients = torch.autograd.grad(loss, inputs, create_graph=False)
        return gradients

    def _calc_logit(self, edges_all, weights_all):
        edge_index = torch.cat([self.edge_index, edges_all], dim=-1)
        edge_weight = torch.cat([torch.ones(self.edge_index.size(1), dtype=torch.float32, device=self.device), weights_all], dim=-1)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
        mod_adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, (self.n, self.n))
        logit = self.attacked_model.forward(self.x, mod_adj_t, edge_weight)
        return logit

    def _calc_loss(self, logit, eps=5.0, target_weight=None):

        loss = None
        targets = self.attacking_targets
        labels = self.labels[targets]
        if self.loss_type == 'CW':
            logit = F.log_softmax(logit / eps, dim=1)
            loss = F.nll_loss(logit[targets], labels) - \
                   F.nll_loss(logit[targets], self.anchor_labels)
        elif self.loss_type == 'CE':
            loss = F.cross_entropy(logit[targets], labels)
        elif self.loss_type == 'MCE':
            target_preds = logit[targets].argmax(1)
            not_flipped = target_preds == labels
            loss = F.cross_entropy(logit[targets][not_flipped], labels[not_flipped])
        elif self.loss_type == 'MCW':
            target_preds = logit[targets].argmax(1)
            logit = F.log_softmax(logit, dim=1)
            not_flipped = target_preds == labels
            loss = F.nll_loss(logit[targets][not_flipped], labels[not_flipped]) - \
                   F.nll_loss(logit[targets][not_flipped], self.anchor_labels[not_flipped])
        elif self.loss_type == 'tanhMargin':
            target_preds = logit[targets].argmax(1)
            not_flipped = target_preds == labels
            logit_safe = logit[targets][not_flipped]
            margin_safe = (
                logit_safe[np.arange(logit_safe.size(0)), labels[not_flipped]]
                - logit_safe[np.arange(logit_safe.size(0)), self.anchor_labels[not_flipped]]
            )
            loss = torch.tanh(-margin_safe).mean()
        elif self.loss_type.startswith('tanhMarginMCE-'):
            alpha = float(self.loss_type.split('-')[-1])
            target_preds = logit[targets].argmax(1)
            not_flipped = target_preds == labels
            logit_safe = logit[targets][not_flipped]
            margin_safe = (
                    logit_safe[np.arange(logit_safe.size(0)), labels[not_flipped]]
                    - logit_safe[np.arange(logit_safe.size(0)), self.anchor_labels[not_flipped]]
            )
            loss = alpha * torch.tanh(-margin_safe) + (1 - alpha) * F.cross_entropy(logit[targets][not_flipped], labels[not_flipped], reduction='none')
            if target_weight is not None:
                loss = target_weight[not_flipped] * loss
            loss = loss.mean()

        assert loss is not None, f"Not support loss type: {self.loss_type}"

        return loss



class PoisonPGA(PGA):

    def __init__(self,
                 attack_config, pyg_data,
                 model, device, logger, **kwargs):
        super(PoisonPGA, self).__init__(attack_config, pyg_data, model, device, logger, **kwargs)

        self.surrogate = type(self.attacked_model)(
            self.attacked_model.config, pyg_data, device, logger,
            with_relu=False)
        self.surrogate = self.surrogate.to(self.device)

    def inner_train(self, adj_t):
        self.surrogate.fit(self.pyg_data, adj_t=adj_t, train_iters=100, verbose=False)

    def _calc_logit(self, edges_all, weights_all):
        edge_index = torch.cat([self.edge_index, edges_all], dim=-1)
        edge_weight = torch.cat([torch.ones(self.edge_index.size(1), dtype=torch.float32, device=self.device), weights_all], dim=-1)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')
        mod_adj_t = SparseTensor.from_edge_index(edge_index, edge_weight, (self.n, self.n))
        logit = self.surrogate.forward(self.x, mod_adj_t, edge_weight)
        return logit

    def _attack(self, n_perturbations, **kwargs):

        logits = self.attacked_model.predict().cpu()
        self.attacking_targets = select_attacking_targets(
            logits, self.labels, self.idx_attack, self.adj,
            self.select_policy, self.pre_ratio, self.select_ratio, self.graph_statistics)
        self.anchor_labels = kth_best_wrong_label(logits[self.attacking_targets],
                                                  self.labels[self.attacking_targets]).to(self.device)
        (
            edge_index, non_edge_index, self_loop, edges_all,
            edge_weight, non_edge_weight, self_loop_weight
        ) = self._construct_modified_edges()

        added_set = set()
        removed_set = set()

        for _ in tqdm(range(n_perturbations), desc="Perturbing Graph"):
            modified_adj = self.get_modified_adj()
            self.inner_train(modified_adj)
            weights = torch.cat([edge_weight, edge_weight,
                                 non_edge_weight, non_edge_weight,
                                 self_loop_weight], dim=-1)
            edge_grad, non_edge_grad = self._compute_gradient(edges_all, weights, (edge_weight, non_edge_weight))

            while True:
                max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
                max_non_edge_grad, max_non_edge_idx = torch.max(non_edge_grad, dim=0)
                if max_edge_grad.max().item() == 0.0 and max_non_edge_grad.max().item() == 0:
                    break

                if max_edge_grad > max_non_edge_grad:  # 删除边
                    edge_grad.data[max_edge_idx] = 0.0
                    best_edge = edge_index[:, max_edge_idx]
                    best_edge_reverse = best_edge[[1, 0]]
                    edge_weight.data[max_edge_idx] = 0.0
                    mask = ~((self.edge_index == best_edge.reshape(-1, 1)).all(0) | (
                                self.edge_index == best_edge_reverse.reshape(-1, 1)).all(0))
                    self.edge_index = self.edge_index[:, mask]

                    u, v = best_edge.tolist()
                    if (u, v) in removed_set or (v, u) in removed_set:
                        continue
                    removed_set.add((u, v))
                    break

                else:  # 加边
                    non_edge_grad.data[max_non_edge_idx] = 0.0
                    best_edge = non_edge_index[:, max_non_edge_idx]
                    best_edge_reverse = best_edge[[1, 0]]
                    non_edge_weight.data[max_non_edge_idx] = 1.0
                    self.edge_index = pyg_coalesce(
                        torch.cat([
                            self.edge_index, best_edge.reshape(-1, 1), best_edge_reverse.reshape(-1, 1),
                        ], dim=1)
                    )
                    u, v = best_edge.tolist()
                    if (u, v) in added_set or (v, u) in added_set:
                        continue
                    added_set.add((u, v))
                    break

        self.adj_adversary = SparseTensor.from_edge_index(edge_index=self.edge_index, sparse_sizes=(self.n, self.n)).coalesce().detach()

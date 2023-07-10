import random
import numpy as np
import torch
import logging
import os
from copy import deepcopy
import json
import networkx as nx

import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid, CitationFull, KarateClub
from torch_geometric.utils import convert
import torch_geometric.transforms as T

from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph import utils as _utils


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        device = f'cuda:{gpu_id}'
    else:
        device = 'cpu'
    return device


def freeze_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_perturbed_adj(dataset, attack, ptb_rate, path):
    assert dataset in ['cora', 'citeseer', 'pubmed', 'cora_ml']
    # assert attack in ['prbcd', 'greedy-rbcd', 'pga', 'apga', 'mettack', 'minmax', 'pgdattack', 'graD', 'greedy']
    assert os.path.exists(path)
    filename = attack + '-' + dataset + '-' + f'{ptb_rate}' + '.pth'
    filename = os.path.join(path, filename)
    return torch.load(filename)


def load_pretrained_model(dataset, model, path):
    assert dataset in ['cora', 'citeseer', 'pubmed', 'cora_ml']
    # assert model in ['gcn', 'gat', 'sgc', 'rgcn', 'graph-sage', 'median-gcn', 'gcnsvd', 'gcn-jaccard', 'grand', 'gnn-guard', 'simp-gcn', 'dense-gcn']
    assert os.path.exists(path)
    filename = model + '-' + dataset + '.pth'
    filename = os.path.join(path, filename)
    assert os.path.exists(filename)
    return torch.load(filename)


def gen_pseudo_label(model, labels, mask):
    device = labels.device
    model.eval()
    logit = model.predict()
    pred = logit.argmax(dim=1)
    labels[mask] = pred[mask].to(device)
    return labels


def evaluate_attack_performance(victim, x, mod_adj, labels, mask):
    victim.eval()
    device = victim.device
    logit_detach = victim.predict(x.to(device), mod_adj.to(device))
    return _utils.accuracy(logit_detach[mask], labels[mask])


def load_data(name='cora', path='/root/cmy/datasets/', seed=15, x_normalize=True):
    assert name in ['cora', 'citeseer', 'pubmed', 'cora_ml']
    # x_normalize = False if name == 'polblogs' else True
    # freeze_seed(seed)
    if name in ['cora_ml']:
        dataset = Dataset(root=path, name=name, setting='gcn')
        dataset = Dpr2Pyg(dataset, transform=T.ToSparseTensor(remove_edge_index=False))
    elif name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name, transform=T.ToSparseTensor(remove_edge_index=False))
    elif name == 'karate_club':
        dataset = KarateClub(transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if name == 'karate_club':
        data.test_mask = ~(data.train_mask)
        data.val_mask = torch.zeros_like(data.test_mask, dtype=torch.bool)
    if x_normalize:
        data.x = normalize_feature_tensor(data.x)
    return data


def calc_modified_rate(clean_adj_t, mod_adj_t, n_edges):
    sp_clean = clean_adj_t.to_scipy(layout='coo')
    sp_modified = mod_adj_t.to_scipy(layout='coo')
    diff = sp_clean - sp_modified
    n_diff = diff.getnnz()
    return float(n_diff) / n_edges


def check_undirected(mod_adj_t):
    adj = mod_adj_t.to_dense()
    adj_T = mod_adj_t.t().to_dense()
    is_symmetric = bool(torch.all(adj == adj_T))
    assert is_symmetric is True


def normalize_feature_tensor(x):
    x = _utils.to_scipy(x)
    x = _utils.normalize_feature(x)
    x = torch.FloatTensor(np.array(x.todense()))
    return x


def classification_margin(logits: torch.Tensor, labels: torch.Tensor):
    probs = torch.exp(logits).cpu()
    label = labels.cpu()
    fill_zeros = torch.zeros_like(probs)
    true_probs = probs.gather(1, label.view(-1, 1)).flatten()
    probs.scatter_(1, label.view(-1, 1), fill_zeros)
    best_wrong_probs = probs.max(dim=1)[0]
    return true_probs - best_wrong_probs

def calculate_entropy(logits: torch.Tensor):
    return -(logits * logits.log()).sum(1)


def calculate_degree(adj_t):
    assert type(adj_t) is torch.Tensor or type(adj_t) is SparseTensor
    if type(adj_t) is SparseTensor:
        return torch_sparse.sum(adj_t, dim=1).to_dense().cpu()
    # TODO: edge_index和dense_adj计算度


def kth_best_wrong_label(logits, labels, k=1):
    logits = logits.exp()
    prev = deepcopy(labels).detach().cpu()
    best_wrong_label = None
    while k > 0:
        fill_zeros = torch.zeros_like(logits)
        logits.scatter_(1, prev.view(-1, 1), fill_zeros)
        best_wrong_label = logits.argmax(1)
        prev = best_wrong_label
        k = k - 1
    return best_wrong_label


def get_logger(filename, level=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[level])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


###################### analysis
"""
计算graph中各个结点的统计量：
    degree
    degree_centrality
    pagerank
    clustering_coefficient
    eigenvector_centrality
    # betweenness_centrality 计算复杂度过大, 所以舍弃
"""
def calc_statistic_data(pyg_data, logits):

    G = convert.to_networkx(pyg_data)

    degrees = calculate_degree(pyg_data.adj_t)
    degree_centrality = torch.as_tensor(list(nx.degree_centrality(G).values()))
    pagerank = torch.as_tensor(list(nx.pagerank(G).values()))
    clustering_coefficient = torch.as_tensor(list(nx.clustering(G).values()))
    eigenvector_centrality = torch.as_tensor(list(nx.eigenvector_centrality(G).values()))
    cls_margin = classification_margin(logits, logits.argmax(1))

    return degrees, degree_centrality, pagerank, clustering_coefficient, eigenvector_centrality, cls_margin



def save_result_to_json(attack, dataset, victim, ptb_rate, attacked_acc, attack_type):
    assert attack_type in ['evasion', 'poisoning']
    # Data to be written
    dictionary = {
        "attack": attack,
        "dataset": dataset,
        "victim": victim,
        "ptb_rate": ptb_rate,
        "attacked_acc": attacked_acc,
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    save_name = f"{attack}-{dataset}-{victim}-{ptb_rate}.json"
    # Writing to sample.json
    path = f"/root/cmy/project_1/results/{attack_type}"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f"{path}/{save_name}", "w") as outfile:
        outfile.write(json_object)
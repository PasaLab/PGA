import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from common.utils import freeze_seed

"""
obj = {
    'degrees': degrees,
    'degree_centrality': degree_centrality,
    'pagerank': pagerank,
    'clustering_coefficient': clustering_coefficient,
    # 'betweenness_centrality': betweenness_centrality,
    'eigenvector_centrality': eigenvector_centrality,
    'cls_margin': cls_margin,

    'stable_mask': stable_mask,
    'fragile_mask': fragile_mask,
    'lucky_mask': lucky_mask,
    'foolish_mask': foolish_mask,
    'test_mask': test_mask,

    'logits': logits,
    'neighbor_mean': neighbor_mean,
    'neighbor_var': neighbor_var,
    'neighbor_skewness': neighbor_skewness,
}
"""


def calculate(name):
    statistics = torch.load(f"./{name}-greedy-rbcd-gcn.pth")
    test_mask = statistics['test_mask']
    stable_mask = statistics['stable_mask']
    fragile_mask = statistics['fragile_mask']
    mask = test_mask & (stable_mask | fragile_mask)
    stable_node_mask = mask & stable_mask
    # fragile_node_mask = mask & fragile_mask

    Y = torch.zeros(mask.size(0), dtype=torch.long)
    Y[stable_node_mask] = 1
    Y = Y[mask].numpy()

    freeze_seed(42)

    print(f"Dataset= {name}")

    for key in statistics.keys():
        if not key.endswith('mask') and not key.startswith('neighbor') and key != 'logits':
            kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
            X = statistics[key][mask].numpy().reshape(-1, 1)
            X_discrete = kb.fit_transform(X)
            mi = mutual_info_classif(X_discrete, Y)
            print(f"key= {key:30s} mi= {mi.item():.4f}")
    print("\n\n")


def analysis(name):
    stat_data = torch.load(f"./{name}-greedy-rbcd-gcn.pth")
    test_mask = stat_data['test_mask']
    stable_mask = stat_data['stable_mask']  # T-T
    frag_mask = stat_data['fragile_mask']  # T-F
    lucky_mask = stat_data['lucky_mask']  # F-T
    fool_mask = stat_data['foolish_mask']  # F-F

    Y = torch.zeros(test_mask.size(0), dtype=torch.long)
    Y[stable_mask] = 0
    Y[frag_mask] = 1
    Y[lucky_mask] = 2
    Y[fool_mask] = 3

    freeze_seed(15)

    print(f"Dataset= {name}")

    for key in stat_data.keys():
        if not key.endswith("mask") and not key.startswith("neighbor") and key != 'logits':
            kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
            X = stat_data[key][test_mask].numpy().reshape(-1, 1)
            X_dis = kb.fit_transform(X)
            mi = mutual_info_classif(X_dis, Y)
            print(f"key= {key:30s} mi= {mi.item():.4f}")
    print("\n\n")


if __name__ == '__main__':
    datasets = ['cora']
    for name in datasets:
        analysis(name)



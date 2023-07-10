import os
import sys

import torch

sys.path.insert(0, os.path.abspath('../'))

from models import model_map
from common.utils import *


import argparse
import networkx as nx

from torch_geometric.utils import convert




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'polblogs', 'pubmed'])
    parser.add_argument('--attack', type=str, default='greedy-rbcd')
    parser.add_argument('--victim', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sgc', 'graph-sage', 'rgcn', 'median-gcn'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, choices=[0.05, 0.10, 0.15, 0.20, 0.25])
    args = parser.parse_args()


    # 读取数据
    pyg_data = load_data(name=args.dataset)


    # 读取攻击后得到的adj
    perturbed_adj = load_perturbed_adj(args.dataset, args.attack, args.ptb_rate, path='./perturbed_adjs/')
    modified_adj_list = perturbed_adj['modified_adj_list']

    # 读取受害者模型
    victim_type = args.victim
    victim_pretrained_model = load_pretrained_model(args.dataset, victim_type, path='../victims/models/')
    victim_state_dicts = victim_pretrained_model['state_dicts']
    victim_config = victim_pretrained_model['config']

    device = get_device(args.gpu_id)
    logger = get_logger('tmp')
    victim = model_map[victim_type](config=victim_config, pyg_data=pyg_data, device=device, logger=logger)
    victim = victim.to(device)
    init_seed = victim_config['seed']
    # for i in range(1):
    i = 0
    freeze_seed(init_seed + i)
    victim.load_state_dict(victim_state_dicts[i])
    mod_adj = modified_adj_list[i]

    logits = victim.predict(x=pyg_data.x, edge_index=pyg_data.adj_t).cpu()
    clean_pred = logits.argmax(1)
    G = convert.to_networkx(pyg_data)

    degrees = calculate_degree(pyg_data.adj_t).cpu()
    degree_centrality = torch.as_tensor(list(nx.degree_centrality(G).values()))
    pagerank = torch.as_tensor(list(nx.pagerank(G).values()))
    clustering_coefficient = torch.as_tensor(list(nx.clustering(G).values()))
    betweenness_centrality = torch.as_tensor(list(nx.betweenness_centrality(G).values()))
    eigenvector_centrality = torch.as_tensor(list(nx.eigenvector_centrality(G).values()))
    cls_margin = classification_margin(logits, logits.argmax(1)).cpu()


    attacked_pred = victim.predict(x=pyg_data.x, edge_index=mod_adj).cpu().argmax(1)
    stable_mask = ((clean_pred == pyg_data.y) & (attacked_pred == pyg_data.y))
    fragile_mask = ((clean_pred == pyg_data.y) & (attacked_pred != pyg_data.y))
    lucky_mask = ((clean_pred != pyg_data.y) & (attacked_pred == pyg_data.y))
    foolish_mask = ((clean_pred != pyg_data.y) & (attacked_pred != pyg_data.y))


    save_file = f'{args.dataset}-{args.attack}-{args.victim}.pth'
    torch.save(obj={
        'degrees': degrees,
        'degree_centrality': degree_centrality,
        'pagerank': pagerank,
        'clustering_coefficient': clustering_coefficient,
        'betweenness_centrality': betweenness_centrality,
        'eigenvector_centrality': eigenvector_centrality,
        'cls_margin': cls_margin,
        'stable_mask': stable_mask,
        'fragile_mask': fragile_mask,
        'lucky_mask': lucky_mask,
        'foolish_mask': foolish_mask
    }, f=save_file)



if __name__ == '__main__':
    main()


import os
import sys

sys.path.insert(0, os.path.abspath('../'))

from models import model_map
from common.utils import *


import argparse
import networkx as nx

from torch_geometric.utils import convert




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=-1, choices=[-1, 0, 1, 2, 3])
    parser.add_argument('--dataset', type=str, default='cora_ml', choices=['cora', 'citeseer', 'pubmed', 'cora_ml', 'karate_club'])
    parser.add_argument('--attack', type=str, default='greedy-rbcd')
    parser.add_argument('--victim', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sgc', 'graph-sage', 'rgcn', 'median-gcn'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, choices=[0.05, 0.10, 0.15, 0.20, 0.25])
    args = parser.parse_args()


    # 读取数据
    pyg_data = load_data(name=args.dataset)
    test_mask = pyg_data.test_mask.cpu()
    perturbed_adj_data = load_perturbed_adj(args.dataset, args.attack, args.ptb_rate, path='../attack/perturbed_adjs/')
    modified_adj_list = perturbed_adj_data['modified_adj_list']
    selected_index = 0
    mod_adj = modified_adj_list[selected_index]
    clean_adj = pyg_data.adj_t
    dense_adj = clean_adj.to_dense()

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



    freeze_seed(init_seed + selected_index)
    victim.load_state_dict(victim_state_dicts[selected_index])


    logits = victim.predict(x=pyg_data.x, edge_index=clean_adj).cpu()
    clean_pred = logits.argmax(1)
    G = convert.to_networkx(pyg_data)

    # 如果计算复杂度过高就不用
    degrees = calculate_degree(pyg_data.adj_t).cpu()
    print("degrees计算完成")
    degree_centrality = torch.as_tensor(list(nx.degree_centrality(G).values()))
    print("degree_centrality计算完成")
    pagerank = torch.as_tensor(list(nx.pagerank(G).values()))
    print("pagerank计算完成")
    clustering_coefficient = torch.as_tensor(list(nx.clustering(G).values()))
    print("clustering_coefficient计算完成")
    # betweenness_centrality = torch.as_tensor(list(nx.betweenness_centrality(G).values()))
    # print("betweenness_centrality计算完成")
    eigenvector_centrality = torch.as_tensor(list(nx.eigenvector_centrality(G).values()))
    print("eigenvector_centrality计算完成")
    cls_margin = classification_margin(logits, logits.argmax(1)).cpu()
    print("cls_margin计算完成")


    attacked_pred = victim.predict(x=pyg_data.x, edge_index=mod_adj).cpu().argmax(1)
    stable_mask = ((clean_pred == pyg_data.y) & (attacked_pred == pyg_data.y))
    fragile_mask = ((clean_pred == pyg_data.y) & (attacked_pred != pyg_data.y))
    lucky_mask = ((clean_pred != pyg_data.y) & (attacked_pred == pyg_data.y))
    foolish_mask = ((clean_pred != pyg_data.y) & (attacked_pred != pyg_data.y))

    logits = logits.exp()
    neighbor_mean = torch.mm(dense_adj, logits) / dense_adj.sum(1).view(-1, 1)
    neighbor_var = (torch.mm(dense_adj, logits.pow(2)) / dense_adj.sum(1).view(-1, 1)).pow(1/2)
    neighbor_skewness = (torch.mm(dense_adj, logits.pow(3)) / dense_adj.sum(1).view(-1, 1)).pow(1/3)
    neighbor_mean[degrees == 0] = logits[degrees == 0]
    neighbor_var[degrees == 0] = logits[degrees == 0]
    neighbor_skewness[degrees == 0] = logits[degrees == 0]

    save_file = f'{args.dataset}-{args.victim}.pth'

    all_logits = []

    for state_dict in victim_state_dicts:
        victim.load_state_dict(state_dict)
        logit = victim.predict(x=pyg_data.x, edge_index=clean_adj).cpu()
        logit = logit.exp()
        all_logits.append(logit)

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
        'logits_all': all_logits,
        'neighbor_mean': neighbor_mean,
        'neighbor_var': neighbor_var,
        'neighbor_skewness': neighbor_skewness,
    }
    torch.save(obj=obj, f=save_file)



if __name__ == '__main__':
    main()


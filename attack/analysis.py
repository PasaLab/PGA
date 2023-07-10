import os
import sys

import torch

sys.path.insert(0, os.path.abspath('../'))

from models import model_map
from common.utils import *


import argparse
import pandas as pd




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--logger_level', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'polblogs', 'pubmed'])
    parser.add_argument('--attack', type=str, default='greedy-rbcd')
    parser.add_argument('--victim', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sgc', 'graph-sage', 'rgcn', 'median-gcn'])
    parser.add_argument('--ptb_rate', type=float, default=0.05, choices=[0.05, 0.10, 0.15, 0.20, 0.25])
    args = parser.parse_args()
    assert args.gpu_id in range(0, 4)
    assert args.logger_level in [0, 1, 2]

    logger_filename = "analyse-" + args.attack + '-' + args.dataset + '-' + args.victim + '.log'
    logger_name = 'analyse'
    logger = get_logger(logger_filename, level=args.logger_level, name=logger_name)
    logger.info(args)

    device = get_device(args.gpu_id)
    logger.info(f"Device: {device}")
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
    victim_performance = victim_pretrained_model['performance']
    logger.info(f"Loaded pretrained victim: {victim_type}")
    logger.info(f"Victim config: {victim_config}")
    logger.info(f"Surrogate performance: {victim_performance}\n\n")
    victim = model_map[victim_type](config=victim_config, pyg_data=pyg_data, device=device, logger=logger)
    victim = victim.to(device)

    n_perturbs = int(args.ptb_rate * (pyg_data.num_edges // 2))
    logger.info(f"Rate of perturbation: {args.ptb_rate}")
    logger.info(f"The number of perturbations: {n_perturbs}")


    init_seed = victim_config['seed']
    # for i in range(1):
    i = 0
    freeze_seed(init_seed + i)
    victim.load_state_dict(victim_state_dicts[i])
    mod_adj = modified_adj_list[i]


    statistics = torch.load(f'{args.dataset}-{args.attack}-{args.victim}.pth')

    stable_data = {}
    fragile_data = {}
    lucky_data = {}
    foolish_data = {}

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    for key in statistics.keys():
        if not key.endswith("mask"):
            stable_data[key] = statistics[key][statistics['stable_mask']].tolist()
            fragile_data[key] = statistics[key][statistics['fragile_mask']].tolist()
            lucky_data[key] = statistics[key][statistics['lucky_mask']].tolist()
            foolish_data[key] = statistics[key][statistics['foolish_mask']].tolist()

    stable_df = pd.DataFrame(stable_data)
    fragile_df = pd.DataFrame(fragile_data)
    lucky_df = pd.DataFrame(lucky_data)
    foolish_df = pd.DataFrame(foolish_data)

    stable_stats = stable_df.agg(['max', 'min', 'mean', 'median'])
    fragile_stats = fragile_df.agg(['max', 'min', 'mean', 'median'])
    lucky_stats = lucky_df.agg(['max', 'min', 'mean', 'median'])
    foolish_stats = foolish_df.agg(['max', 'min', 'mean', 'median'])

    stable_stats.columns = [f'{col}_fragile' for col in stable_stats.columns]
    fragile_stats.columns = [f'{col}_fragile' for col in fragile_stats.columns]
    lucky_stats.columns = [f'{col}_fragile' for col in lucky_stats.columns]
    foolish_stats.columns = [f'{col}_fragile' for col in foolish_stats.columns]

    combined_stats = pd.concat([stable_stats], axis=1).T

    print(combined_stats)
    # combined_stats.to_csv('analysis.csv', index=False)



if __name__ == '__main__':
    main()


















# features = []
    # for node in G.nodes():
    #     feature = [
    #         degrees[node],
    #         degree_centrality[node],
    #         pagerank[node],
    #         clustering_coefficient[node],
    #         betweenness_centrality[node],
    #         eigenvector_centrality[node],
    #         cls_margin[node]
    #     ]
    #     features.append(feature)
    # features = StandardScaler().fit_transform(features)
    #
    # kmeans = KMeans(n_clusters=2, random_state=42).fit(features)
    # labels = kmeans.labels_


    # test_index = torch.nonzero(pyg_data.test_mask).flatten()
    # train_index = torch.nonzero(~pyg_data.test_mask).flatten()
    #
    # X_all = np.array([
    #     list(degree_centrality.values()),
    #     list(pagerank.values()),
    #     list(clustering_coefficient.values()),
    #     list(betweenness_centrality.values()),
    #     list(eigenvector_centrality.values()),
    #     degrees.numpy(),
    #     cls_margin.numpy()
    # ]).T
    # y_all = labels.numpy()
    #
    # X, y = X_all[train_index], y_all[train_index]
    # X_test, y_test = X_all[test_index], y_all[test_index]
    #
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # clf = LogisticRegression(random_state=42).fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))


    # data = {'Node': list(degree_centrality.keys()), 'Degree Centrality': list(degree_centrality.values())}
    # df = pd.DataFrame.from_dict(data)
    # sns.histplot(data=df, x='Degree Centrality', binwidth=0.005)
    # plt.show()

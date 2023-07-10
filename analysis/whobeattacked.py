
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from common.utils import *

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
sns.set_context('paper')

attack_method = [
    'pgdattack',
    'prbcd',
    'greedy-rbcd',
    'pga',
]
name_map = {
    'pgdattack': 'PGDAttack',
    'prbcd': 'PRBCD',
    'greedy-rbcd': "GreedyRBCD",
    'pga': 'PGA(proposed)',
}
datasets = [
    'cora',
    'citeseer',
    'cora_ml',
    'pubmed',
]
dataset_map = {
    'cora': 'Cora',
    'citeseer': 'Citeseer',
    'cora_ml': 'CoraML',
    'pubmed': 'Pubmed',
}
select_range = [
    0,
    1,
    2,
    3,
    4,
]
ptb_rate = 0.05

use_load = True
to_save = True

to_maxes = [
    1, 1, 1, 5
]

if use_load:
    hit_count_map = torch.load(f'hit_count_map_{datasets[0]}.pt')
else:
    hit_count_map = dict()
    for method in attack_method:
        hit_count_map[method] = dict()
        for dataset in datasets:
            hit_count_map[method][dataset] = list()

    for ix, dataset in enumerate(datasets):

        to_max = to_maxes[ix]

        for select_index in select_range:

            freeze_seed(15 + select_index)

            # 加载数据集
            pyg_data = load_data(name=dataset)
            clean_adj = pyg_data.adj_t.to_dense()

            modified_adjs = []
            for method in attack_method:
                perturbed_data = load_perturbed_adj(dataset, method, ptb_rate, path='../attack/perturbed_adjs/')
                modified_adj = perturbed_data['modified_adj_list'][select_index]
                modified_adjs.append(modified_adj)



            target_records = torch.load(f"{dataset}-budgets.pth")
            budget_dict = target_records['budget']
            # print(budget_dict)
            budget_collect = dict()
            for n, b in budget_dict.items():
                if b not in budget_collect:
                    budget_collect[b] = list()
                budget_collect[b].append(n)

            weak_item = []
            for v in range(1, to_max+1):
                weak_item.extend(budget_collect[v])
            weak_item = torch.tensor(weak_item)

            for jx, method in enumerate(attack_method):
                mod_adj = modified_adjs[jx]
                diff = torch.nonzero(clean_adj != mod_adj.to_dense())
                src, dst = diff.t()
                mask1 = torch.isin(dst, weak_item)
                mask2 = torch.isin(src, weak_item)
                mask = mask1 | mask2

                hit_count_map[method][dataset].append(mask.sum().item() / 2 / (mask.size(0)))


        print("\n")
        for k, v in hit_count_map.items():
            print(f"{k:<15} {v}")

    if to_save:
        torch.save(obj=hit_count_map, f=f'hit_count_map_{datasets[0]}.pt')


data_long = []


for method, dts in hit_count_map.items():
    if method not in attack_method:
        continue
    for dt, hr in dts.items():
        if dt not in datasets:
            continue
        for v in hr:
            data_long.append((name_map[method], dataset_map[dt], v))



plt.figure(figsize=(12, 8))
df = pd.DataFrame(data_long, columns=['Attack', 'Dataset', 'Hit Rate'])
sns.set(font_scale=1.2)
sns.set(style="whitegrid")
sns.boxplot(x='Dataset', y='Hit Rate', hue='Attack', data=df,
             palette='Set3',
             saturation=0.55,
             dodge=True,
             linewidth=3.0,
             fliersize=2.0,
             notch=False, showcaps=True, showfliers=False,
             width=0.8)

ax = plt.gca()
# 设置坐标轴标签的字体大小
ax.tick_params(axis='x', labelsize=18)

# sns.barplot(x='Dataset', y='Hit Rate', hue='Attack', data=df,
#             errwidth=2,
#             linewidth=4)

save_postffix = datasets[0]
plt.legend(loc='lower right')
plt.xlabel(None)
plt.ylabel('Hit Rate', fontsize=20)
if len(datasets) == 4:
    save_postffix = "total"
    plt.axvline(x=0.5, color='red', linestyle='--', ymin=0.05, ymax=0.95, alpha=0.2)
    plt.axvline(x=1.5, color='red', linestyle='--', ymin=0.05, ymax=0.95, alpha=0.2)
    plt.axvline(x=2.5, color='red', linestyle='--', ymin=0.05, ymax=0.95, alpha=0.2)
plt.savefig(f"../pictures/hit_rate_{save_postffix}.pdf", format='pdf', dpi=1200)
plt.show()


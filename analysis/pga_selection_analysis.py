import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from torch_geometric.utils import index_to_mask
from common.utils import load_data

s1 = 'cora_selected_targets_0.2_0.6_0.9.pt'
s2 = 'citeseer_selected_targets_0.1_0.6_0.8.pt'
s3 = 'cora_ml_selected_targets_0.1_0.6_0.8.pt'
s4 = 'pubmed_selected_targets_0.05_0.95_0.8.pt'


def demo(s, name, prefix):
    s = torch.load('../attack/' + s)
    mod_cfg = torch.load(f'../attack/perturbed_adjs/{prefix}pga-{name}-0.05.pth')
    mod = mod_cfg['modified_adj_list'][0].to_dense()
    pyg_data = load_data(name=name)
    budgets = torch.load(f'../analysis/{name}-budgets.pth')['budget']

    selected_mask = index_to_mask(s, size=pyg_data.num_nodes)
    no_selected = pyg_data.test_mask & (~selected_mask)
    ts = torch.zeros(pyg_data.num_nodes)
    for k, v in budgets.items():
        ts[k] = v
    ns = ts[no_selected]
    ns = ns[ns.nonzero().squeeze()]
    sm = ts[selected_mask]
    sm = sm[sm.nonzero().squeeze()]
    nsnp = ns.numpy()
    smnp = sm.numpy()
    return nsnp, smnp, selected_mask






nsnp1, smnp1, _ = demo(s4, 'pubmed', prefix='')
bdg_eq_1 = (smnp1==1).sum()
bdg_eq_2 = (smnp1==2).sum()
print(bdg_eq_1, bdg_eq_2, smnp1.size)
# print(smnp2)

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 8))
# sns.set(font_scale=1.2)
# sns.set(style="whitegrid")
# # sns.histplot(
# #     [smnp1, nsnp1],
# #     bins=30,
# #     fill=True,
# #     thresh=0.05,
# #     color='red')
# sns.boxplot(
#     [
#         smnp1,
#         nsnp1,
#         # smnp2,
#         # nsnp2,
#     ],
#     palette='Set3',
#     saturation=0.55,
#     dodge=True,
#     whis=1.5,
#     linewidth=3.0,
#     fliersize=2.0,
#     notch=False, showcaps=True, showfliers=False,
#     width=0.8)
# plt.legend(
#     labels=[
#         'Selected Budget',
#         'Excluded Budget',
#     ],
# )
# plt.show()




import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import seaborn as sns
import matplotlib.pyplot as plt


# load benign data
from common.utils import load_data
cora_data = load_data(name='cora')
citeseer_data = load_data(name='citeseer')
cora_ml_data = load_data(name='cora_ml')
pubmed_data = load_data(name='pubmed')


def get_degree(data):
    adj = data.adj_t.to_dense()
    degree = adj.sum(1)
    return degree


cora_degree = get_degree(cora_data)
citeseer_degree = get_degree(citeseer_data)
cora_ml_degree = get_degree(cora_ml_data)
pubmed_degree = get_degree(pubmed_data)


# load perturbed adj
from common.utils import load_perturbed_adj
cora_p_adj = load_perturbed_adj('cora', 'pga', 0.05, path='../attack/perturbed_adjs/')['modified_adj_list'][0]
citeseer_p_adj = load_perturbed_adj('citeseer', 'pga', 0.05, path='../attack/perturbed_adjs/')['modified_adj_list'][0]
cora_ml_p_adj = load_perturbed_adj('cora_ml', 'pga', 0.05, path='../attack/perturbed_adjs/')['modified_adj_list'][0]
pubmed_p_adj = load_perturbed_adj('pubmed', 'pga', 0.05, path='../attack/perturbed_adjs/')['modified_adj_list'][0]


def get_degree_from_adj(adj):
    adj = adj.to_dense()
    degree = adj.sum(1)
    return degree


cora_p_degree = get_degree_from_adj(cora_p_adj)
citeseer_p_degree = get_degree_from_adj(citeseer_p_adj)
cora_ml_p_degree = get_degree_from_adj(cora_ml_p_adj)
pubmed_p_degree = get_degree_from_adj(pubmed_p_adj)


sns.set(context="paper", font_scale=1.2, style="whitegrid")
params = {
    'figure.figsize': (8, 6),  # 设置图片尺寸
    'figure.dpi': 300,  # 设置图形分辨率
}
plt.rcParams.update(params)


def plot(deg, p_deg, save_name='', save=True):
    # df = {
    #     'Group': ['Clean Graph'] * deg.size(0) + ['Adversarial Graph'] * p_deg.size(0),
    #     'Degree': deg.tolist() + p_deg.tolist(),
    # }
    #
    # # sns.kdeplot(data=df, x='Degree', hue=' ',
    # #             linewidth=2,
    # #             fill=True, hist=True)
    sns.distplot(deg.tolist(), hist=True, kde=True, label='Clean Graph', kde_kws={'linewidth': 2})
    sns.distplot(p_deg.tolist(), hist=True, kde=True, label='Adversarial Graph', kde_kws={'linewidth': 2})
    plt.xlabel('Node Degree')
    plt.legend()
    if save:
        plt.savefig(f"../pictures/{save_name}.pdf", format='pdf', dpi=1200)
    plt.show()


plot(cora_degree, cora_p_degree, 'unnoticeability_cora')
plot(citeseer_degree, citeseer_p_degree, 'unnoticeability_citeseer')
plot(cora_ml_degree, cora_ml_p_degree, 'unnoticeability_cora_ml')
plot(pubmed_degree, pubmed_p_degree, 'unnoticeability_pubmed')






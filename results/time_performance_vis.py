
cora_data = [
    [0.154, 0.207, 7.183, 12.158, 37.391, 19.886, 4.236],
    [81.57, 81.60, 75.67, 73.76, 72.26, 71.52, 70.20],
]

citeseer_data = [
    [0.160, 0.676, 7.183, 12.158, 32.458, 12.389, 4.700],
    [71.05, 70.68, 66.22, 66.55, 64.85, 64.39, 62.94],
]

cora_ml_data = [
    [0.08, 0.305, 14.204, 15.270, 56.122, 14.204, 7.571],
    [81.71, 81.28, 74.70, 72.89, 71.34, 70.76, 67.68],
]

pubmed_data = [
    [1.429, 4.048, 17662.156, 1797.404, 71.939, 41.486, 46.337],
    [77.68, 77.70, 58.51, 69.23, 45.70, 40.27, 38.74],
]

datas = {
    'cora': cora_data, 'citeseer': citeseer_data, 'cora_ml': cora_ml_data, 'pubmed': pubmed_data,
}


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

methods = [
    'Random', 'DICE', 'Greedy', 'PGDAttack', 'PRBCD', 'GreedyRBCD', 'PGA'
]

# cora
dataset = 'cora'

costs = datas[dataset][0]
err_rate = datas[dataset][1]

data = {
    'Attackers': methods,
    'Time Cost': costs,
    'Accuracy': err_rate,
    'sz': [5 * rt for rt in err_rate],
}

df = pd.DataFrame(data)
sns.set(context="paper", font_scale=1.2, style="whitegrid")
params = {
    'figure.figsize': (8, 6),  # 设置图片尺寸
    'figure.dpi': 300,  # 设置图形分辨率
}

plt.rcParams.update(params)
sns.scatterplot(
    x='Time Cost', y='Accuracy',
    data=df, hue='Attackers',
    # size='sz',
    style='Attackers',
    # sizes=(np.min(data['sz']), np.max(data['sz'])),
    # palette='Paired_r',
    markers="x",
    linewidth=2.5,
    s=120,
)

plt.title('Attack Effect vs. Time Cost')
if dataset == 'pubmed':
    plt.xlim(left=-10, right=100)
plt.xlabel('Time Cost (second)', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

# plt.legend(loc='lower right')
plt.savefig(f"../pictures/time_cost_{dataset}.pdf", format='pdf', dpi=1200)
plt.show()




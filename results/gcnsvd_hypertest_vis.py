
import json
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# GCN: 82.02, 2.27

k = [20, 40, 60, 80, 100, 120, 140, 160, 300, 1000]
k = [f"k-{str(kv)}" for kv in k]
k.append("gcn")
print(k)
acc_rate = [
    70.30,
    72.50,
    74.54,
    73.70,
    74.90,
    75.80,
    76.97,
    76.50,
    78.90,
    81.00,
    82.02,
]

costs = [
    12.26,
    13.03,
    13.88,
    14.25,
    14.56,
    14.80,
    15.12,
    16.04,
    17.60,
    41.15,
    2.27,
]

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Methods': k,
    'Time Cost': costs,
    'Acc Rate': acc_rate,
    'sz': [5 * rt for rt in acc_rate],
}

# print(data)

df = pd.DataFrame(data)
sns.set(context="paper", font_scale=1.2, style="whitegrid")
params = {
    'figure.figsize': (8, 6),  # 设置图片尺寸
    'figure.dpi': 300,  # 设置图形分辨率
}

plt.rcParams.update(params)
sns.scatterplot(
    x='Time Cost', y='Acc Rate',
    data=df, hue='Methods',
    # style='Methods',
    palette='Paired_r',
    markers="x",
    linewidth=2.5,
    s=120,
)

plt.title('Performance vs. Time Cost')
plt.xlabel('Time Cost (second)')
plt.ylabel('Acc Rate')

plt.legend(loc='lower right')
# plt.savefig(f"../pictures/time_cost_{dataset}.pdf", format='pdf', dpi=1200)
plt.show()
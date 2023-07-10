
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
sns.set_context('paper')
dataset = 'pubmed'
ptb_rate = 0.05
select_range = [0]

for select_index in select_range:

    target_records = torch.load(f"{dataset}-budgets.pth")
    budget_dict = target_records['budget']
    budget_collect = dict()
    for n, b in budget_dict.items():
        if b not in budget_collect:
            budget_collect[b] = list()
        budget_collect[b].append(n)

budget_dict = dict()

for k, v in budget_collect.items():
    budget_dict[k] = len(v)

keys = sorted(budget_dict.keys())
values = [budget_dict[key] for key in keys]
print(keys)
print(values)

data = pd.DataFrame({f'{dataset}': keys, 'y': values})

plt.figure(figsize=(12, 8))
# sns.set(font_scale=10.0)
sns.set(style="white")
# Create the distribution plot
p = sns.histplot(
    data=data,
    bins=range(21),
    x=f'{dataset}', weights='y',
    color='#4c72b0',
    alpha=0.5,
    edgecolor='k')
p.set_ylabel("The number of nodes", fontsize=30)
p.set_xlabel(f"Targeted attack budgets", fontsize=30)

plt.xlim((0, 21))
# Set x-axis labels as dictionary keys
plt.xticks(range(21))

plt.savefig(f"../pictures/budget_plot_{dataset}.pdf", format='pdf', dpi=1200)
plt.show()


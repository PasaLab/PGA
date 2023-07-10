
import json
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import pandas as pd

root_path = 'poisoning'
datasets = ['cora', 'citeseer', 'cora_ml', 'pubmed']
models = [
    'gcn', 'gat', 'sgc', 'graph-sage', 'appnp',
    'rgcn', 'median-gcn',
    'gcn-jaccard',
    'grand',
    'gnn-guard',
]

attackers = [
    'greedy',
    'pgdattack',
    'prbcd',
    'greedy-rbcd',
    'pga'
]

ptb_rate = [0.05]

df = pd.DataFrame(columns=['attack', 'dataset', 'victim', 'ptb_rate', 'accuracy'])

for atk in attackers:
    for rate in ptb_rate:
        for dt in datasets:
            for md in models:
                json_name = f"{root_path}/{atk}-{dt}-{md}-{rate}.json"
                with open(json_name) as f:
                    data = json.load(f)
                    # mean = data['attacked_acc'][:5]
                    # std = data['attacked_acc'][6:]
                    acc = data['attacked_acc']
                    df.loc[len(df.index)] = [atk, dt, md, rate, acc]
                    # df.append([atk, dt, md, rate, acc])


A = 'greedy'
B = 'cora'

sub_df = df[(df['attack'] == A) & (df['dataset'] == B)]['accuracy']

ss = ""
n = len(sub_df.keys())
for ix, item in enumerate(sub_df.items()):
    ss += f"&{item[1]}"
    if ix != n-1:
        ss += "\n"
    else:
        ss += "\\\\"

print(ss)

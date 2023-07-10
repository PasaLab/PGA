
import json
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

root_path = 'evasion'
dataset = 'pubmed'
model = 'normal'

attackers = [
    # 'greedy',
    # 'pgdattack',
    'prbcd',
    'greedy-rbcd',
    'pga',
]

attack_map = {
    'greedy': 'Greedy',
    'pgdattack': 'PGDAttack',
    'prbcd': 'PRBCD',
    'greedy-rbcd': 'GreedyRBCD',
    'pga': 'PGA',
}

ptb_rate = [0.01, 0.02, 0.03, 0.04, 0.05]

result_dict = dict()
for atk in attackers:
    result_dict[attack_map[atk]] = list()
    for rate in ptb_rate:
        # if atk == 'pgdattack-CW':
        #     model_name = 'gcn'
        # else:
        #     model_name = model
        json_name = f"{root_path}/{atk}-{dataset}-{model}-{rate}.json"
        with open(json_name) as f:
            data = json.load(f)
            acc = data['attacked_acc'][:5]
            result_dict[attack_map[atk]].append(float(acc))
    # print(f"method: {atk}, accs: {result_dict[atk]}")


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(result_dict)

sns.set(context="paper", font_scale=1.2, style="whitegrid")
params = {
    'figure.figsize': (8, 6),  # 设置图片尺寸
    'figure.dpi': 300,  # 设置图形分辨率
}
plt.rcParams.update(params)

sns.lineplot(
    data=df,
    linewidth=2,
    palette={
        'Greedy': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
        'PRBCD': (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
        'GreedyRBCD': (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
        'PGA': (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    },
)


for column in df.columns:
    sns.scatterplot(data=df[column], alpha=0.7, s=80)
plt.xticks(
    range(len(result_dict[attack_map['pga']])),
    ['0.01', '0.02', '0.03', '0.04', '0.05'])
plt.legend(loc='upper right')
plt.ylabel('Accuracy', fontsize=20)
plt.xlabel('Perturbed Rate', fontsize=20)
plt.savefig(f"../pictures/evasion_1-5_{dataset}.pdf", format='pdf', dpi=1200)
plt.show()
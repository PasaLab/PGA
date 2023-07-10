
import json
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

root_path = 'evasion'
dataset = 'citeseer'
model = 'gcn'


f_ratio = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

att_result = list()
for ratio in f_ratio:
    name = f"{root_path}/hyptest_{str(int(100*ratio))}_pga-{dataset}-{model}-0.05.json"
    with open(name) as f:
        data = json.load(f)
        acc = data['attacked_acc'][:5]
        att_result.append(float(acc))

print(att_result)


import seaborn as sns
import matplotlib.pyplot as plt


sns.set(context="paper", font_scale=1.2, style="whitegrid")
params = {
    'figure.figsize': (8, 6),  # 设置图片尺寸
    'figure.dpi': 300,  # 设置图形分辨率
}
plt.rcParams.update(params)
sns.lineplot(
    att_result,
    linewidth=2,
)
# for pt in att_result:
sns.scatterplot([att_result], alpha=0.7, s=80)
plt.xticks(
    range(len(f_ratio)),
    f_ratio)
plt.legend().remove()
plt.ylim(60, 62)
plt.xlabel('Filter Ratio')
plt.ylabel('Accuracy')
plt.savefig(f"../pictures/hyptest_f_ratio_{dataset}.pdf", format='pdf', dpi=1200)
plt.show()
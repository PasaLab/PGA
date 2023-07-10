
import os
import sys

import torch

sys.path.insert(0, os.path.abspath('../'))

from attackers import SGAttack
from deeprobust.graph.defense import SGC

from models import model_map
from common.utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--logger_level', type=int, default=0)
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'cora_ml'])
parser.add_argument('--victim', type=str, default='gcn')
parser.add_argument('--ptb_rate', type=float, default=0.05)
args = parser.parse_args()
assert args.gpu_id in range(0, 4)
assert args.logger_level in [0, 1, 2]

logger_filename = "evaluate-" + '-' + args.dataset + '-' + args.victim + '.log'
logger = get_logger(logger_filename, level=args.logger_level, name='evaluate')
device = get_device(args.gpu_id)
# 读取数据
init_seed = 15
freeze_seed(init_seed)
pyg_data = load_data(name=args.dataset)
features = pyg_data.x
labels = pyg_data.y
raw_adj = pyg_data.adj_t

pretrained_models = load_pretrained_model(args.dataset, args.victim, path='../victims/models/')
state_dicts = pretrained_models['state_dicts'][0]
config = pretrained_models['config']

victim = model_map[args.victim](config=config, pyg_data=pyg_data, device=device, logger=logger)
victim = victim.to(device)
victim.load_state_dict(state_dicts)
victim.test(pyg_data.test_mask)
clean_pred = victim.predict().cpu().argmax(1)


def construct_attacker():
    surrogate = SGC(nfeat=features.shape[1],
                    nclass=labels.max().item() + 1, K=2,
                    lr=0.01, device=device).to(device)
    surrogate.fit([pyg_data], verbose=False)  # train with earlystopping
    surrogate.test()
    attacker = SGAttack(surrogate, attack_structure=True, attack_features=False, device=device)
    attacker = attacker.to(device)
    return attacker


attack_result = dict()


def single_attack(attacker, target_node):
    n_perturbations = 1
    adj_csr = pyg_data.adj_t.to_scipy(layout='csr')
    while True:
        if n_perturbations >= 20:
            return
        attacker.attack(features.to(device), adj_csr, labels, target_node, n_perturbations, direct=True)
        modified_adj = SparseTensor.from_scipy(attacker.modified_adj)
        a, b = single_test(target_node, modified_adj)
        if a == b:
            n_perturbations += 1
            continue
        break
    logger.debug(f"Budget on {target_node}: {n_perturbations}")
    attack_result[target_node] = n_perturbations


def single_test(target_node, mod_adj):
    mod_pred = victim.predict(x=features, edge_index=mod_adj).cpu().argmax(1)[target_node]
    return clean_pred[target_node], mod_pred


attacker = construct_attacker()
idx_test = torch.nonzero(pyg_data.test_mask).flatten().tolist()
for node_idx in idx_test:
    if clean_pred[node_idx] == pyg_data.y[node_idx]:
        single_attack(attacker, node_idx)

torch.save(obj={
    'budget': attack_result
}, f=f'{args.dataset}-budgets-{args.victim}.pth')



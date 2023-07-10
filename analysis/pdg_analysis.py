
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

from models import model_map
from common.utils import *


modified_index = 0
dataset = 'cora'
assert dataset in ['cora', 'citeseer', 'cora_ml', 'pubmed']

pyg_data = load_data(name=dataset)
clean_adj = pyg_data.adj_t.to_dense()
perturbed_data = load_perturbed_adj(dataset, 'pgdattack', 0.05, path='../attack/perturbed_adjs/')
modified_adj = perturbed_data['modified_adj_list'][modified_index].tp_dense()




from .gcn import GCN, DenseGCN, GCNSVD, GCNJaccard, GNNGuard
from .sgc import SGC
from .gat import GAT
from .rgcn import RGCN
from .median_gcn import MedianGCN
from .medoid_gcn import MedoidGCN
from .graph_sage import GraphSage
from .grand import Grand
from .simp_gcn import SimpGCN
from .appnp import APPNP

model_map = {
    'gcn': GCN,
    'sgc': SGC,
    'gat': GAT,
    'graph-sage': GraphSage,
    'rgcn': RGCN,
    'median-gcn': MedianGCN,
    'gcnsvd': GCNSVD,
    'gcn-jaccard': GCNJaccard,
    'gnn-guard': GNNGuard,
    'grand': Grand,
    'simp-gcn': SimpGCN,
    'dense-gcn': DenseGCN,
    'appnp': APPNP,
}

normal_model = [
    'gcn',
    'sgc',
    'gat',
    'graph-sage',
    'appnp',
]
robust_model = [
    'rgcn',
    'median-gcn',
    'gcn-jaccard',
    'grand',
    'gnn-guard',
]
all_model = list(set(normal_model).union(set(robust_model)))


choice_map = dict()
choice_map['normal'] = normal_model
choice_map['robust'] = robust_model
choice_map['total'] = all_model
for item in all_model:
    choice_map[item] = [item]

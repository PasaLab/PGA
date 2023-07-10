

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor

from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

from common.utils import gen_pseudo_label



class AttackABC(ABC):
    """
    attack class的基类
    构造函数inputs：
        - 超参数attack_config，PRBCD、GreedyRBCD、PGA等
        - 数据输入pyg_data
        - 代理模型/攻击对象model，如果直接把攻击对象传进来就是白盒攻击；只是利用model生成adj_adversary，然后攻击其他模型，就是灰盒攻击
        - GPU/CPU device
        - 日志logger
    """
    def __init__(self,
                 attack_config, pyg_data,
                 model, device, logger):
        self.device = device
        self.attack_config = attack_config
        self.logger = logger

        self.loss_type = attack_config['loss_type']
        # self.attacked_model = deepcopy(model).to(self.device)
        self.attacked_model = model  # 注意这里直接引用, 源代码实现是深拷贝
        self.attacked_model.eval()

        self.pyg_data = deepcopy(pyg_data)
        # if self.__class__.__name__ not in ['DICE', 'Random']:  # 给这两个方法真实标签
        #     pseudo_label = gen_pseudo_label(self.attacked_model, self.pyg_data.y, self.pyg_data.test_mask)
        #     self.pyg_data.y = pseudo_label
        pseudo_label = gen_pseudo_label(self.attacked_model, self.pyg_data.y, self.pyg_data.test_mask)
        self.pyg_data.y = pseudo_label
        self.pyg_data = self.pyg_data.to(self.device)


        for p in self.attacked_model.parameters():
            p.requires_grad = False
        self.eval_model = self.attacked_model

        self.attr_adversary = self.pyg_data.x
        self.adj_adversary = self.pyg_data.adj_t


    @abstractmethod
    def _attack(self, n_perturbations):
        pass

    def attack(self, n_perturbations, **kwargs):
        if n_perturbations > 0:
            return self._attack(n_perturbations, **kwargs)
        else:
            self.attr_adversary = self.pyg_data.x
            self.adj_adversary = self.pyg_data.adj_t

    def get_perturbations(self):
        adj_adversary, attr_adversary = self.adj_adversary, self.attr_adversary
        if isinstance(self.adj_adversary, torch.Tensor):
            adj_adversary = SparseTensor.from_dense(self.adj_adversary)
        if isinstance(self.attr_adversary, SparseTensor):
            attr_adversary = self.attr_adversary.to_dense()

        return adj_adversary, attr_adversary

    def calculate_loss(self, logits, labels):
        if self.loss_type == 'CW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -torch.clamp(margin, min=0).mean()
        elif self.loss_type == 'LCW':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.leaky_relu(margin, negative_slope=0.1).mean()
        elif self.loss_type == 'tanhMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'Margin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -margin.mean()
        elif self.loss_type.startswith('tanhMarginCW-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = (alpha * torch.tanh(-margin) - (1 - alpha) * torch.clamp(margin, min=0)).mean()
        elif self.loss_type.startswith('tanhMarginMCE-'):
            alpha = float(self.loss_type.split('-')[-1])
            assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
            assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'

            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )

            not_flipped = logits.argmax(-1) == labels

            loss = alpha * torch.tanh(-margin).mean() + (1 - alpha) * \
                F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'eluMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = -F.elu(margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'NCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss


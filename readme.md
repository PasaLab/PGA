

## Introduction
The implementation of the paper "Simple and Efficient Partial Graph Adversarial Attack: A New Perspective", under the setting of global attack, treats different nodes differently to perform more efficient adversarial attacks. 


## Main Structure

- models: implementation of GNN models
- victims: experiments for training
  - configs: configurations of models
  - models: trained models
- attackers: implementation of attack methods
- attack: experiments for attacking
  - configs: hyperparameter of attackers
  - perturbed_adjs: adversarial adj generated

## Running Step
1. training models
```
> cd victims
> python train.py --model=gcn --dataset=cora
```
2. performing attacks
```
> cd attack
> python gen_attack.py 
```

## PGA 
1. training models
```
> cd victims
> python train.py
```
2. performing attack
```
> cd attack
> python gen_attack.py --attack=pga --dataset=cora
```

## Evaluation (evasion attack)
```
> python evasion_attack.py --victim=robust --dataset=cora
> python evasion_attack.py --victim=normal --dataset=cora
```


## Evaluation (poisoning attack)
```
> python poison_attack.py --victim=gcn --dataset=cora
> python poison_attack.py --victim=gat --dataset=cora
```

## requirements
- deeprobust
- torch_geometry
- torch_sparse
- torch_scatter

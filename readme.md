
## 主体结构

- models: GNN模型实现
- victims: 训练
  - configs: 模型训练参数
  - models: 保存的模型
- attackers: 攻击代码实现
- attack: 执行攻击
  - configs: 攻击参数
  - perturbed_adjs: 生成的对抗图

## 运行步骤
1. 训练模型
```
> cd victims
> python train.py --model=gcn --dataset=cora
```
2. 执行攻击
```
> cd attack
> python gen_attack.py 
```

## PGA攻击
1. 训练模型
```
> cd victims
> python train.py
```
2. 生成图的一些统计信息，例如结点度、classification margin
```
> cd analysis
> python gen_statistics.py --dataset=cora
```
3. 执行攻击
```
> cd attack
> python gen_attack.py --attack=pga --dataset=cora
```

## 评估(evasion attack)
```
> python evasion_attack.py --victim=robust --dataset=cora
> python evasion_attack.py --victim=normal --dataset=cora
```


## 评估(poisoning attack)
```
> python poison_attack.py --victim=gcn --dataset=cora
> python poison_attack.py --victim=gat --dataset=cora
```

## requirements
- deeprobust
- torch_geometry
- torch_sparse
- torch_scatter
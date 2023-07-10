import os.path as osp
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import argparse
import yaml
from yaml import SafeLoader
import time

from common.utils import *
from attackers import attacker_map
from models import model_map, choice_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--logger_level', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'cora_ml'])
    parser.add_argument('--attack', type=str, default='pga', choices=['prbcd', 'greedy-rbcd', 'apga', 'pga', 'pgdattack', 'graD', 'greedy', 'dice', 'random'])
    parser.add_argument('--victim', type=str, default='normal')
    parser.add_argument('--ptb_rate', type=float, default=0.05)
    parser.add_argument('--n_running', type=int, default=5)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--save_prefix', type=str, default="")
    args = parser.parse_args()
    assert args.gpu_id in range(0, 4)
    assert args.logger_level in [0, 1, 2]

    # load attack config
    config_file = osp.join(osp.expanduser('./configs'), args.attack + '.yaml')
    attack_config = yaml.load(open(config_file), Loader=SafeLoader)[args.dataset]
    logger_filename = f"./logs/{args.attack}-{args.dataset}-{args.victim}-{args.ptb_rate}.log"
    logger_name = 'attack_train'
    logger = get_logger(logger_filename, level=args.logger_level, name=logger_name)
    logger.info(args)

    logger.info(f"Attack config: {attack_config}")

    device = get_device(args.gpu_id)
    logger.info(f"Attacker's NAME: {args.attack}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}")


    # load surrogate
    surrogate_type = attack_config['surrogate']
    surrogate_pretrained_model = load_pretrained_model(args.dataset, surrogate_type, path='../victims/models/')
    surrogate_state_dicts = surrogate_pretrained_model['state_dicts']
    surrogate_config = surrogate_pretrained_model['config']

    init_seed = surrogate_config['seed']
    freeze_seed(init_seed)
    # load dataset
    pyg_data = load_data(name=args.dataset)  # polblogs数据集对随机种子有依赖
    logger.info(f"Dataset Information: ")
    logger.info(f"\t\t The Number of Nodes: {pyg_data.num_nodes}")
    logger.info(f"\t\t The Number of Edges: {pyg_data.num_edges}")
    logger.info(f"\t\t The Dimension of Features: {pyg_data.num_features}")
    logger.info(f"\t\t The Number of Classes: {pyg_data.num_classes}")
    logger.info(f"\t\t Split(train/val/test): "
                f"{pyg_data.train_mask.sum().item()}, "
                f"{pyg_data.val_mask.sum().item()}, "
                f"{pyg_data.test_mask.sum().item()}")

    surrogate_performance = surrogate_pretrained_model['performance']
    surrogate = model_map[surrogate_type](config=surrogate_config, pyg_data=pyg_data, device=device, logger=logger)
    surrogate = surrogate.to(device)
    logger.info("\n\n")
    logger.info(f"Loaded pretrained surrogate: {surrogate_type}")
    logger.info(f"Surrogate config: {surrogate_config}")
    logger.info(f"Surrogate performance: {surrogate_performance}\n\n")


    victim_types = []
    choices = choice_map[args.victim]
    victim_types.extend(choices)

    victims = []
    attack_acc_dict = dict()
    clean_acc_dict = dict()
    for victim_name in victim_types:
        victim_pretrained_model = load_pretrained_model(args.dataset, victim_name, path='../victims/models/')
        victim_state_dicts = victim_pretrained_model['state_dicts']
        victim_config = victim_pretrained_model['config']
        victim = model_map[victim_name](config=victim_config, pyg_data=pyg_data, device=device, logger=logger)
        victim = victim.to(device)

        victims.append({
            'name': victim_name,
            'model': victim,
            'state_dicts': victim_state_dicts,
            'configs': victim_config,
            'performance': victim_pretrained_model['performance']
        })
        attack_acc_dict[victim_name] = list()
        clean_acc_dict[victim_name] = list()



    n_perturbs = int(args.ptb_rate * (pyg_data.num_edges // 2))
    logger.info(f"Rate of perturbation: {args.ptb_rate}")
    logger.info(f"The number of perturbations: {n_perturbs}")


    modified_adj_list = []
    time_cost_list = []


    n_running = min(len(victims[0]['state_dicts']), args.n_running)

    for i in range(n_running):  # 注意这里每次生成一个新的attacker, 额外开销其实不大；如果每个running修改全局attacker数据，会让代码稍乱一点
        freeze_seed(init_seed + i)
        # victim.load_state_dict(victim_state_dicts[i])
        surrogate.load_state_dict(surrogate_state_dicts[i])
        attacker = attacker_map[args.attack](
            attack_config=attack_config, pyg_data=pyg_data,
            model=surrogate, device=device, logger=logger, dataset_name=args.dataset,
        )

        start = time.time()
        attacker.attack(
            n_perturbs,
            dataset=args.dataset,
        )
        time_cost_list.append(time.time()-start)

        mod_adj, _ = attacker.get_perturbations()
        mod_rate = calc_modified_rate(pyg_data.adj_t, mod_adj, pyg_data.num_edges)
        logger.debug(f"Modified rate: {mod_rate:.2f}")
        assert mod_rate <= args.ptb_rate
        modified_adj_list.append(deepcopy(mod_adj.detach().cpu()))

        logger.info(f"Running[{i + 1:03d}], time cost= {time_cost_list[-1]:.3f}")

        for victim in victims:
            name = victim['name']
            model = victim['model']
            model.load_state_dict(victim['state_dicts'][i])
            attack_acc = evaluate_attack_performance(model, pyg_data.x, mod_adj, pyg_data.y, pyg_data.test_mask)
            attack_acc_dict[name].append(attack_acc)

            clean_acc = model.test(test_mask=pyg_data.test_mask, verbose=False)
            clean_acc_dict[name].append(clean_acc)

            logger.info(f"name= {name:15s} clean accuracy= {clean_acc * 100:.2f}, attacked accuracy= {attack_acc * 100:.2f}")


    total_mean = []
    total_std = []

    for k in attack_acc_dict.keys():
        attack_accs = attack_acc_dict[k]
        clean_accs = clean_acc_dict[k]
        total_mean.append(float(f"{np.mean(attack_accs) * 100:.2f}"))
        total_std.append(float(f"{np.std(attack_accs) * 100:.2f}"))
        logger.info(f"victim= {k:15s} clean acc= {np.mean(clean_accs) * 100:.2f}{chr(177)}{np.std(clean_accs) * 100:.2f}, "
                    f"attacked acc= {np.mean(attack_accs)*100:.2f}{chr(177)}{np.std(attack_accs)*100:.2f}"
                    f"\t#ptb_rate= {args.ptb_rate}")
        if args.save:
            attack_name = args.attack
            if attack_name == 'pgdattack':
                loss_name = attack_config['loss_type']
                if loss_name != 'tanhMargin':
                    attack_name = attack_name + "-" + loss_name
            attack_name = args.save_prefix + attack_name
            save_result_to_json(
                attack=attack_name,
                dataset=args.dataset,
                victim=k,
                ptb_rate=args.ptb_rate,
                attacked_acc=f"{np.mean(attack_accs)*100:.2f}{chr(177)}{np.std(attack_accs)*100:.2f}",
                attack_type='evasion',
            )

    if args.victim == 'all':
        if args.save:
            attack_name = args.attack
            if attack_name == 'pgdattack':
                loss_name = attack_config['loss_type']
                if loss_name != 'tanhMargin':
                    attack_name = attack_name + "-" + loss_name
            attack_name = args.save_prefix + attack_name
            save_result_to_json(
                attack=attack_name,
                dataset=args.dataset,
                victim='all',
                ptb_rate=args.ptb_rate,
                attacked_acc=f"{np.mean(total_mean):.2f}{chr(177)}{np.mean(total_std):.2f}",
                attack_type='evasion',
            )
        logger.info(f"Averaged Attack Performance= {np.mean(total_mean):.2f}{chr(177)}{np.mean(total_std):.2f}")



    if args.save:
        save_path = "./perturbed_adjs/"
        if not osp.exists(save_path):
            os.makedirs(save_path)
        attack_name = args.attack
        if attack_name == 'pgdattack':
            loss_name = attack_config['loss_type']
            if loss_name != 'tanhMargin':
                attack_name = attack_name + "-" + loss_name
        attack_name = args.save_prefix + attack_name
        save_path += f"{attack_name}-{args.dataset}-{args.ptb_rate}.pth"
        torch.save(obj={
            'modified_adj_list': modified_adj_list,
            'attack_config': attack_config,
        }, f=save_path)


if __name__ == '__main__':
    main()
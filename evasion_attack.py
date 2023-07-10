import os
from models import model_map, choice_map
from common.utils import *


import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--logger_level', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'citeseer', 'pubmed', 'cora_ml'])
    parser.add_argument('--attack', type=list, default=[
        # 'random',
        # 'dice',
        # 'greedy',
        # 'pgdattack-CW',
        # 'prbcd',
        # 'greedy-rbcd',
        'pga',
    ])
    parser.add_argument('--victim', type=str, default='robust')
    parser.add_argument('--ptb_rate', type=float, default=0.05)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--cmp', type=bool, default=True)
    args = parser.parse_args()
    assert args.gpu_id in range(0, 4)
    assert args.logger_level in [0, 1, 2]

    # attacker_name = args.attack[0]

    rng = 1 if not args.cmp else len(args.attack)

    logger_filename = 'evasion_attack-' + args.dataset + '-' + args.victim + '.log'
    logger_name = 'evaluate'
    logger = get_logger(logger_filename, level=args.logger_level, name=logger_name)
    logger.info(args)

    device = get_device(args.gpu_id)
    logger.info(f"Device: {device}")
    # 读取数据
    init_seed = 15
    freeze_seed(init_seed)
    pyg_data = load_data(name=args.dataset)

    n_perturbs = int(args.ptb_rate * (pyg_data.num_edges // 2))
    logger.info(f"Rate of perturbation: {args.ptb_rate}")
    logger.info(f"The number of perturbations: {n_perturbs}")

    for ix in range(rng):

        logger.info("\n\n")
        attacker_name = args.attack[ix]

        # 读取攻击后得到的adj
        perturbed_adj = load_perturbed_adj(args.dataset, attacker_name, args.ptb_rate, path='./attack/perturbed_adjs/')
        modified_adj_list = perturbed_adj['modified_adj_list']

        victims = []
        choices = choice_map[args.victim]
        victims.extend(choices)


        raw_total_mean = []

        total_mean = []
        total_std = []

        for name in victims:
            pretrained_models = load_pretrained_model(args.dataset, name, path='./victims/models/')
            state_dicts = pretrained_models['state_dicts']
            config = pretrained_models['config']
            clean_performance = pretrained_models['performance']

            victim = model_map[name](config=config, pyg_data=pyg_data, device=device, logger=logger)
            victim = victim.to(device)

            n_running = len(state_dicts)

            attack_acc_list = []
            # clean_acc_list = []
            init_seed = config['seed']
            for i in range(n_running):
                freeze_seed(init_seed + i)
                victim.load_state_dict(state_dicts[i])
                mod_adj = modified_adj_list[i]
                attack_acc = evaluate_attack_performance(victim, pyg_data.x, mod_adj, pyg_data.y, pyg_data.test_mask)
                attack_acc_list.append(attack_acc)

            total_mean.append(float(f"{np.mean(attack_acc_list) * 100:.2f}"))
            total_std.append(float(f"{np.std(attack_acc_list) * 100:.2f}"))
            logger.info(f"Clean Acc= {clean_performance}, "
                        f"Attacked Acc= {np.mean(attack_acc_list) * 100:.2f}{chr(177)}{np.std(attack_acc_list) * 100:.2f} \tModel= {name}")
            raw_acc = float(clean_performance.split(f'{chr(177)}')[0])
            raw_total_mean.append(raw_acc)

            if args.save:
                save_result_to_json(
                    attack=attacker_name,
                    dataset=args.dataset,
                    victim=name,
                    ptb_rate=args.ptb_rate,
                    attacked_acc=f"{np.mean(attack_acc_list) * 100:.2f}{chr(177)}{np.std(attack_acc_list) * 100:.2f}",
                    attack_type='evasion',
                )

        print("\n\n")
        logger.info(f"Averaged Attack Performance= {np.mean(total_mean):.2f}{chr(177)}{np.mean(total_std):.2f}")
        logger.info(f"Averaged Benigh Performance= {np.mean(raw_total_mean):.2f}")

        if args.victim in ['robust', 'normal', 'total'] and args.save:
            save_result_to_json(
                attack=attacker_name,
                dataset=args.dataset,
                victim=args.victim,
                ptb_rate=args.ptb_rate,
                attacked_acc=f"{np.mean(total_mean):.2f}{chr(177)}{np.mean(total_std):.2f}",
                attack_type='evasion',
            )


if __name__ == '__main__':
    main()


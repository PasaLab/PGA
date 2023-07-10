
from models import model_map, choice_map
from common.utils import *
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--logger_level', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'citeseer', 'pubmed', 'cora_ml'])
    parser.add_argument('--attack', type=str, default='pga')
    parser.add_argument('--victim', type=str, default='gnn-guard')
    parser.add_argument('--ptb_rate', type=float, default=0.05, choices=[0.05, 0.10, 0.15, 0.20, 0.25])
    args = parser.parse_args()
    assert args.gpu_id in range(0, 4)
    assert args.logger_level in [0, 1, 2]

    logger_filename = "poison-" + args.attack + '-' + args.dataset + '-' + args.victim + '.log'
    logger_name = 'poison attack'
    logger = get_logger(logger_filename, level=args.logger_level, name=logger_name)
    logger.info(args)

    device = get_device(args.gpu_id)
    logger.info(f"Device: {device}")
    # 读取数据
    init_seed = 15
    freeze_seed(init_seed)
    pyg_data = load_data(name=args.dataset)

    # 读取攻击后得到的adj
    perturbed_adj = load_perturbed_adj(args.dataset, args.attack, args.ptb_rate, path='./attack/perturbed_adjs/')
    modified_adj_list = perturbed_adj['modified_adj_list']
    n_running = len(modified_adj_list)

    for mod_adj_t in modified_adj_list:
        check_undirected(mod_adj_t)

    n_perturbs = int(args.ptb_rate * (pyg_data.num_edges // 2))
    logger.info(f"Rate of perturbation: {args.ptb_rate}")
    logger.info(f"The number of perturbations: {n_perturbs}")

    victims = []
    choices = choice_map[args.victim]
    victims.extend(choices)

    # if args.dataset == 'pubmed' and 'gcnsvd' in victims:
    #     victims.remove('gcnsvd')

    total_mean = []
    total_std = []
    attack_performance = []


    for name in victims:
        pretrained_models = load_pretrained_model(args.dataset, name, path='./victims/models/')
        config = pretrained_models['config']
        performance = pretrained_models['performance']

        attack_acc_list = []
        init_seed = config['seed']
        for i in range(n_running):
            freeze_seed(init_seed + i)
            victim = model_map[name](config=config, pyg_data=pyg_data, device=device, logger=logger)
            victim = victim.to(device)

            if len(modified_adj_list) > i:
                mod_adj = modified_adj_list[i]
            else:
                mod_adj = modified_adj_list[-1]
            pyg_data.adj_t = mod_adj
            victim.fit(pyg_data, verbose=True)
            attack_acc_list.append(victim.test(pyg_data.test_mask))

        total_mean.append(float(f"{np.mean(attack_acc_list) * 100:.2f}"))
        total_std.append(float(f"{np.std(attack_acc_list) * 100:.2f}"))
        attack_str = f"Clean Acc= {performance}, " \
              f"Attacked Acc= {np.mean(attack_acc_list) * 100:.2f}{chr(177)}{np.std(attack_acc_list) * 100:.2f} \tModel= {name}"
        logger.info(attack_str)
        attack_performance.append(attack_str)

        save_result_to_json(
            attack=args.attack,
            dataset=args.dataset,
            victim=name,
            ptb_rate=args.ptb_rate,
            attacked_acc=f"{np.mean(attack_acc_list) * 100:.2f}{chr(177)}{np.std(attack_acc_list) * 100:.2f}",
            attack_type='poisoning',
        )

    logger.info("\n\n")
    for acc_str in attack_performance:
        logger.info(acc_str)
    logger.info("\n\n")
    logger.info(f"Averaged Attack Performance= {np.mean(total_mean):.2f}{chr(177)}{np.mean(total_std):.2f}")

    if args.victim in ['robust', 'normal', 'total']:
        save_result_to_json(
            attack=args.attack,
            dataset=args.dataset,
            victim=args.victim,
            ptb_rate=args.ptb_rate,
            attacked_acc=f"{np.mean(total_mean):.2f}{chr(177)}{np.mean(total_std):.2f}",
            attack_type='poisoning',
        )


if __name__ == '__main__':
    main()


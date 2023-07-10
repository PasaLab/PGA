

from .prbcd import PRBCD
from .greedy_rbcd import GreedyRBCD
from .pga import PGA
from .topology_attack import PGDAttack
from .greedy import Greedy
from .dice import DICE
from .random_attack import RandomAttack
from .sga import SGAttack


attacker_map = {
    'prbcd': PRBCD, 
    'greedy-rbcd': GreedyRBCD,
    'pga': PGA,
    'pgdattack': PGDAttack,
    'greedy': Greedy,
    'dice': DICE,
    'random': RandomAttack,
    'sga': SGAttack,
}
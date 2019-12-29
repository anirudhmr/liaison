from collections import namedtuple
from typing import Dict, List, Tuple

from liaison.daper import ConfigDict

MILPHeuristic = lambda: ConfigDict(random=ConfigDict(
    seeds=List[int], n_local_moves=int, results=List[Dict]), )

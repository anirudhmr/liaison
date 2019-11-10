from collections import namedtuple
from typing import List, Tuple

import numpy as np
from daper import ConfigDict

BaselineResultsDict = lambda: ConfigDict(
    path=List[int], objective=float, solve_time=float)

# locs -> np.ndarray with last dimension = 2
# concorde_path list of nodes
# objective value achieved by concorde
# concorde_solve_time: Time required for concorde to solve in seconds.
# concorde_seed: int
TSP = lambda: ConfigDict(
    locs=List[Tuple[float, float]],
    seed=int,
    # baseline features.
    baseline_results=ConfigDict(concorde=BaselineResultsDict(),
                                gurobi=BaselineResultsDict(),
                                insertion_heuristics=ConfigDict(
                                    random=BaselineResultsDict(),
                                    nearest=BaselineResultsDict(),
                                    farthest=BaselineResultsDict())))

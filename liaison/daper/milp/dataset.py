from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
from liaison.daper import ConfigDict
from liaison.daper.milp.primitives import MIPInstance

MILP = lambda: ConfigDict(
    mip=MIPInstance,
    problem_type=str,
    problem_size=int,
    # Optimal objective achieved.
    optimal_objective=float,
    optimal_solution=Dict[str, int],
    optimal_sol_metadata=ConfigDict(
        n_nodes=int,
        gap=float,
        primal_integral=float,
        primal_gaps=List[float],
        n_sum=int,
    ),
    # whether the solution is actually optimal
    # as opposed to the best within timelimit.
    is_optimal=bool,
    feasible_objective=float,
    feasible_solution=Dict[str, int],
    seed=int,
)

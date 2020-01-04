from collections import namedtuple
from typing import Dict, List, Tuple

from liaison.daper import ConfigDict

MILPHeuristic = lambda: ConfigDict(
    random=ConfigDict(
        seeds=List[int],
        n_local_moves=int,
        k=int,
        results=List[List[Dict]],
    ),
    least_integral=ConfigDict(
        seeds=List[int],
        n_local_moves=int,
        k=int,
        results=List[List[Dict]],
    ),
    most_integral=ConfigDict(
        seeds=List[int],
        n_local_moves=int,
        k=int,
        results=List[List[Dict]],
    ),
    rins=ConfigDict(
        seeds=List[int],
        n_local_moves=int,
        k=int,
        results=List[List[Dict]],
    ),
)

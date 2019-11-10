import os
import time
import uuid
from subprocess import CalledProcessError, check_call, check_output

import numpy as np


def read_concorde_tour(filename):
  with open(filename, 'r') as f:
    n = None
    tour = []
    for line in f:
      if n is None:
        n = int(line)
      else:
        tour.extend([int(node) for node in line.rstrip().split(" ")])
  assert len(tour) == n, "Unexpected tour length"
  return tour


def read_tsplib(filename):
  with open(filename, 'r') as f:
    tour = []
    dimension = 0
    started = False
    for line in f:
      if started:
        loc = int(line)
        if loc == -1:
          break
        tour.append(loc)
      if line.startswith("DIMENSION"):
        dimension = int(line.split(" ")[-1])

      if line.startswith("TOUR_SECTION"):
        started = True

  assert len(tour) == dimension
  tour = np.array(tour).astype(
      int) - 1  # Subtract 1 as depot is 1 and should be 0
  return tour.tolist()


def calc_tsp_length(loc, tour):
  assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
  assert len(tour) == len(loc)
  sorted_locs = np.array(loc)[np.concatenate((tour, [tour[0]]))]
  return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def write_tsplib(filename, loc, name="problem"):

  with open(filename, 'w') as f:
    f.write("\n".join([
        "{} : {}".format(k, v) for k, v in (
            ("NAME", name),
            ("TYPE", "TSP"),
            ("DIMENSION", len(loc)),
            ("EDGE_WEIGHT_TYPE", "EUC_2D"),
        )
    ]))
    f.write("\n")
    f.write("NODE_COORD_SECTION\n")
    f.write("\n".join([
        "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5),
                            int(y * 10000000 +
                                0.5))  # tsplib does not take floats
        for i, (x, y) in enumerate(loc)
    ]))
    f.write("\n")
    f.write("EOF\n")


def solve_concorde(executable, directory, loc):
  """
  Args:
    executable: Path to concorde executable
    directory: Temporary directory to store the intermediate files
    loc: locations of the cities in TSP.
  """
  name = str(uuid.uuid4())
  problem_filename = os.path.join(directory, "{}.tsp".format(name))
  tour_filename = os.path.join(directory, "{}.tour".format(name))
  output_filename = os.path.join(directory, "{}.concorde.pkl".format(name))
  log_filename = os.path.join(directory, "{}.log".format(name))
  executable = os.path.abspath(executable)

  try:
    write_tsplib(problem_filename, loc, name=name)

    with open(log_filename, 'w') as f:
      start = time.time()
      try:
        # Concorde is weird, will leave traces of solution in current directory so call from target dir
        check_call([
            executable, '-s', '1234', '-x', '-o',
            os.path.abspath(tour_filename),
            os.path.abspath(problem_filename)
        ],
                   stdout=f,
                   stderr=f,
                   cwd=directory)
      except CalledProcessError as e:
        # Somehow Concorde returns 255
        assert e.returncode == 255
      duration = time.time() - start

      tour = read_concorde_tour(tour_filename)

    return calc_tsp_length(loc, tour), tour, duration

  except Exception as e:
    print("Exception occured")
    print(e)
    return None

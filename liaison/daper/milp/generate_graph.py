import argparse
import os
from itertools import combinations

import numpy as np
import scipy.sparse
from liaison.daper.milp.primitives import (BinaryVariable, ContinuousVariable,
                                           MIPInstance)


def generate_cauctions(random,
                       n_items=100,
                       n_bids=500,
                       min_value=1,
                       max_value=100,
                       value_deviation=0.5,
                       add_item_prob=0.9,
                       max_n_sub_bids=5,
                       additivity=0.2,
                       budget_factor=1.5,
                       resale_factor=0.5,
                       integers=False,
                       warnings=False):
  """
  Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
    Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
    Towards a universal test suite for combinatorial auction algorithms.
    Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

  Saves it as a CPLEX LP file.

  Parameters
  ----------
  random : numpy.random.RandomState
    A random number generator.
  n_items : int
    The number of items.
  n_bids : int
    The number of bids.
  min_value : int
    The minimum resale value for an item.
  max_value : int
    The maximum resale value for an item.
  value_deviation : int
    The deviation allowed for each bidder's private value of an item, relative from max_value.
  add_item_prob : float in [0, 1]
    The probability of adding a new item to an existing bundle.
  max_n_sub_bids : int
    The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
  additivity : float
    Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity > 0 gives super-additive bids.
  budget_factor : float
    The budget factor for each bidder, relative to their initial bid's price.
  resale_factor : float
    The resale factor for each bidder, relative to their initial bid's resale value.
  integers : logical
    Should bid's prices be integral ?
  warnings : logical
    Should warnings be printed ?
  """

  assert min_value >= 0 and max_value >= min_value
  assert add_item_prob >= 0 and add_item_prob <= 1

  def choose_next_item(bundle_mask, interests, compats, add_item_prob, random):
    n_items = len(interests)
    prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
    prob /= prob.sum()
    return random.choice(n_items, p=prob)

  # common item values (resale price)
  values = min_value + (max_value - min_value) * random.rand(n_items)

  # item compatibilities
  compats = np.triu(random.rand(n_items, n_items), k=1)
  compats = compats + compats.transpose()
  compats = compats / compats.sum(1)

  bids = []
  n_dummy_items = 0

  # create bids, one bidder at a time
  while len(bids) < n_bids:

    # bidder item values (buy price) and interests
    private_interests = random.rand(n_items)
    private_values = values + max_value * value_deviation * (
        2 * private_interests - 1)

    # substitutable bids of this bidder
    bidder_bids = {}

    # generate initial bundle, choose first item according to bidder interests
    prob = private_interests / private_interests.sum()
    item = random.choice(n_items, p=prob)
    bundle_mask = np.full(n_items, 0)
    bundle_mask[item] = 1

    # add additional items, according to bidder interests and item compatibilities
    while random.rand() < add_item_prob:
      # stop when bundle full (no item left)
      if bundle_mask.sum() == n_items:
        break
      item = choose_next_item(bundle_mask, private_interests, compats,
                              add_item_prob, random)
      bundle_mask[item] = 1

    bundle = np.nonzero(bundle_mask)[0]

    # compute bundle price with value additivity
    price = private_values[bundle].sum() + np.power(len(bundle),
                                                    1 + additivity)
    if integers:
      price = int(price)

    # drop negativaly priced bundles
    if price < 0:
      if warnings:
        print("warning: negatively priced bundle avoided")
      continue

    # bid on initial bundle
    bidder_bids[frozenset(bundle)] = price

    # generate candidates substitutable bundles
    sub_candidates = []
    for item in bundle:

      # at least one item must be shared with initial bundle
      bundle_mask = np.full(n_items, 0)
      bundle_mask[item] = 1

      # add additional items, according to bidder interests and item compatibilities
      while bundle_mask.sum() < len(bundle):
        item = choose_next_item(bundle_mask, private_interests, compats,
                                add_item_prob, random)
        bundle_mask[item] = 1

      sub_bundle = np.nonzero(bundle_mask)[0]

      # compute bundle price with value additivity
      sub_price = private_values[sub_bundle].sum() + np.power(
          len(sub_bundle), 1 + additivity)
      if integers:
        sub_price = int(sub_price)

      sub_candidates.append((sub_bundle, sub_price))

    # filter valid candidates, higher priced candidates first
    budget = budget_factor * price
    min_resale_value = resale_factor * values[bundle].sum()
    for bundle, price in [
        sub_candidates[i]
        for i in np.argsort([-price for bundle, price in sub_candidates])
    ]:

      if len(bidder_bids
             ) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
        break

      if price < 0:
        if warnings:
          print("warning: negatively priced substitutable bundle avoided")
        continue

      if price > budget:
        if warnings:
          print("warning: over priced substitutable bundle avoided")
        continue

      if values[bundle].sum() < min_resale_value:
        if warnings:
          print("warning: substitutable bundle below min resale value avoided")
        continue

      if frozenset(bundle) in bidder_bids:
        if warnings:
          print("warning: duplicated substitutable bundle avoided")
        continue

      bidder_bids[frozenset(bundle)] = price

    # add XOR constraint if needed (dummy item)
    if len(bidder_bids) > 2:
      dummy_item = [n_items + n_dummy_items]
      n_dummy_items += 1
    else:
      dummy_item = []

    # place bids
    for bundle, price in bidder_bids.items():
      bids.append((list(bundle) + dummy_item, price))

  m = MIPInstance()
  # first define the objective
  obj = m.get_objective()

  bids_per_item = [[] for _ in range(n_items + n_dummy_items)]

  for i, (bundle, price) in enumerate(bids):
    obj.add_term(f"x{i+1}", -price)  # negative sign to have minimization.
    for item in bundle:
      bids_per_item[item].append(i)

  for item_bids in bids_per_item:
    if item_bids:
      c = m.new_constraint('LE', 1)
      for i in item_bids:
        c.add_term(f"x{i+1}", 1)

  for i in range(len(bids)):
    m.add_variable(BinaryVariable(f"x{i+1}"))

  m.validate()
  return m


def generate_capacited_facility_location(rng, n_customers, n_facilities,
                                         ratio):
  """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    rng : numpy.random.RandomState
        A random number generator.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
  c_x = rng.rand(n_customers)
  c_y = rng.rand(n_customers)

  f_x = rng.rand(n_facilities)
  f_y = rng.rand(n_facilities)

  demands = rng.randint(5, 35 + 1, size=n_customers)
  capacities = rng.randint(10, 160 + 1, size=n_facilities)
  fixed_costs = rng.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
          + rng.randint(90+1, size=n_facilities)
  fixed_costs = fixed_costs.astype(int)

  total_demand = demands.sum()
  total_capacity = capacities.sum()

  # adjust capacities according to ratio
  capacities = capacities * ratio * total_demand / total_capacity
  capacities = capacities.astype(int)
  total_capacity = capacities.sum()

  # transportation costs
  trans_costs = np.sqrt(
          (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
          + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

  demands = demands.tolist()
  capacities = capacities.tolist()
  trans_costs = trans_costs.tolist()
  fixed_costs = fixed_costs.tolist()

  m = MIPInstance()
  # first define the objective
  obj = m.get_objective()
  for i in range(n_customers):
    for j in range(n_facilities):
      obj.add_term(f"x_{i+1}_{j+1}", trans_costs[i][j])

  for j in range(n_facilities):
    obj.add_term(f"y_{j+1}", fixed_costs[j])

  # now come the constraints
  for i in range(n_customers):
    c = m.new_constraint('LE', -1, name=f"demand_{i+1}")
    for j in range(n_facilities):
      c.add_term(f"x_{i+1}_{j+1}", -1)

  for j in range(n_facilities):
    c = m.new_constraint('LE', 0, name=f"capacity_{j+1}")
    for i in range(n_customers):
      c.add_term(f"x_{i+1}_{j+1}", demands[i])
    c.add_term(f"y_{j+1}", -capacities[j])

  # optional constraints for LP relaxation tightening
  c = m.new_constraint('LE', -total_demand, name='total_capaacity')
  for j in range(n_facilities):
    c.add_term(f"y_{j+1}", -capacities[j])

  for i in range(n_customers):
    for j in range(n_facilities):
      c = m.new_constraint('LE', 0, name=f"affectation_{i+1}_{j+1}")
      c.add_term(f"x_{i+1}_{j+1}", 1)
      c.add_term(f"y_{j+1}", -1)

  # now declare variables
  # x_{i+1}_{j+1} are continuous variables
  for i in range(n_customers):
    for j in range(n_facilities):
      m.add_variable(ContinuousVariable(f"x_{i+1}_{j+1}", 0, 1))

  # y_{j+1} is binary variable
  for j in range(n_facilities):
    m.add_variable(BinaryVariable(f"y_{j+1}"))

  m.validate()
  return m


def generate_instance(problem, problem_size, seed):

  rng = np.random.RandomState(seed)
  if problem == 'cauction':
    number_of_items = problem_size
    number_of_bids = 5 * problem_size
    m = generate_cauctions(rng,
                           n_items=number_of_items,
                           n_bids=number_of_bids,
                           add_item_prob=0.7)
  elif problem == 'facilities':
    m = generate_capacited_facility_location(rng,
                                             n_customers=problem_size,
                                             n_facilities=problem_size,
                                             ratio=5)
  else:
    raise Exception(f"Unknown problem type {problem}")
  return m

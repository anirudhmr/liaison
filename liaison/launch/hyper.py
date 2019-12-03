import builtins
import itertools

import numpy as np


def range(option, start, stop, step=1):
  vals = []
  for val in np.arange(start=start, stop=stop, step=step):
    vals.append(val)

  return [{option: val} for val in vals]


def discrete(option, vals):
  return [{option: val} for val in vals]


def product(*args):
  if len(args) == 0:
    return []
  elif len(args) == 1:
    return args[0]

  result = []
  for d1 in args[0]:
    for d2 in product(*(args[1:])):
      result.append(dict(**d1, **d2))

  return result


def chain(*args):
  return itertools.chain(args)


def merge_list_of_dicts(l):
  return {k: v for d in l for k, v in d.items()}


def zip(*args):
  results = []
  for d_l in builtins.zip(*args):
    results.append(merge_list_of_dicts(d_l))
  return results


def to_commandline(param):
  """param is a dict of params"""
  l = []
  for k, v in param.items():
    l.extend(['--' + k, str(v)])
  return l

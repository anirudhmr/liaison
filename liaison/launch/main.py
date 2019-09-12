"""Entry script for each component in the setup."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from argon import ArgumentParser

parser = ArgumentParser('main entry script for all spawned processes.')
# agent_config
parser.add_config_file(name='agent')

# sess_config
parser.add_config_file(name='sess')

# learner_config
parser.add_config_file(name='learner')

parser.add_argument()


def launch():
  pass


def main():
  launch()


if __name__ == "__main__":
  main()

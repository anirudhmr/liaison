"""Note this can break the server that
the connection is being tested to."""

from caraml.zmq import ZmqTimeoutError, get_remote_client, ZmqClient
from absl import app
import argparse
from spy import Server as SpyServer

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--timeout', type=int, default=2)


def main(argv):
  args = parser.parse_args(argv[1:])
  rc = get_remote_client(SpyServer, host=args.ip, port=args.port, timeout=2)
  capacity = rc.get_capacity()
  print(capacity)
  rc.get_instantaneous_profile(1)


if __name__ == '__main__':
  app.run(main)

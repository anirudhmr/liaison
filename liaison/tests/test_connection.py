"""Note this can break the server that
the connection is being tested to."""

from caraml.zmq import ZmqTimeoutError, get_remote_client, ZmqClient
from absl import app
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--timeout', type=int, default=2)


def main(argv):
  args = parser.parse_args(argv[1:])
  cli = ZmqClient(host=args.ip,
                  port=args.port,
                  timeout=args.timeout,
                  serializer='pyarrow',
                  deserializer='pyarrow')
  cli.request([0])


if __name__ == '__main__':
  app.run(main)

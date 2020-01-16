"""Note this can break the server that
the connection is being tested to."""

import argparse

from absl import app
from caraml.zmq import ZmqClient, ZmqServer, ZmqTimeoutError, get_remote_client

parser = argparse.ArgumentParser()
parser.add_argument('mode', help='server/client')
parser.add_argument('-i', '--ip')
parser.add_argument('-p', '--port', type=int, default=50000)
parser.add_argument('--timeout', type=int, default=2)
args = None
MSG = [9]


def client_fn():
  cli = ZmqClient(host=args.ip,
                  port=args.port,
                  timeout=args.timeout,
                  serializer='pyarrow',
                  deserializer='pyarrow')
  cli.request(MSG)
  print('Done')


def server_fn():
  server = ZmqServer(host='*',
                     port=args.port,
                     serializer='pyarrow',
                     deserializer='pyarrow')

  def f(msg):
    assert msg == MSG
    print('message received succesfully')

  server.start_loop(f)


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  if args.mode == 'server':
    server_fn()
  elif args.mode == 'client':
    client_fn()


if __name__ == '__main__':
  app.run(main)

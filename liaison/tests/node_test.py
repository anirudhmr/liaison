# Tests for liaison.tmux.host
import argparse
import logging

from absl.testing import absltest, parameterized
from absl import flags
from liaison.tmux.node import Node
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'test_server_ip', None,
    'Required argument. Provide a server ip to ssh into and run commands for testing'
)

flags.DEFINE_string('test_server_ssh_key_file', None, 'ssh key to use')
flags.DEFINE_string('test_server_ssh_username', 'ubuntu', 'Username for ssh')
flags.DEFINE_integer('test_server_ssh_port', 22, 'SSH Port')


class HostTestTest(parameterized.TestCase):

  def _setup_host_local(self):
    return HostTest(ip_addr='localhost', use_ssh=False)

  def _setup_host_ssh(self):
    return HostTest(ip_addr=FLAGS.test_server_ip,
                    ssh_key_file=FLAGS.test_server_ssh_key_file,
                    ssh_port=FLAGS.test_server_ssh_port,
                    ssh_username=FLAGS.test_server_ssh_username)

  @parameterized.parameters((True, ), (False, ))
  def testLocalHostTest(self, local_host):
    if local_host:
      cli = self._setup_host_local()
    else:
      cli = self._setup_host_ssh()
    res = cli.get_unavailable_ports()
    self.assertIn(22, res)
    logging.info('Got unavailable ports: %s',
                 ' '.join([str(port) for port in res]))


if __name__ == '__main__':
  flags.mark_flag_as_required('test_server_ip')
  absltest.main()

import paramiko
import os
import subprocess


class Node:

  def __init__(self,
               ip_addr,
               use_ssh=True,
               ssh_key_file=None,
               ssh_port=22,
               ssh_username=None,
               ssh_config_file_path='/home/ubuntu/.ssh/config'):
    """
      TODO:
        Handle the case of ssh into the local host.
    """
    self._ip_addr = ip_addr
    self.ssh_key_file = ssh_key_file
    self.ssh_port = ssh_port
    self._ssh_client = None
    self.ssh_username = ssh_username
    self.use_ssh = use_ssh
    self.reserved_ports = []
    # If key file is not given, check
    # if default config option is viable
    if use_ssh and self.ssh_key_file is None:
      config = paramiko.SSHConfig()
      with open(ssh_config_file_path, 'r') as f:
        config.parse(f)

      res = config.lookup(ip_addr)
      if 'identityfile' in res:
        self.ssh_key_file = res['identityfile']
      else:
        raise Exception(
            'ssh key file not provided and default not found in ssh config file at %s'
            % ssh_config_file_path)

  def _get_ssh_client(self):
    if self._ssh_client:
      return self._ssh_client

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(self._ip_addr,
                   username=self.ssh_username,
                   port=self.ssh_port,
                   key_filename=self.ssh_key_file)
    self._ssh_client = client
    return self._ssh_client

  def _ssh_run_cmd(self, ssh_client, cmd):
    _, out, err = ssh_client.exec_command(cmd)
    out = list(out)
    err = list(err)
    if err:
      raise Exception(
          'Exception encountered when executing command "{cmd}" on server {ip}:{port}: %s'
          .format(
              cmd=cmd,
              ip=self._ip_addr,
              port=self.ssh_port,
          ))
    return ''.join(out)

  def _local_run_cmd(self, cmd):
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    if result.stderr:
      raise Exception(
          'exception encountered when executing command "{cmd}" locally'.
          format(cmd=cmd))
    if result.stdout is None:
      return ''
    else:
      return result.stdout.decode('utf-8')

  def _run_cmd(self, cmd):
    if self.use_ssh:
      ssh_cli = self._get_ssh_client()
      return self._ssh_run_cmd(ssh_cli, cmd)
    else:
      return self._local_run_cmd(cmd)

  # ========== PUBLIC API ================

  @property
  def ip_addr(self):
    return self._ip_addr

  def get_unavailable_ports(self):
    # https://superuser.com/questions/529830/get-a-list-of-open-ports-in-linux
    out = self._run_cmd('ss -lnt')
    ports = list(self.reserved_ports)
    for x in out.split('\n')[1:]:  # read line by line skipping header
      l = x.split()  # split on whitespace.
      if len(l) > 3:  # Look for the 4th field.
        port = l[3].split(':')[-1]  # split :::3356 => 3356
        ports.append(int(port))
    return ports

  def get_ssh_cmd(self):
    cmd = 'ssh -o StrictHostKeyChecking=no'
    if self.ssh_port:
      cmd += ' -p %d' % self.ssh_port
    if self.ssh_key_file:
      cmd += ' -i %s' % self.ssh_key_file
    cmd += ' %s@%s' % (self.ssh_username, self._ip_addr)
    return cmd

  def reserve_port(self, port):
    # no double commitment
    assert port not in self.reserved_ports
    self.reserved_ports.append(port)

import os
import shlex
import subprocess


def local_run_cmd(cmd):
  result = subprocess.Popen(shlex.split(cmd),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

  out_lines = []
  if result.stdout is not None:
    for l in result.stdout:
      out_lines.append(l)

  result.wait()

  if result.stderr and result.returncode != 0:
    raise Exception(
        'exception encountered when executing command "{cmd}" locally: {err}'.
        format(cmd=cmd,
               err=''.join(map(lambda k: k.decode('utf-8'), result.stderr))))

  if result.stdout is None:
    return ''
  else:
    return ''.join(map(lambda k: k.decode('utf-8'), out_lines))


def get_public_ip():
  """Returns the public ip of the local machine."""
  return local_run_cmd('curl ifconfig.me').rstrip('\n')

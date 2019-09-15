import xmanager
from caraml.zmq import ZmqClient, ZmqTimeoutError


class XManagerClient:

  def __init__(self, *, host=None, port=None, timeout=5):
    self._client = ZmqClient(
        host=host,
        port=port,
        timeout=timeout,
        serializer='pyarrow',
        deserializer='pyarrow',
    )

  def register(self, *, name, host_name, results_folder, n_work_units,
               network):
    req = dict(request_type='register',
               request_args=[],
               request_kwargs=dict(name=name,
                                   host_name=host_name,
                                   results_folder=results_folder,
                                   n_work_units=n_work_units,
                                   network=network))
    rep = self._client.request(req)
    assert isinstance(rep, int)
    return rep

  def record_commands(self, exp_id, commands):
    """exp_id -> int, commands -> dict."""
    req = dict(request_type='record_commands',
               request_args=[],
               request_kwargs=dict(exp_id=exp_id, commands=commands))

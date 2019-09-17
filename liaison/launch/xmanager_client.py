import xmanager
import caraml.zmq
from caraml.zmq import set_password, ZmqClient, ZmqTimeoutError

caraml.zmq.set_password(
    "2/'{jl4<452(#>?$*%&(5KBL#GHHHJFIHoiGHrhgbrghbrgq^#4*^#$&#23983745~`")


class XManagerClient:

  def __init__(self, *, host=None, port=None, timeout=5):
    self._client = ZmqClient(
        host=host,
        port=port,
        timeout=timeout,
        serializer='pyarrow',
        deserializer='pyarrow',
        auth=True,
    )

  def register(self,
               *,
               name,
               host_name,
               results_folder,
               n_work_units,
               network,
               dry_run=False):
    req = dict(request_type='register',
               args=[],
               kwargs=dict(name=name,
                           host_name=host_name,
                           results_folder=results_folder,
                           n_work_units=n_work_units,
                           network=network,
                           dry_run=dry_run))
    rep = self._client.request(req)
    assert isinstance(rep, int)
    return rep

  def record_commands(self, exp_id, commands, dry_run=False):
    """exp_id -> int, commands -> dict."""
    req = dict(request_type='record_commands',
               args=[],
               kwargs=dict(exp_id=exp_id, commands=commands, dry_run=dry_run))
    rep = self._client.request(req)
    return rep

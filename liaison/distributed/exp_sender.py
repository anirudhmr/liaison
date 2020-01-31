"""
Agent side.
Send experience chunks (buffered) to Replay node.
"""
import liaison.utils as U
import pyarrow as pa
from caraml.zmq import ZmqSender
from liaison.session import PeriodicTracker

from .exp_serializer import get_deserializer, get_serializer


class ExpBuffer(object):
  """
        Temporarily holds and deduplicates experience
    """

  def __init__(self, serialize_fn):
    self.exp_list = []  # list of exp dicts
    self.ob_storage = {}
    self._serialize_fn = serialize_fn

  def add(self, hash_dict, nonhash_dict):
    """
        Args:
            hash_dict: {obs_hash: [ .. can be nested .. ]}
            nonhash_dict: {reward: -1.2, done: True, ...}
        """
    U.assert_type(hash_dict, dict)
    U.assert_type(nonhash_dict, dict)
    exp = {}
    for key, values in hash_dict.items():
      assert not key.endswith('_hash'), 'do not manually append `_hash`'
      exp[key + '_hash'] = self._hash_nested(values)
    exp.update(nonhash_dict)
    self.exp_list.append(exp)

  def flush(self):
    """
        Serialized all currenct content of the buffer into binary
        Also reset the buffer.

        Returns:
            binary data of (exp_list, ob_storage)
        """
    binary = (self.exp_list, self.ob_storage)
    self.exp_list = []
    self.ob_storage = {}
    return binary

  def _hash_nested(self, values):
    if isinstance(values, list):
      return [self._hash_nested(v) for v in values]
    if isinstance(values, tuple):
      return tuple([self._hash_nested(v) for v in values])
    elif isinstance(values, dict):
      return {k: self._hash_nested(v) for k, v in values.items()}
    elif values is None:
      return None
    else:  # values is a single object
      obj = values
      hsh = U.binary_hash(self._serialize_fn(obj))
      if hsh not in self.ob_storage:
        self.ob_storage[hsh] = obj
      return hsh  # returns string here


class ExpSender(object):
  """
  `send()` logic can be overwritten to support
  more complicated agent experiences,
  such as multiagent, self-play, etc.
  """

  def __init__(self,
               *,
               host,
               port,
               flush_iteration,
               compress_before_send,
               manual_flush=False):
    """
        Args:
            flush_iteration: how many send() calls before we flush the buffer
        """
    U.assert_type(flush_iteration, int)
    self._client = ZmqSender(
        host=host,
        port=port,
        serializer=get_serializer(compress_before_send),
        deserializer=get_deserializer(compress_before_send))
    self._exp_buffer = ExpBuffer(get_serializer(compress_before_send))
    if not manual_flush:
      self._flush_tracker = PeriodicTracker(flush_iteration)
    self._manual_flush = manual_flush

  def send(self, hash_dict, nonhash_dict=None):
    """
      Args:
          hash_dict: Large/Heavy data that should be deduplicated
                     by the caching mekanism
          nonhash_dict: Small data that we can afford to keep copies of
    """
    if nonhash_dict is None:
      nonhash_dict = {}

    self._exp_buffer.add(
        hash_dict=hash_dict,
        nonhash_dict=nonhash_dict,
    )
    if not self._manual_flush:
      if self._flush_tracker.track_increment():
        self.flush()

  def flush(self):
    exp_binary = self._exp_buffer.flush()
    self._client.send(exp_binary)

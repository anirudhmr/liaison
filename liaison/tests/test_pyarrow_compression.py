"""
  Maintain a seperate serializer class for experience to handle custom types.
  Add custom types to the dicts below.

  Use compression to shrink the experiences.
"""
import numpy as np
import pyarrow as pa
from absl.testing import absltest

# loss-less compression scheme
CODEC = 'lz4'


def get_serializer(compress=True):

  def f(val):
    # first serialize the data
    buf = pa.serialize(val).to_buffer()
    if compress:
      original_len = len(buf)

      # compress the data
      buf = pa.compress(buf, codec=CODEC, asbytes=True)

      # add metadata required for decompression
      return pa.serialize((original_len, CODEC, buf)).to_buffer()
    else:
      return buf

  return f


def get_deserializer(compress=True):
  """If compression is used at the serializer end."""

  def f(buf):
    if compress:
      # first deserialize the compressed data
      l, codec, buf = pa.deserialize(buf)

      # extract the data
      buf = pa.decompress(buf, l, codec=codec)

    # deserialize the actual data
    return pa.deserialize(buf)

  return f


class TestCompression(absltest.TestCase):

  def test_compress_crash(self):
    serializer = get_serializer()
    deserializer = get_deserializer()
    for _ in range(100):
      buf = serializer(np.random.rand(1000, 1000))
      deserializer(buf)


if __name__ == '__main__':
  absltest.main()

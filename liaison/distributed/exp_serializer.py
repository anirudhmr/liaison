"""
  Maintain a seperate serializer class for experience to handle custom types.
  Add custom types to the dicts below.

  Use compression to shrink the experiences.
"""
import pyarrow as pa
import graph_nets as gn

# Add custom types here as required.
CUSTOM_TYPES = {'GraphsTuple': gn.graphs.GraphsTuple}
SERIALIZERS = {'GraphsTuple': gn.utils_np.graphs_tuple_to_data_dicts}
DESERIALIZERS = {'GraphsTuple': gn.utils_np.data_dicts_to_graphs_tuple}

# same keys should be declared in all the above dicts
for d1, d2 in [(CUSTOM_TYPES, SERIALIZERS), (CUSTOM_TYPES, DESERIALIZERS)]:
  assert sorted(d1.keys()) == sorted(d2.keys())


def _get_pa_context():
  # Add custom types to context for serializing, deserializing using pyarrow.
  ctxt = pa.SerializationContext()
  for k, v in CUSTOM_TYPES.items():
    ctxt.register_type(v,
                       k,
                       custom_serializer=SERIALIZERS[k],
                       custom_deserializer=DESERIALIZERS[k])
  return ctxt


# loss-less compression scheme
CODEC = 'lz4'


def get_serializer(compress=True):
  context = _get_pa_context()

  def f(val):
    # first serialize the data
    buf = pa.serialize(val, context=context).to_buffer()
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
  context = _get_pa_context()

  def f(buf):
    if compress:
      # first deserialize the compressed data
      l, codec, buf = pa.deserialize(buf)

      # extract the data
      buf = pa.decompress(buf, l, codec=codec)

    # deserialize the actual data
    return pa.deserialize(buf, context=context)

  return f

"""
  Used to aggregate the graph features without having to pad.
  Abandoned for the time-being.
  Get back when the performance problems become more pressing due to padding.
"""

class BatchedEnv:

  def __init__(self, base_class):

    class BatchedEnv(base_class):

      def _stack_specs(self, specs):
        def f()
        return nest.map_structure(f, self._step_spec, *specs)





  def __getattr__(self, attrname):
    """
        Delegate any unknown methods to the underlying self.socket
        """
    if attrname in dir(self):
      return object.__getattribute__(self, attrname)
    else:
      return getattr(self._socket, attrname)


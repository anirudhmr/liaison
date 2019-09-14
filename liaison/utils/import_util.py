"""Utils to import function or class by its path or module name."""
import importlib
import os


def import_by_file_path(obj_to_import, file_path):
  """Import obj_to_import from file path.
  Example:
    import_by_file_path('Shell', 'liaison/distributed/shell.py')
  """
  spec = importlib.util.spec_from_file_location(obj_to_import, file_path)
  foo = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(foo)
  return foo


def import_by_module_name(obj_to_import, module_name):
  """Import obj from module name.

  Ex:
    import_by_module_name('Shell', 'liaison.distributed')
  """
  module = importlib.import_module(module_name)
  if obj_to_import in module.__dict__:
    return module.__dict__[obj_to_import]
  else:
    raise ValueError('%s not found in the module %s' %
                     (obj_to_import, module_name))


def import_obj(obj_to_import, module_name_or_file_path):
  """Combines import_by_file_path and import_by_module_name based on
  passed argument."""

  if os.path.exists(module_name_or_file_path):
    return import_by_file_path(obj_to_import, module_name_or_file_path)
  else:
    return import_by_module_name(obj_to_import, module_name_or_file_path)

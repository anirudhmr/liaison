from liaison.utils import ConfigDict
"""
Must have attribute class_path attribute.
"""


def get_config():
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.agents.ur_discrete"  # should be rel to the parent directory.
  config.class_name = "Agent"

  return config

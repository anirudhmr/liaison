"""Bundle together shell config and actor config in one file here. """

from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.env.xor_env"  # should be rel to the parent directory.
  config.class_name = "Env"

  # makes observations suitable for the MLP model.
  config.make_obs_for_mlp = False

  return config

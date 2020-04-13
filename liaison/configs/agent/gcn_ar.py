from liaison.configs.agent.config import get_config as get_base_config
from liaison.utils import ConfigDict


def get_config():
  config = get_base_config()

  # required fields.
  config.class_path = "liaison.agents.gcn_multi_actions"
  config.class_name = "Agent"

  config.model = ConfigDict()
  config.model.class_path = 'liaison.agents.models.transformer_auto_regressive'
  config.model.num_blocks = 4
  config.model.d_ff = 32
  config.model.num_heads = 4
  config.model.d_model = 64
  config.model.dropout_rate = 0.
  config.model.use_mlp_value_func = False

  # The following code duplicated in gcn_rins.py as well.
  # Propagate any changes made as needed.
  config.model.model_kwargs = ConfigDict()
  config.model.model_kwargs.class_path = "liaison.agents.models.bipartite_gcn_rins"
  config.model.model_kwargs.n_prop_layers = 4
  config.model.model_kwargs.edge_embed_dim = 32
  config.model.model_kwargs.node_embed_dim = 32
  config.model.model_kwargs.global_embed_dim = 32
  config.model.model_kwargs.policy_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.value_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.policy_summarize_hidden_layer_sizes = [16]
  config.model.model_kwargs.value_summarize_hidden_layer_sizes = [16]
  config.model.model_kwargs.supervised_prediction_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.sum_aggregation = False
  config.model.model_kwargs.use_layer_norm = True

  config.clip_rho_threshold = 1.0
  config.clip_pg_rho_threshold = 1.0

  config.loss = ConfigDict()
  config.loss.vf_loss_coeff = 1.0

  config.loss.al_coeff = ConfigDict()
  config.loss.al_coeff.init_val = 0.
  config.loss.al_coeff.min_val = 0.
  config.loss.al_coeff.start_decay_step = int(1e10)
  config.loss.al_coeff.decay_steps = 5000
  # dec_val not used for linear scheme
  config.loss.al_coeff.dec_val = .1
  config.loss.al_coeff.dec_approach = 'linear'

  # applicable for agent 'liaison.agents.gcn_large_batch'
  config.apply_grads_every = 1

  config.log_features_every = -1  # disable
  config.freeze_graphnet_weights_step = int(1e9)

  return config

export KMP_AFFINITY=none
N=31
N_LOCAL_MOVES=20

MAX_NODES=50
DATASET=milp-cauction-100-filtered
# RESTORE_FROM=/data/nms/tfp/results/1419/100-dataset-no-ar/checkpoints/0/246000/learner-246000
RESTORE_FROM=/home/gridsan/addanki/results/1446/100-dataset-no-ar/checkpoints/3/163000/learner-163000
NAME=gcn_100
K=5
EXTRA_ARGS=""

# DATASET='milp-facilities-100'
# RESTORE_FROM='/home/gridsan/addanki/results/1433/milp-facilities-100-2/checkpoints/0/98000/learner-98000'
# NAME=facilities_100
# K=5
# EXTRA_ARGS="--agent_config.model.n_prop_layers=4 --agent_config.model.edge_embed_dim=16 --agent_config.model.node_embed_dim=16 --agent_config.model.global_embed_dim=8"

# DATASET='milp-corlat'
# RESTORE_FROM='/home/gridsan/addanki/results/1462/corlat/checkpoints/2/10000/learner-10000'
# NAME=corlat
# K=50
# MAX_NODES=20
# EXTRA_ARGS="--heur_frequency=10000 --agent_config.model.n_prop_layers=4 --agent_config.model.choose_stop_switch=True --env_config.primal_gap_reward=False --env_config.primal_gap_reward_with_work=True --env_config.adapt_k.min_k=0"

# DATASET='milp-corlat'
# RESTORE_FROM='/home/gridsan/addanki/results/1479/corlat-dataset-k-25/checkpoints/0/164000/learner-164000'
# NAME=corlat-k-25
# K=25
# MAX_NODES=50
# EXTRA_ARGS="--heur_frequency=10000 --agent_config.model.n_prop_layers=4 --agent_config.choose_stop_switch=False --env_config.primal_gap_reward=True --env_config.primal_gap_reward_with_work=False --env_config.adapt_k.min_k=0"

for i in `seq 0 $N`; do
  python liaison/scip/run_graph.py -- \
  -n $NAME \
  --max_nodes=${MAX_NODES} \
  --use_parallel_envs \
  --n_local_moves=${N_LOCAL_MOVES} \
  --batch_size=8 \
  --agent_config_file=liaison/configs/agent/gcn_rins.py \
  --sess_config_file=liaison/configs/session_config.py \
  --env_config_file=liaison/configs/env/rins.py \
  --agent_config.model.class_path='liaison.agents.models.bipartite_gcn_rins' \
  --sess_config.shell.restore_from=${RESTORE_FROM} \
  \
  --env_config.class_path=liaison.env.rins_v2 \
  --env_config.make_obs_for_bipartite_graphnet=True \
  --env_config.muldi_actions=False \
  --env_config.dataset=${DATASET} \
  --env_config.dataset_type=test \
  --env_config.n_graphs=1 \
  --env_config.k=$K \
  --gpu_ids=`expr $i % 2` \
  --env_config.graph_start_idx=$i \
  ${EXTRA_ARGS} &
done
wait
sudo killall run_graph.py

for i in `seq 0 $N`; do
  # without agent.
  python liaison/scip/run_graph.py -- \
  -n $NAME \
  --max_nodes=${MAX_NODES} \
  --without_agent \
  --use_parallel_envs \
  --n_local_moves=${N_LOCAL_MOVES} \
  --batch_size=8 \
  --agent_config_file=liaison/configs/agent/gcn_rins.py \
  --sess_config_file=liaison/configs/session_config.py \
  --env_config_file=liaison/configs/env/rins.py \
  \
  --agent_config.model.class_path='liaison.agents.models.bipartite_gcn_rins' \
  --sess_config.shell.restore_from=${RESTORE_FROM} \
  \
  --env_config.class_path=liaison.env.rins_v2 \
  --env_config.make_obs_for_bipartite_graphnet=True \
  --env_config.muldi_actions=False \
  --env_config.dataset=${DATASET} \
  --env_config.dataset_type=test \
  --env_config.n_graphs=1 \
  --env_config.k=$K \
  --env_config.graph_start_idx=$i \
  ${EXTRA_ARGS} &
done
wait
sudo killall run_graph.py

for i in `seq 0 $N`; do
  # rins
  python liaison/scip/run_graph.py -- \
    -n $NAME \
    --max_nodes=${MAX_NODES} \
    --gap=.05 \
    --heuristic=rins \
    --use_parallel_envs \
    --n_local_moves=${N_LOCAL_MOVES} \
    --batch_size=8 \
    --agent_config_file=liaison/configs/agent/gcn_rins.py \
    --sess_config_file=liaison/configs/session_config.py \
    --env_config_file=liaison/configs/env/rins.py \
    \
    --agent_config.model.class_path='liaison.agents.models.bipartite_gcn_rins' \
    --sess_config.shell.restore_from=${RESTORE_FROM} \
    \
    --env_config.class_path=liaison.env.rins_v2 \
    --env_config.make_obs_for_bipartite_graphnet=True \
    --env_config.muldi_actions=False \
    --env_config.dataset=${DATASET} \
    --env_config.dataset_type=test \
    --env_config.n_graphs=1 \
    --env_config.k=$K \
    --env_config.graph_start_idx=$i \
    ${EXTRA_ARGS} &
done
wait
sudo killall run_graph.py

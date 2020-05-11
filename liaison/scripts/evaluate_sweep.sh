export KMP_AFFINITY=none
N=16
N_LOCAL_MOVES=20
MAX_NODES=20
DATASET=milp-cauction-100-filtered
RESTORE_FROM=/data/nms/tfp/results/1419/100-dataset-no-ar/checkpoints/0/246000/learner-246000
NAME=gcn_100
K=5
EXTRA_ARGS=""

# DATASET='milp-facilities-100'
# RESTORE_FROM='/home/gridsan/addanki/results/1433/milp-facilities-100-2/checkpoints/0/98000/learner-98000'
# NAME=facilities_100
# K=5
# EXTRA_ARGS="--agent_config.model.n_prop_layers=4 --agent_config.model.edge_embed_dim=16 --agent_config.model.node_embed_dim=16 --agent_config.model.global_embed_dim=8"

# DATASET='milp-corlat'
# RESTORE_FROM='/home/gridsan/addanki/results/1429/corlat/checkpoints/0/90000/learner-90000'
# NAME=corlat
# K=5
# MAX_NODES=10000
# EXTRA_ARGS="--heur_frequency=10000 --agent_config.model.n_prop_layers=8"

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

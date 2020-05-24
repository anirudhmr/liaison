export KMP_AFFINITY=none
N_LOCAL_MOVES=25
DATASET='milp-corlat'
K=50
MAX_NODES=20
EXTRA_ARGS="--heur_frequency=10000 --agent_config.model.n_prop_layers=4 --agent_config.choose_stop_switch=True --env_config.primal_gap_reward=True --env_config.primal_gap_reward_with_work=False --env_config.adapt_k.min_k=0"

RESTORE_FROM=('/home/gridsan/addanki/results/1464/corlat-single-graph/checkpoints/0/122000/learner-122000' '/home/gridsan/addanki/results/1464/corlat-single-graph/checkpoints/3/244000/learner-244000' '/home/gridsan/addanki/results/1464/corlat-single-graph/checkpoints/4/132000/learner-132000' '/home/gridsan/addanki/results/1464/corlat-single-graph/checkpoints/6/210000/learner-210000')
GRAPH=(0 1 2 3)

for (( i=0; i<${#RESTORE_FROM[*]}; ++i )); do
  NAME=corlat-single-graph--adapt-$i
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
  --sess_config.shell.restore_from=${RESTORE_FROM[$i]} \
  \
  --env_config.class_path=liaison.env.rins_v2 \
  --env_config.make_obs_for_bipartite_graphnet=True \
  --env_config.muldi_actions=False \
  --env_config.dataset=${DATASET} \
  --env_config.dataset_type=test \
  --env_config.n_graphs=1 \
  --env_config.k=$K \
  --gpu_ids=`expr $i % 2` \
  --env_config.graph_start_idx=${GRAPH[$i]} \
  ${EXTRA_ARGS} &

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
  --sess_config.shell.restore_from=${RESTORE_FROM[$i]} \
  \
  --env_config.class_path=liaison.env.rins_v2 \
  --env_config.make_obs_for_bipartite_graphnet=True \
  --env_config.muldi_actions=False \
  --env_config.dataset=${DATASET} \
  --env_config.dataset_type=test \
  --env_config.n_graphs=1 \
  --env_config.k=$K \
  --env_config.graph_start_idx=${GRAPH[$i]} \
  ${EXTRA_ARGS} &

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
    --sess_config.shell.restore_from=${RESTORE_FROM[$i]} \
    \
    --env_config.class_path=liaison.env.rins_v2 \
    --env_config.make_obs_for_bipartite_graphnet=True \
    --env_config.muldi_actions=False \
    --env_config.dataset=${DATASET} \
    --env_config.dataset_type=test \
    --env_config.n_graphs=1 \
    --env_config.k=$K \
    --env_config.graph_start_idx=${GRAPH[$i]} \
    ${EXTRA_ARGS} &
done
wait
sudo killall run_graph.py

python liaison/daper/milp/heuristics/sweep.py -s --run_random_only -d milp-corlat --n_training_samples=128 --n_valid_samples=0 --n_test_samples=0 --out_dir=/data/nms/tfp/heuristics/milp-corlat/ --n_local_moves=20


python liaison/daper/milp/heuristics/sweep.py -s --run_random_only -d milp-cauction-100 --n_training_samples=128 --n_valid_samples=0 --n_test_samples=0 --out_dir=/data/nms/tfp/heuristics/milp-corlat/ --n_local_moves=20

python liaison/daper/milp/heuristics/sweep.py -s --run_random_only -d milp-cauction-300-filtered --n_training_samples=64 --n_valid_samples=0 --n_test_samples=0 --out_dir=/data/nms/tfp/heuristics/milp-cauction-300-filtered/ --n_local_moves=50

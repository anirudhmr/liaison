for dataset in cauction corlat facilities indset setcover; do
  srun --exclusive bash liaison/scripts/evaluate_standalone.sh $dataset &
  # srun --exclusive bash liaison/scripts/evaluate_sweep2.sh $dataset &
done

wait

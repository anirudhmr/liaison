for dataset in cauction corlat facilities indset setcover; do
  # srun --exclusive bash liaison/scripts/evaluate_standalone.sh $dataset &
  srun --exclusive bash liaison/scripts/evaluate_sweep2.sh $dataset &
done; wait

# for dataset in cauction_transfer_k cauction_transfer_to_300; do
#   srun --exclusive bash liaison/scripts/evaluate_standalone.sh $dataset &
# done; wait

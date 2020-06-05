datasets=( "milp-cauction-100-filtered" "milp-facilities-100" "milp-corlat" "milp-setcover-100-filtered" "milp-indset-100-filtered" )
names=( "cauction_100" "facilities_100" "corlat" "setcover" "indset" )

for ((i=0;i<${#datasets[@]};++i)); do
  python liaison/scripts/plts/plt_scip.py -- \
    --scip \
    -r /home/arc/vol/mnt/nms/tfp/evaluation/ -o /home/arc/vol/mnt/nms/tfp/paper/figs/ \
    -d "${datasets[$i]}" \
    -n "${names[$i]}" $@ &

  python liaison/scripts/plts/plt_scip.py -- \
    -r /home/arc/vol/mnt/nms/tfp/evaluation/ -o /home/arc/vol/mnt/nms/tfp/paper/figs/ \
    -d "${datasets[$i]}" \
    -n "${names[$i]}" $@ &
done; wait

# datasets=( "milp-cauction-100-filtered" "milp-facilities-100" "milp-corlat" "milp-setcover-100-filtered" "milp-indset-100-filtered" "milp-cauction-100-filtered" "milp-cauction-300-filtered")
# names=( "cauction_100" "facilities_100" "corlat" "setcover" "indset" "cauction_transfer_k" "cauction_transfer_to_300")

# for ((i=0;i<${#datasets[@]};++i)); do
#   python liaison/scripts/plts/plt_scip.py -- \
#     -r /home/arc/vol/mnt/nms/tfp/evaluation/ -o /home/arc/vol/mnt/nms/tfp/paper/figs/ \
#     -d "${datasets[$i]}" \
#     -n "${names[$i]}" $@ &
# done; wait

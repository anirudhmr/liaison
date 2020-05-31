datasets=( "milp-cauction-100-filtered" "milp-facilities-100" "milp-corlat" "milp-setcover-100-filtered" "milp-indset-100-filtered" )

for ((i=0;i<${#datasets[@]};++i)); do
  python liaison/daper/milp/extract_simple_fields.py \
    -d "${datasets[$i]}" \
    -o /home/arc/vol/mnt/nms/tfp/datasets_simple/${datasets[$i]} &
done; wait

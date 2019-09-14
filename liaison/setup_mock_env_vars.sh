# use `source setup_mock_env_vars.sh` to execute this script. 
l=(
    'ps-frontend'
    'ps-backend'
    'collector-frontend'
    'collector-backend'
    'sampler-frontend'
    'sampler-backend'
    'parameter-publish'
    'prefetch-queue'
    'tensorplex'
    'loggerplex'
)

port=6000
for s in "${l[@]}"; do
  # toupper(s)
  s=`printf '%s\n' "$s" | awk '{print toupper($0) }'`

  # replace - with _
  s=`echo $s | sed 's/-/_/g'`

  echo "SYMPH_${s}_PORT $port"

  export "SYMPH_${s}_PORT"=$port
  export "SYMPH_${s}_HOST"=localhost
  (( port = port + 1 ))
done

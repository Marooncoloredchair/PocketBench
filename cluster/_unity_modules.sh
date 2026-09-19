#!/bin/bash
set -euo pipefail
KEY="${HOME}/.ssh/unity_key"
HOST="terrell_osborne_uri_edu@login.unityhpc.org"
ssh -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=40 -i "$KEY" "$HOST" 'bash -l -s' <<'REMOTE'
set +e
# login shell should init lmod
echo "MODULEPATH=$MODULEPATH"
module avail cuda 2>&1 | head -30
module avail conda 2>&1 | head -20
module avail miniforge 2>&1 | head -15
module avail python 2>&1 | head -20
module overview cuda 2>&1 | head -20
echo "=== qos/account ==="
sacctmgr show associations user=$USER format=Account,Partition,QOS%-40 2>&1 | head -20
echo "=== home mods.txt ==="
head -50 ~/mods.txt 2>/dev/null
REMOTE

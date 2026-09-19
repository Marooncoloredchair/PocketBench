#!/bin/bash
set -euo pipefail
KEY="${HOME}/.ssh/unity_key"
HOST="terrell_osborne_uri_edu@login.unityhpc.org"
SSH=(ssh -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=40 -i "$KEY" "$HOST")

"${SSH[@]}" 'bash -s' <<'REMOTE'
set +e
echo "=== whoami/host ==="
whoami; hostname
echo "=== home ==="
ls -la "$HOME" | head -25
echo "=== work pi ==="
ls -la /work/pi_nzawia_uri_edu/ | head -40
echo "=== tools ==="
command -v sbatch; command -v conda; command -v git; command -v python3; command -v vina
echo "=== modules ==="
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module avail cuda 2>&1 | head -25
module avail miniconda 2>&1 | head -15
module avail anaconda 2>&1 | head -15
echo "=== sinfo ==="
sinfo -p uri-gpu 2>&1 | head -8
echo "=== search ==="
find /work/pi_nzawia_uri_edu -maxdepth 3 \( -iname '*diffsbdd*' -o -iname '*vina*' -o -iname '*pocket*' -o -iname '*PocketBench*' \) 2>/dev/null | head -40
ls /work/pi_nzawia_uri_edu/bndf-envs 2>/dev/null | head -20
df -h /work/pi_nzawia_uri_edu 2>/dev/null | head -5
REMOTE

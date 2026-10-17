#!/usr/bin/env bash
# Launch one mission detached on this machine (the rig):  launch_mission.sh <recipe> <mission_id> [minutes] [budget] [workers]
set -euo pipefail
ROOT=${FORGE_ROOT:-$HOME/forge}
PY=${FORGE_PY:-$HOME/venvs/vllm-bolt/bin/python}
RECIPE=$1; MID=$2; MIN=${3:-45}; BUDGET=${4:-6}; WORKERS=${5:-2}
cd "$ROOT"
mkdir -p state/specs
export FORGE_ROOT=$ROOT PYTHONPATH=$ROOT
$PY -m forge.agents mission --recipe "$RECIPE" --machine rig --minutes "$MIN" --budget "$BUDGET" \
    --workers "$WORKERS" --id "$MID" --out "state/specs/$MID.json"
$PY -c "import json,sys; p=sys.argv[1]; s=json.load(open(p)); s['gate_grace_min']=float(sys.argv[2]); json.dump(s,open(p,'w'),indent=1)" \
    "state/specs/$MID.json" "${GATE_GRACE_MIN:-40}"
setsid nohup $PY -u -m forge.agents run --spec "state/specs/$MID.json" > "state/$MID.out" 2>&1 < /dev/null &
echo "launched $MID pid $!"

#!/usr/bin/env bash
# Deploy forge/dash to the rig and (re)start it on :8790.   usage: forge/dash/deploy.sh [user@host]
# Copies only forge/dash (tar over ssh, no git on the rig) and restarts only the dash process it started.
set -euo pipefail
HOST="${1:-cube@100.68.107.11}"
cd "$(dirname "$0")/../.."
tar --exclude=__pycache__ -cf - forge/__init__.py forge/dash | ssh "$HOST" 'mkdir -p ~/forge && tar -xf - -C ~/forge'
ssh "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
cd ~/forge
PIDF=~/forge/state/dash/dash.pid
mkdir -p ~/forge/state/dash
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null \
   && grep -q "forge.dash" "/proc/$(cat "$PIDF")/cmdline" 2>/dev/null; then
  kill "$(cat "$PIDF")"; sleep 1
fi
if ss -ltn | grep -q ':8790 '; then echo "port 8790 is held by something else; not starting" >&2; exit 1; fi
FORGE_ROOT=~/forge nohup setsid ~/venvs/vllm-bolt/bin/python -m forge.dash --root ~/forge --port 8790 \
  > ~/forge/state/dash/dash.log 2>&1 < /dev/null &
echo $! > "$PIDF"
for i in $(seq 1 30); do
  if curl -fsS -m 2 http://127.0.0.1:8790/api/health >/dev/null 2>&1; then echo "dash up: pid $(cat "$PIDF")"; exit 0; fi
  sleep 0.5
done
echo "dash did not come up:" >&2; tail -20 ~/forge/state/dash/dash.log >&2; exit 1
REMOTE

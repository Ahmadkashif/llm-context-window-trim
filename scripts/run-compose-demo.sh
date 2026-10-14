#!/usr/bin/env bash
# Runs the before/after demo and the overhead bench through the compose stack
# (Caddy edge -> ctxtrim -> 2 fake API replicas; portal-style shim) and writes evidence/06-compose-demo.txt.
# Needs: Docker with Compose v2, and a prior scripts/run-evidence.sh (for .venv, bin/ and the tokenizer).
#   scripts/run-compose-demo.sh          up, test, leave running
#   scripts/run-compose-demo.sh --down   also stop and remove the demo stack afterwards
set -euo pipefail
cd "$(dirname "$0")/.."
DC=(docker compose -p ctxtrim-poc -f deploy/docker-compose.yml)
"${DC[@]}" up -d --build >/dev/null 2>&1
for _ in $(seq 50); do curl -sf http://127.0.0.1:18480/v1/models >/dev/null && break; sleep 0.2; done
{
  echo "# compose demo: edge (Caddy) -> ctxtrim -> api-1/api-2 (fakellm, 8192); portal-shim -> api-1"
  date -u +%FT%TZ; echo "commit: $(git rev-parse HEAD)"
  "${DC[@]}" ps --format '{{.Service}} {{.Image}} {{.State}} {{.Ports}}'
  echo
  .venv/bin/python -I demo/overflow_demo.py --before http://127.0.0.1:18480/bypass --api http://127.0.0.1:18480 \
    --portal http://127.0.0.1:18482 --turns 40 --tokenizer .cache/qwen3-tokenizer.json | grep -E '400|at the end|RESULT|turn 40'
  echo
  echo "# bench through the edge: Caddy -> ctxtrim -> 2 replicas, vs the demo-only bypass route"
  for c in 4 8 16; do
    bin/bench -url http://127.0.0.1:18480/bypass -c $c -n 20 -history-words 3000 -label "edge-bypass  hist=3000w"
    bin/bench -url http://127.0.0.1:18480 -c $c -n 20 -history-words 3000 -label "edge+ctxtrim hist=3000w"
  done
  echo
  echo "# hardening of the running containers"
  for c in $("${DC[@]}" ps -q); do
    docker inspect -f '{{.Name}} user={{.Config.User}} read_only={{.HostConfig.ReadonlyRootfs}} cap_drop={{.HostConfig.CapDrop}} cap_add={{.HostConfig.CapAdd}}' "$c"
  done
  echo
  echo "# last ctxtrim trim log lines"
  "${DC[@]}" logs --no-log-prefix ctxtrim | grep 'context trimmed' | tail -3
} > evidence/06-compose-demo.txt 2>&1
cat evidence/06-compose-demo.txt
[[ "${1:-}" == --down ]] && "${DC[@]}" down --remove-orphans >/dev/null 2>&1 && echo "demo stack removed"
exit 0

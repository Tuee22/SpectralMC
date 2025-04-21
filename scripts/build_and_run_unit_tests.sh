#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0.  Resolve paths
###############################################################################
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

###############################################################################
# 1.  Verify the host can launch NVIDIA‑enabled containers
###############################################################################
"$SCRIPT_DIR/verify_docker_cuda.sh"

###############################################################################
# 2.  Build (if needed) and start the docker‑compose stack
###############################################################################
cd "$REPO_ROOT/docker"
docker compose up -d

###############################################################################
# 3.  Run the test suite in the running container
###############################################################################
docker compose exec spectralmc bash -c '
  set -euo pipefail
  cd /spectralmc/python            # <- repo root inside the image (adapt if different)
  echo "🧪  Running unit tests …"
  pytest -s                # spectralmc is already on PYTHONPATH
'
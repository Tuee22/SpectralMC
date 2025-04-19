#!/usr/bin/env bash
set -e

# Get the absolute path to the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the CUDA check
"$SCRIPT_DIR/verify_docker_cuda.sh"

# Move to the docker directory relative to the script location
cd "$SCRIPT_DIR/../docker"

# Bring up container and run tests
docker compose up -d # will automatically build the image if it doesn't exist
docker compose exec spectralmc pytest -s --pyargs spectralmc.tests
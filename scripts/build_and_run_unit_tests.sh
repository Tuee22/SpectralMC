!/usr/bin/env bash
set -e

./verify_docker_cuda.sh
cd ../docker
docker compose up -d --build
docker compose exec -it spectralmc pytest -s spectralmc/tests.py
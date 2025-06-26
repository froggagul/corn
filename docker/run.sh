#!/usr/bin/env bash

set -exu

# CONFIGURE DATA PATHS
IG_PATH="/home/rogga/research/efficient_planning/isaacgym"
CACHE_PATH="/home/${USER}/.cache/pkm"
DATA_PATH="/home/rogga/research/efficient_planning/corn_dataset"

# Figure out repository root.
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Create a temporary directory to be shared between host<->docker.
mkdir -p "${CACHE_PATH}"
mkdir -p '/tmp/docker/'

# Launch docker with the following configuration:
# * Display/Gui connected
# * Network enabled (passthrough to host)
# * Privileged
# * GPU devices visible
# * Current working git repository mounted at ${HOME}
# * 8Gb Shared Memory
# NOTE: comment out `--network host` for profiling with `nsys-ui`.

xhost +local:docker


docker run -it -d \
    --mount type=bind,source="${REPO_ROOT}",target="/home/root/$(basename ${REPO_ROOT})" \
    --mount type=bind,source="${IG_PATH}",target="/opt/isaacgym/" \
    --mount type=bind,source="${CACHE_PATH}",target="/home/root/.cache/pkm" \
    --mount type=bind,source="${DATA_PATH}",target="/input" \
    --mount type=bind,source=/tmp/docker,target="/tmp/docker" \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY \
    --shm-size=32g \
    --network host \
    --privileged \
    --gpus all \
    "$@" \
    "pkm:latest"

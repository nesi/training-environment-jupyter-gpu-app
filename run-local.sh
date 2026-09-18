#!/usr/bin/env bash
#
# Run the training session on your own machine, without deploying anything.
#
# This starts the same container image the cluster runs, with the GPU emulator
# and JupyterLab, and prints a URL to open. What it does not reproduce is the
# Open OnDemand wrapper around it: no login page, no k8s, no NFS home
# directories, no LDAP. Everything a learner actually does inside the session -
# nvidia-smi, nvtop, sbatch, the notebooks, PyTorch - behaves identically.
#
# Usage:
#   ./run-local.sh                          # defaults, as deployed
#   ./run-local.sh --device rtxpro6000      # emulate a different card
#   ./run-local.sh --vram 8GiB --gpus 2     # bigger card, two of them
#   ./run-local.sh --build                  # build from this checkout first
#   ./run-local.sh --shell                  # drop to a terminal instead
#
set -euo pipefail

REGISTRY_IMAGE="ghcr.io/nesi/training-environment-jupyter-gpu-app"

# Derive the tag from this checkout rather than hardcoding one. A pinned
# default goes stale the moment a release is cut, and then this script quietly
# runs an old image while claiming to be the current app - which is exactly
# what happened with v0.1.1. Falls back to :latest outside a git checkout.
default_image() {
    local tag
    tag=$(git -C "$(dirname "${BASH_SOURCE[0]}")" describe --tags --abbrev=0 2>/dev/null || true)
    echo "${REGISTRY_IMAGE}:${tag:-latest}"
}

IMAGE="$(default_image)"
PORT=8888
DEVICE="l4"
VRAM="1GiB"
GPUS="1"
CPUS="4"
MEMORY="8g"
BUILD=0
SHELL_ONLY=0

# \s is a GNU extension that BSD sed (macOS) does not understand, so spell the
# optional space out longhand.
usage() { sed -n '2,18p' "$0" | sed 's/^#\{1,\} \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)   IMAGE="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --device)  DEVICE="$2"; shift 2 ;;
        --vram)    VRAM="$2"; shift 2 ;;
        --gpus)    GPUS="$2"; shift 2 ;;
        --cpus)    CPUS="$2"; shift 2 ;;
        --memory)  MEMORY="$2"; shift 2 ;;
        --build)   BUILD=1; shift ;;
        --shell)   SHELL_ONLY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$BUILD" == "1" ]]; then
    echo "Building from this checkout..."
    IMAGE="gpu-app:local"
    docker build -t "$IMAGE" "$(dirname "$0")/docker"
fi

# The cluster runs amd64, so an image we have to pull should be the amd64 one
# even on an Apple Silicon Mac - it runs under emulation, slowly but correctly.
# An image already present locally is used as-is, whatever it was built for;
# forcing a platform on it would trigger a pointless pull.
PLATFORM=()
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
        PLATFORM=(--platform linux/amd64)
        echo "Note: pulling the amd64 image; it runs under emulation and is slow"
        echo "      to start. Use --build for a native image if you are iterating."
    fi
fi

# -i needs a terminal on stdin. Without this the script cannot be run from a
# pipe, a CI job or the background, which is exactly how you would smoke-test it.
TTY=(-t)
[[ -t 0 ]] && TTY=(-it)

# macOS still ships bash 3.2, where expanding an empty array under `set -u` is
# an "unbound variable" error. The ${a[@]+"${a[@]}"} form expands to nothing
# when the array is empty instead of failing.
COMMON=(
    --rm
    "${TTY[@]}"
    ${PLATFORM[@]+"${PLATFORM[@]}"}
    --cpus "$CPUS"
    --memory "$MEMORY"
    -e GPUEMU_DEVICE="$DEVICE"
    -e GPUEMU_MEM_TOTAL="$VRAM"
    -e GPUEMU_GPUS="$GPUS"
    -e TERM=xterm-256color
)

if [[ "$SHELL_ONLY" == "1" ]]; then
    echo "Starting a shell with the emulator running. Try: nvidia-smi, nvtop, sbatch"
    exec docker run "${COMMON[@]}" "$IMAGE" bash -lc 'gpuemu-ctl start >/dev/null && exec bash -l'
fi

cat <<BANNER

  Starting the GPU training session locally.

    Emulated GPU : $DEVICE, ${VRAM:-card default} VRAM, x$GPUS
    Resources    : $CPUS CPUs, $MEMORY RAM
    Image        : $IMAGE

  JupyterLab will be at:  http://localhost:${PORT}/lab
  Notebooks are under     /root/gpu-training/
  Press Ctrl-C to stop.

BANNER

exec docker run "${COMMON[@]}" -p "${PORT}:8888" "$IMAGE" bash -lc '
    set -e
    gpuemu-ctl start >/dev/null

    # Mirror what template/script.sh.erb does on the cluster, so the session
    # you see locally has the same contents as the deployed one.
    mkdir -p "${HOME}/gpu-training"
    rsync --ignore-existing -a /opt/gpu-training/notebooks/ "${HOME}/gpu-training/"
    rsync --ignore-existing -a /opt/gpu-training/examples/  "${HOME}/gpu-training/examples/"

    nvidia-smi -L
    echo

    cd "${HOME}"
    exec jupyter lab \
        --ip 0.0.0.0 --port 8888 --no-browser --allow-root \
        --ServerApp.token="" --ServerApp.password="" \
        --ServerApp.root_dir="${HOME}"
'

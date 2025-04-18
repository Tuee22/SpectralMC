#!/usr/bin/env bash
# setup_ubuntu24_docker_nvidia_idempotent.sh
# Installs Docker CE, NVIDIA driver, and NVIDIA Container Toolkit on Ubuntu 24.04
# Idempotent; avoids apt updates, service restarts, and reboots unless required.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Compatibility check
# ──────────────────────────────────────────────────────────────────────────────
if [[ ! -r /etc/os-release ]]; then
  echo "❌  /etc/os-release not found; can’t determine distro." >&2; exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ $ID != ubuntu || $VERSION_ID != 24.04 ]]; then
  echo "❌  Ubuntu 24.04 (“noble”) required. Detected: ${PRETTY_NAME}" >&2; exit 1
fi
if [[ $(dpkg --print-architecture) != amd64 ]]; then
  echo "❌  Only x86‑64 (amd64) is supported." >&2; exit 1
fi
command -v apt-get >/dev/null || { echo "❌  apt-get not found." >&2; exit 1; }
echo "✓ Environment check passed: ${PRETTY_NAME} (amd64)"
echo

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Globals & helpers
# ──────────────────────────────────────────────────────────────────────────────
need_update=false            # run `apt-get update` only when true
docker_service_changed=false # restart docker.service only when true
reboot_required=false        # reboot only when true

update_apt() {
  if $need_update; then
    echo "→ Updating package index"
    sudo apt-get update
    need_update=false
  fi
}

add_key_once() {
  local url=$1 dst=$2
  if [[ ! -f $dst ]]; then
    echo "→ Adding key: $dst"
    curl -fsSL "$url" | sudo gpg --dearmor -o "$dst"
    need_update=true
  fi
}

add_repo_once() {
  local entry=$1 listfile=$2
  if [[ ! -f $listfile ]] || ! grep -qF "$entry" "$listfile"; then
    echo "→ Adding repo: $listfile"
    echo "$entry" | sudo tee "$listfile" > /dev/null
    need_update=true
  fi
}

install_if_missing() {
  # Usage: install_if_missing <var_to_set_true_if_installed> pkg1 pkg2 …
  local _flag=$1; shift
  local miss=()
  for p in "$@"; do dpkg -s "$p" &>/dev/null || miss+=("$p"); done
  if ((${#miss[@]})); then
    update_apt
    sudo apt-get install -y "${miss[@]}"
    printf -v "$_flag" true   # set the referenced flag variable
  else
    echo "✓ Packages already present: $*"
  fi
}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Basic dependencies
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 1: Basic package prerequisites ==="
install_if_missing _dummy ca-certificates curl gnupg lsb-release software-properties-common
echo

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Docker CE
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 2: Docker CE ==="
sudo mkdir -p /etc/apt/keyrings
add_key_once https://download.docker.com/linux/ubuntu/gpg /etc/apt/keyrings/docker.gpg
docker_entry="deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu noble stable"
add_repo_once "$docker_entry" /etc/apt/sources.list.d/docker.list

docker_pkgs=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin)
install_if_missing docker_service_changed "${docker_pkgs[@]}"

if id -nG "$USER" | grep -qw docker; then
  echo "✓ User $USER already in docker group"
else
  echo "→ Adding $USER to docker group (re‑login needed)"
  sudo groupadd -f docker
  sudo usermod -aG docker "$USER"
  # no service restart or reboot needed
fi
echo

# ──────────────────────────────────────────────────────────────────────────────
# 4.  NVIDIA driver
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 3: NVIDIA driver ==="
if ! ls /etc/apt/sources.list.d/graphics-drivers-ubuntu-ppa* &>/dev/null; then
  echo "→ Adding graphics‑drivers PPA"
  sudo add-apt-repository -y ppa:graphics-drivers/ppa
  need_update=true
fi

# Install driver only if absent. Set reboot_required accordingly.
install_if_missing reboot_required nvidia-driver-570-open
echo

# ──────────────────────────────────────────────────────────────────────────────
# 5.  NVIDIA Container Toolkit
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Step 4: NVIDIA Container Toolkit ==="
add_key_once  https://nvidia.github.io/libnvidia-container/gpgkey \
              /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
nvidia_entry="deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /"
add_repo_once "$nvidia_entry" /etc/apt/sources.list.d/nvidia-container-toolkit.list

install_if_missing _dummy nvidia-docker2
echo

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Docker service (only if Docker changed)
# ──────────────────────────────────────────────────────────────────────────────
if $docker_service_changed; then
  echo "=== Step 5: Restarting docker.service (Docker packages changed) ==="
  sudo systemctl enable docker
  sudo systemctl restart docker
  echo
fi

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Conditional reboot (only if driver installed)
# ──────────────────────────────────────────────────────────────────────────────
if $reboot_required; then
  echo "=== Reboot required: NVIDIA driver installed ==="
  echo "System will reboot in 10 seconds. Press Ctrl‑C to cancel."
  sleep 10
  sudo reboot
else
  echo "No reboot necessary; all done."
fi
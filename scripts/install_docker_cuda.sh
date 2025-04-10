#!/usr/bin/env bash
#
# setup_ubuntu24_docker_nvidia.sh
# Installs Docker CE, NVIDIA driver, and NVIDIA Container Toolkit on Ubuntu 24.04.
# Also configures Docker so it can be run by non-root users (no sudo needed).

set -e

echo "=== Step 1: System Update and Basic Dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common

echo "=== Step 2: Install Docker CE ==="
# Add Docker's official GPG key:
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the stable Docker repository (using Ubuntu 24.04 codename 'noble'):
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
   https://download.docker.com/linux/ubuntu noble stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to the docker group so Docker can be used without sudo
echo "=== Step 2: Grant non-root Docker usage ==="
sudo groupadd -f docker
sudo usermod -aG docker "$USER"

echo "=== Step 3: Install NVIDIA Driver ==="
# Install from Ubuntu's official repository (which typically provides newer drivers for each release).
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-driver-570-open

echo "=== Step 4: Install NVIDIA Container Toolkit ==="
# Add NVIDIA Container Toolkit public GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the stable .deb repository
# This is a generic repository that supports Debian-based distros (including Ubuntu 24.04).
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2

echo "=== Step 5: Restart Docker ==="
sudo systemctl enable docker
sudo systemctl restart docker

echo "=== Installation Complete ==="
echo "1) For best results, reboot if you haven't since installing the NVIDIA driver."
echo "2) After reboot, verify NVIDIA driver with 'nvidia-smi'."
echo "3) Test Docker GPU support with:"
echo "   docker run --rm --gpus all nvidia/cuda:12.8.1-devel-ubuntu22.04 nvidia-smi"
echo "4) IMPORTANT: You must log out and log back in (or run 'newgrp docker') to use Docker without sudo."
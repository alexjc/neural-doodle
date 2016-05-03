#!/usr/bin/env bash

# Install dependencies

apt-get update
apt-get install --assume-yes "wget"

# Install Docker Engine

wget --quiet --output-document="-" "https://get.docker.com" | sh
usermod --append --groups="docker" "$SUDO_USER"

# Install an NVIDIA Docker wrapper

NVIDIA_DOCKER="nvidia-docker_1.0.0.beta.2-1_amd64.deb"
NVIDIA_DOCKER_URL="https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-beta.2"

wget --directory-prefix="/tmp/" --continue "$NVIDIA_DOCKER_URL/$NVIDIA_DOCKER"
dpkg --install "/tmp/$NVIDIA_DOCKER"
rm --interactive "/tmp/$NVIDIA_DOCKER"

nvidia-docker volume setup

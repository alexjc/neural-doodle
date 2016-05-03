#!/usr/bin/env bash

# Install dependencies

apt-get update
apt-get install --assume-yes                      \
    "linux-source"                                \
    "linux-headers-$(uname --kernel-release)"     \
    "linux-image-extra-$(uname --kernel-release)" \
    "build-essential"                             \
    "wget"

# Install NVIDIA CUDA with NVIDIA GPU drivers

CUDA_REPOSITORY="cuda-repo-ubuntu1404_7.5-18_amd64.deb"
CUDA_REPOSITORY_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64"

wget --directory-prefix="/tmp/" --continue "$CUDA_REPOSITORY_URL/$CUDA_REPOSITORY"
dpkg --install "/tmp/$CUDA_REPOSITORY"
apt-get update
apt-get install "cuda"

rm --interactive "/tmp/$CUDA_REPOSITORY"

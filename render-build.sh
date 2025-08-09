#!/usr/bin/env bash
set -e

# Install Rust 
curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --default-toolchain stable
source $HOME/.cargo/env

# Upgrade pip & install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

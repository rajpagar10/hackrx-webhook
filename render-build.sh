#!/usr/bin/env bash
set -e  # stop if any command fails

# Install Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env

# Upgrade pip & install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

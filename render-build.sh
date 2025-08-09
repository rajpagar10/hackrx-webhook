#!/usr/bin/env bash
set -o errexit

# Check Rust exists
rustc --version
cargo --version

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

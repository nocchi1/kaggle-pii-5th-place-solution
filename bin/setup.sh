#!/bin/bash

# Install rye and
curl -sSf https://rye.astral.sh/get | bash
source ~/.profile

# Set up virtual environment
rye pin $(cat .python-version)
rye sync

# Add project root directory to PYTHONPATH
append_dir=$(pwd)
if ! grep -q "PYTHONPATH=.*$append_dir" ~/.bashrc; then
    echo "export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$append_dir" >> ~/.bashrc
fi
source ~/.bashrc

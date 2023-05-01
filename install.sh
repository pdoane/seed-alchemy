#!/usr/bin/env bash

rm -rf .venv
python3 -m venv .venv --prompt SimpleDiffusion
source .venv/bin/activate
pip install -r requirements.txt
python simple_diffusion/install.py

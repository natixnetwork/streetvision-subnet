#!/bin/bash
set -a
source validator.env
set +a

export PYTHONPATH=$(pwd):$PYTHONPATH

poetry run python natix/validator/scripts/run_cache_updater.py
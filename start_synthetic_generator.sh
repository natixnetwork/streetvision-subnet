#!/bin/bash
set -a
source validator.env
set +a

export PYTHONPATH=$(pwd):$PYTHONPATH

# Default batch size - can be overridden by environment variable
BATCH_SIZE=${SYNTHETIC_BATCH_SIZE:-3}

poetry run python natix/validator/scripts/run_data_generator.py --batch-size $BATCH_SIZE
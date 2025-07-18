#!/bin/bash
set -a
source validator.env
set +a

export PYTHONPATH=$(pwd):$PYTHONPATH

# Default batch size - can be overridden by environment variable
# Using batch size 1 to avoid memory issues with large models
BATCH_SIZE=${SYNTHETIC_BATCH_SIZE:-1}

export TOKENIZERS_PARALLELISM=false

poetry run python natix/validator/scripts/run_data_generator.py --batch-size $BATCH_SIZE
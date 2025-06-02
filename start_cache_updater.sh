#!/bin/bash

# Load environment variables from .env file & set defaults
set -a
source validator.env
set +a

export PYTHONPATH=$(pwd):$PYTHONPATH

CACHE_UPDATE_PROCESS_NAME="natix_cache_updater"

# Clear cache if specified
while [[ $# -gt 0 ]]; do
  case $1 in
    --clear-cache)
      rm -rf ~/.cache/natix
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# STOP REAL DATA CACHE UPDATER PROCESS
if pm2 list | grep -q "$CACHE_UPDATE_PROCESS_NAME"; then
  echo "Process '$CACHE_UPDATE_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $CACHE_UPDATE_PROCESS_NAME
fi

echo "Starting real data cache updater process"
poetry run python natix/validator/scripts/run_cache_updater.py

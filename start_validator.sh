#!/bin/bash

# Load environment variables from .env file & set defaults
set -a
source validator.env
set +a

: ${VALIDATOR_PROXY_PORT:=10913}
: ${DEVICE:=cuda}

VALIDATOR_PROCESS_NAME="natix_validator"
DATA_GEN_PROCESS_NAME="natix_data_generator"
CACHE_UPDATE_PROCESS_NAME="natix_cache_updater"

# Login to Weights & Biases
if ! wandb login $WANDB_API_KEY; then
  echo "Failed to login to Weights & Biases with the provided API key."
  exit 1
fi

# Login to Hugging Face
if ! huggingface-cli login --token $HUGGING_FACE_TOKEN; then
  echo "Failed to login to Hugging Face with the provided token."
  exit 1
fi

echo "Starting validator process"
poetry run python neurons/validator.py \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $VALIDATOR_AXON_PORT \
  --proxy.port $VALIDATOR_PROXY_PORT \
  --proxy.proxy_client_url $PROXY_CLIENT_URL\
  --logging.debug
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

# STOP VALIDATOR PROCESS
# if pm2 list | grep -q "$VALIDATOR_PROCESS_NAME"; then
#   echo "Process '$VALIDATOR_PROCESS_NAME' is already running. Deleting it..."
#   pm2 delete $VALIDATOR_PROCESS_NAME
# fi

# STOP REAL DATA CACHE UPDATER PROCESS
if pm2 list | grep -q "$CACHE_UPDATE_PROCESS_NAME"; then
  echo "Process '$CACHE_UPDATE_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $CACHE_UPDATE_PROCESS_NAME
fi

# STOP SYNTHETIC DATA GENERATOR PROCESS
# if pm2 list | grep -q "$DATA_GEN_PROCESS_NAME"; then
#   echo "Process '$DATA_GEN_PROCESS_NAME' is already running. Deleting it..."
#   pm2 delete $DATA_GEN_PROCESS_NAME
# fi

# echo "Verifying access to synthetic image generation models. This may take a few minutes."
# if ! python3 natix/validator/verify_models.py; then
#   echo "Failed to verify diffusion models. Please check the configurations or model access permissions."
#   exit 1
# fi

echo "Starting real data cache updater process"
pm2 start natix/validator/scripts/run_cache_updater.py --name $CACHE_UPDATE_PROCESS_NAME

echo "Starting validator process"
python neurons/validator.py \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $VALIDATOR_AXON_PORT \
  --proxy.port $VALIDATOR_PROXY_PORT \
  --logging.debug

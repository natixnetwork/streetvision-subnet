#!/bin/bash

set -a
source miner.env
set +a

export PYTHONPATH=$(pwd):$PYTHONPATH

poetry run python neurons/miner.py \
  --neuron.image_detector ${IMAGE_DETECTOR:-None} \
  --neuron.image_detector_config ${IMAGE_DETECTOR_CONFIG:-None} \
  --neuron.image_detector_device ${IMAGE_DETECTOR_DEVICE:-None} \
  --netuid $NETUID \
  --model_url $MODEL_URL \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $MINER_AXON_PORT \
  --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT \
  --neuron.preferred_challenges $PREFERRED_CHALLENGES \
  --logging.debug

#!/bin/bash

###########################################
# System Updates and Package Installation #
###########################################

# Update system
sudo apt update -y

# Install core dependencies
sudo apt install -y \
    python3 \
    python3-pip \
    nano \
    libgl1 \
    npm \
    ffmpeg \
    unzip

# Install build dependencies
sudo apt install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libjxl-dev

# Install process manager
sudo npm install -g pm2@latest

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

############################
# Environment Files Setup  #
############################

# Create miner.env if it doesn't exist
if [ -f "miner.env" ]; then
    echo "File 'miner.env' already exists. Skipping creation."
else
    cat > miner.env << 'EOL'
# StreetVision Miner Configuration
#--------------------
# following are initial values
IMAGE_DETECTOR=ViT
IMAGE_DETECTOR_CONFIG=ViT_roadwork.yaml
VIDEO_DETECTOR=TALL
VIDEO_DETECTOR_CONFIG=tall.yaml

# Device Settings
IMAGE_DETECTOR_DEVICE=cpu # Options: cpu, cuda
VIDEO_DETECTOR_DEVICE=cpu

NETUID=323                           # 323 for testnet, 72 for mainnet
SUBTENSOR_NETWORK=test               # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443
                                     # Endpoints:
                                     # - wss://entrypoint-finney.opentensor.ai:443
                                     # - wss://test.finney.opentensor.ai:443/
                                     
# Wallet Configuration
WALLET_NAME=
WALLET_HOTKEY=

# Miner Settings
MINER_AXON_PORT=8091
BLACKLIST_FORCE_VALIDATOR_PERMIT=True # Force validator permit for blacklisting

# Miner details
MODEL_URL=
PROXY_CLIENT_URL=https://hydra.natix.network
EOL
    echo "File 'miner.env' created."
fi

# Create validator.env if it doesn't exist
if [ -f "validator.env" ]; then
    echo "File 'validator.env' already exists. Skipping creation."
else
    cat > validator.env << 'EOL'
# StreetVision Validator Configuration
#--------------------
NETUID=72
SUBTENSOR_NETWORK=finney
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443

WALLET_NAME=default
WALLET_HOTKEY=default

VALIDATOR_AXON_PORT=8092
VALIDATOR_PROXY_PORT=10913
PROXY_CLIENT_URL=https://hydra.natix.network
DEVICE=cuda

WANDB_API_KEY=your_wandb_api_key_here
HUGGING_FACE_TOKEN=your_hugging_face_token_here
EOL
    echo "File 'validator.env' created."
fi

echo "Environment setup completed successfully with StreetVision."

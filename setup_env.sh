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

# Detector Configuration
IMAGE_DETECTOR=CAMO                   # Options: CAMO, UCF, NPR, None
IMAGE_DETECTOR_CONFIG=camo.yaml       # Configs in base_miner/deepfake_detectors/configs
VIDEO_DETECTOR=TALL                   # Options: TALL, None
VIDEO_DETECTOR_CONFIG=tall.yaml       # Configs in base_miner/deepfake_detectors/configs

# Device Settings
IMAGE_DETECTOR_DEVICE=cpu             # Options: cpu, cuda
VIDEO_DETECTOR_DEVICE=cpu

# Subtensor Network Configuration
NETUID=72                            # Mainnet NETUID
SUBTENSOR_NETWORK=finney            # Networks: finney, testnet, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-mainnet.opentensor.ai:443
                                     # Endpoints:
                                     # - wss://entrypoint-mainnet.opentensor.ai:443
                                     # - wss://entrypoint-testnet.opentensor.ai:443/

# Wallet Configuration
WALLET_NAME=default
WALLET_HOTKEY=default

# Miner Settings
MINER_AXON_PORT=8091
MODEL_URL=your_username/your_model_name # Model URL on Hugging Face
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

# Subtensor Network Configuration
NETUID=72                            # Mainnet NETUID
SUBTENSOR_NETWORK=finney            # Networks: finney, testnet, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                     # Endpoints:
                                     # - wss://entrypoint-finney.opentensor.ai:443
                                     # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration
WALLET_NAME=default
WALLET_HOTKEY=default

# Validator Settings
VALIDATOR_AXON_PORT=8092             # If using RunPod, must be >= 70000 for symmetric mapping
VALIDATOR_PROXY_PORT=10913
DEVICE=cuda

# API Keys
WANDB_API_KEY=your_wandb_api_key_here
HUGGING_FACE_TOKEN=your_hugging_face_token_here
PROXY_CLIENT_URL=https://hydra.natix.network
EOL
    echo "File 'validator.env' created."
fi

echo "Environment setup completed successfully with StreetVision."

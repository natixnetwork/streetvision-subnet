# Miner Guide

## Table of Contents

1. [Installation 🔧](#installation)

   * [Data 📊](#data)
   * [Registration ✍️](#registration)
2. [Mining ⛏️](#mining)

## Before You Proceed ⚠️

**IMPORTANT**: If you are new to Bittensor, we recommend familiarizing yourself with the basics on the [Bittensor Website](https://bittensor.com/) before proceeding.

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml). A GPU is recommended for training, but not required for inference while running a miner.

## Installation

Download the repository and navigate to the folder:

```bash
git clone https://github.com/natixnetwork/natix-subnet.git && cd natix-subnet
```

We recommend using a [Poetry](https://python-poetry.org/docs/) environment with Python 3.11 to manage dependencies:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry env use 3.11
poetry install
```

This will install all necessary dependencies and prepare your environment.

### Data

*Only required for training -- deployed miner instances do not need access to these datasets.*

Optionally, pre-download the training datasets by running:

```bash
poetry run python base_miner/datasets/download_data.py
```

The default list of datasets and their download location is defined in `base_miner/config.py`.

## Mining Requirements ⚠️

To mine on our subnet, you must have a registered hotkey and [have submitted at least one model](#submitted-a-model).

## Registration

To reduce the risk of deregistration due to technical issues or poor model performance, we recommend the following:

1. Test your miner on testnet before mining on mainnet.
2. Before registering your hotkey on mainnet, verify that your port is open:

```bash
curl your_ip:your_port
```

#### Mainnet

```bash
btcli s register --netuid 72 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 323 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

## Economy registration
Once registered on-chain, you must also register on the **Natix application server**. make sure you've registered, and received your `uid` on Bittensor (as explained above).
To register with the Natix network, you must sign a recent timestamp with your **Bittensor** hot key.


Use the `./register.sh` script to simplify registration with the Natix application server:

```bash
./register.sh <uid> <bt_wallet_name> <bt_hotkey_name> miner <hf_model_path>
```

**Example:**
```bash
./register.sh 10 reyraa default miner reyraa/roadwork
```

This script will:
- Generate a fresh timestamp
- Sign it with your **Bittensor** hot key
- Send a POST request to:  
  `https://hydra.natix.network/participant/register`

---

## Mining
Run `./setup_env.sh` to generate a `miner.env` file with default configuration.

Make sure to update your `miner.env` file with your wallet name, hotkey, miner port, and model configuration.
```
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
MODEL_URL= # The URL to your Hugging-face repository
```

Then, start your miner with:

```bash
chmod +x ./start_miner.sh
./start_miner.sh
```

This will launch `run_neuron.py` using Poetry.

You can also optionally run a cache updater service to improve image caching performance:

```bash
chmod +x ./start_cache_updater.sh
./start_cache_updater.sh
```

This invokes `natix/validator/scripts/run_cache_updater.py` in the background.

## Submitted a Model

Miners must publish their model to Hugging Face and include a `model_card.json` with the following format:

```json
{
  "model_name": "<ARBITRARY_MODEL_NAME>",
  "description": "<DESCRIPTION>",
  "version": <VERSION NUMBER IN X.Y.Z format>,
  "submitted_by": "<WALLET_HOTKEY_ADDRESS>",
  "submission_time": <TIMESTAMP>
}
```

Update the `MODE_URL` variable in your `miner.env` to reflect your Hugging Face repository.

## Deploy Your Model

Update your `miner.env` file to use your trained detector class and configuration.

* Detector types are defined in `base_miner/registry.py`.
* Config files live in `base_miner/detectors/configs/`.
* UCFDetector requires a `train_config` path in its config YAML.

Weights should be placed under `base_miner/<detector_type>/weights`. If missing, they will be pulled from Hugging Face according to the `hf_repo` field in the config.

## Training

To improve beyond the baseline model, experiment with new datasets, architectures, or hyperparameters.

## TensorBoard

Start TensorBoard to view training metrics:

```bash
tensorboard --logdir=./base_miner/checkpoints/<experiment_name>
```

For remote machines:

```bash
ssh -L 7007:localhost:6006 your_username@your_ip
```

Then on the remote machine:

```bash
tensorboard --logdir=./base_miner/checkpoints/<experiment_name> --host 0.0.0.0 --port 6006
```

View it locally at `http://localhost:7007`

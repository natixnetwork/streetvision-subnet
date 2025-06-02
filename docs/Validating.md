# Validator Guide

## Table of Contents

1. [Installation 🔧](#installation)
2. [Validator Requirements ⚠️](#validator-requirements-⚠️)
3. [Registration ✍️](#registration)
   - [Scripted Registration](#scripted-registration)
4. [Validating ✅](#validating)

## Before You Proceed ⚠️

**Ensure you're running Subtensor locally** to minimize outages and improve performance.  
Refer to the [Run a Subtensor Node Locally guide](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

**Check the minimum compute requirements** for our subnet, defined in the [Minimum compute YAML configuration](../min_compute.yml).

---

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/natixnetwork/natix-subnet.git && cd natix-subnet
```

We now use [Poetry](https://python-poetry.org/) for dependency management.  
Make sure Poetry is installed, then activate the environment and install dependencies:

```bash
poetry env use python3.11
poetry install
poetry shell
```

Python **3.11** is the preferred version.


## Acquiring a UID

### Mainnet Registration

```bash
btcli s register --netuid 72 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

### Testnet Registration

```bash
btcli s register --netuid 323 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

## NATIX Network Registration
Once registered on-chain, you must also register on the **NATIX Application Server**. After   `uid` on Bittensor (as explained above) is received, Validatos must sign a recent timestamp with  **Bittensor** hot key to be white-listed in NATIX application sever.

This step is **required** in order to receive organic requests comming from NATIX Network.

Use the `./register.sh` script to perform this step.

```bash
./register.sh <uid> <bt_wallet_name> <bt_hotkey_name> validator
```

**Example:**
```bash
./register.sh 10 reyraa default validator
```

This script will:
- Generate a fresh timestamp
- Sign it with your **Bittensor** hot key
- Send a POST request to:  
  `https://hydra.natix.network/participant/register`

---

## Validating

Update your `validator.env` file with your configuration:

```bash
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
```

To run the validator:
```bash
pm2 start ecosystem.validator.config.js
```

Optional flags:
- `--no-auto-updates`: Disables automatic code updates
- `--no-self-heal`: Disables automatic restart every 6 hours

---

### Exposed Ports
Please note that you need to expose the port numbers you define by `VALIDATOR_AXON_PORT` and `VALIDATOR_PROXY_PORT` for incoming requests.

---

That’s it — you’re ready to validate!


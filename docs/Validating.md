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

---

## Validator Requirements ⚠️  
**Last updated: May 20th, 2025**

To validate on the Natix subnet, you must have a registered hotkey and meet **both** of the following criteria:

- **$NATIX Staking**:  
  Validators must stake **72,727,272 $NATIX**, equivalent to approximately $50,000 based on the average price over the past 30 days. This requirement is reviewed and adjusted by the subnet owner every 1–3 months.

- **Alpha Token Holding**:  
  Validators must hold **12,500 Alpha tokens**, also approximately $50,000 in value. This requirement is reviewed and adjusted periodically.

> **Grace Period**: Validators who registered before **May 20th, 2025**, have a **4-week grace period** to meet these requirements.

---

## Acquiring a UID

### Mainnet Registration

```bash
btcli s register --netuid 72 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

### Testnet Registration

```bash
btcli s register --netuid 323 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

## Economy registration
Once registered on-chain, you must also register on the **Natix application server**. make sure you've registered, and received your `uid` on Bittensor (as explained above).
To register with the Natix network, you must sign a recent timestamp with your **Bittensor** hot key.


Use the `./register` script to simplify registration with the Natix application server:

```bash
./register <uid> <bt_wallet_name> <bt_hotkey_name> validator
```

**Example:**
```bash
./register 10 reyraa default validator
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

PROXY_CLIENT_URL=https://hydra.natix.network
```

To run the validator, use the `./start_validator.sh` script:

```bash
chmod +x ./start_validator.sh
./start_validator.sh
```

This script runs with **no arguments** and uses the values defined in `validator.env`.

> **Note**: You may optionally run the validator using [PM2](https://pm2.keymetrics.io/), but this is not required. If you choose to use PM2:

```bash
pm2 start run_neuron.py -- --validator
```

Optional flags:
- `--no-auto-updates`: Disables automatic code updates
- `--no-self-heal`: Disables automatic restart every 6 hours

---

### PM2 Note

You may choose to manage your validator with PM2 if desired, but by default, it does **not** use PM2.


### Exposed Ports
Please note that you need to expose the port numbers you define by `VALIDATOR_AXON_PORT` and `VALIDATOR_PROXY_PORT` for incoming requests.

---

That’s it — you’re ready to validate!


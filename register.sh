#!/bin/bash

# Colors for style
CYAN="\033[36m"
YELLOW="\033[33m"
GREEN="\033[32m"
RED="\033[31m"
RESET="\033[0m"

# Help banner
show_help() {
  echo -e "${CYAN}"
  echo "┌────────────────────────────────────────────────────────────┐"
  echo -e "│         ${GREEN}NATIX Street Vision economy registration${CYAN}         │"
  echo "├────────────────────────────────────────────────────────────┤"
  echo -e "│ ${YELLOW}Usage:${CYAN} ./register.sh <uid> <bt_wallet_name> <bt_hotkey_name> <solana_keypair_path> <participant_type> [hf_model_path]"
  echo "│"
  echo "│ Example for miner:"
  echo "│   ./register.sh 10 reyraa default ~/.config/solana/reyraa.json miner reyraa/roadwork"
  echo "│"
  echo "│ Example for validator:"
  echo "│   ./register.sh 10 reyraa default ~/.config/solana/reyraa.json validator"
  echo "│"
  echo "│ This script will:"
  echo "│ - Generate a secure timestamp"
  echo "│ - Sign it with both Solana & Bittensor keys"
  echo "│ - Register with the NATIX Street Vision endpoint"
  echo "└────────────────────────────────────────────────────────────┘"
  echo -e "${RESET}"
  exit 1
}

# Validate args
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  show_help
fi

BT_UID=$1
BT_WALLET=$2
BT_HOTKEY=$3
SOLANA_KEYPAIR=$4
PARTICIPANT_TYPE=$5

echo $PARTICIPANT_TYPE

if [ "$PARTICIPANT_TYPE" == "miner" ]; then
  if [ "$#" -ne 6 ]; then
    echo -e "${RED}Error: 'miner' type requires <hf_model_path> argument.${RESET}"
    show_help
  fi
  HF_MODEL=$6
else
  HF_MODEL=""
fi

echo "UID at start: $BT_UID"

# Install missing tools
command -v solana >/dev/null 2>&1 || {
  echo -e "${YELLOW}Installing Solana CLI...${RESET}"
  sh -c "$(curl -sSfL https://release.solana.com/stable/install)"
  export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
}

command -v btcli >/dev/null 2>&1 || {
  echo -e "${RED}Error: Bittensor CLI (btcli) not found. Please install it manually.${RESET}"
  exit 1
}

command -v jq >/dev/null 2>&1 || {
  echo -e "${YELLOW}Installing jq...${RESET}"
  if command -v apt >/dev/null 2>&1; then sudo apt install -y jq
  elif command -v brew >/dev/null 2>&1; then brew install jq
  else echo -e "${RED}Error: Cannot install jq. Install it manually.${RESET}"; exit 1
  fi
}

# Generate timestamp
TIMESTAMP=$(date +%s)
echo -e "${GREEN}Generated timestamp:${RESET} $TIMESTAMP"

# Solana signing
SOLANA_PUBKEY=$(solana address --keypair "$SOLANA_KEYPAIR")
SOLANA_SIGNATURE=$(solana sign-offchain-message --keypair "$SOLANA_KEYPAIR" --no-address-labels "$TIMESTAMP")

if [[ -z "$SOLANA_SIGNATURE" || -z "$SOLANA_PUBKEY" ]]; then
  echo -e "${RED}[ERROR]${RESET} Failed to extract Solana signature or public key."
  exit 1
fi

echo -e "${GREEN}Generated signature for${RESET} Solana"

# Bittensor signing
BT_SIGNATURE=$(btcli w sign --wallet-name "$BT_WALLET" --hotkey "$BT_HOTKEY" --use-hotkey --message "$TIMESTAMP" --json-out | jq -r '.signed_message')
BT_PUBKEY=$(btcli w list --json-out | jq -r ".wallets[] | select(.name==\"$BT_WALLET\") | .hotkeys[] | select(.name==\"$BT_HOTKEY\") | .ss58_address")
echo -e "${GREEN}Generated signature for${RESET} Bittensor"

# Construct JSON
if [ "$PARTICIPANT_TYPE" == "miner" ]; then
  JSON=$(jq -n \
    --arg uid "$BT_UID" \
    --arg msg "$TIMESTAMP" \
    --arg bt_pk "$BT_PUBKEY" \
    --arg bt_sig "$BT_SIGNATURE" \
    --arg sol_pk "$SOLANA_PUBKEY" \
    --arg sol_sig "$SOLANA_SIGNATURE" \
    --arg type "$PARTICIPANT_TYPE" \
    --arg repo "$HF_MODEL" \
    '{
      uid: $uid,
      message: $msg,
      public_key: $sol_pk,
      signature: $sol_sig,
      natix_public_key: $bt_pk,
      natix_signature: $bt_sig,
      type: $type,
      model_repo: $repo
    }'
  )
else
  JSON=$(jq -n \
    --arg uid "$BT_UID" \
    --arg msg "$TIMESTAMP" \
    --arg bt_pk "$BT_PUBKEY" \
    --arg bt_sig "$BT_SIGNATURE" \
    --arg sol_pk "$SOLANA_PUBKEY" \
    --arg sol_sig "$SOLANA_SIGNATURE" \
    --arg type "$PARTICIPANT_TYPE" \
    '{
      uid: $uid,
      message: $msg,
      public_key: $sol_pk,
      signature: $sol_sig,
      natix_public_key: $bt_pk,
      natix_signature: $bt_sig,
      type: $type
    }'
  )
fi

# POST request
echo -e "${GREEN}Sending registration request to Natix...${RESET}"
curl -s -X POST https:/hydra.natix.network/participant/register \
  -H "Content-Type: application/json" \
  -d "$JSON" | jq

echo -e "${YELLOW}⚠️  Registration request sent. Check the response above to confirm success.${NC}"
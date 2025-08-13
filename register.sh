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
  echo -e "│ ${YELLOW}Usage:${CYAN} ./register.sh <uid> <bt_wallet_name> <bt_hotkey_name> <participant_type> [hf_model_path]"
  echo "│"
  echo "│ Example for miner:"
  echo "│   ./register.sh 10 reyraa default miner reyraa/roadwork"
  echo "│"
  echo "│ Example for validator:"
  echo "│   ./register.sh 10 reyraa default validator"
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
if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
  show_help
fi

BT_UID=$1
BT_WALLET=$2
BT_HOTKEY=$3
PARTICIPANT_TYPE=$4

if [ "$PARTICIPANT_TYPE" == "miner" ]; then
  if [ "$#" -ne 6 ]; then
    echo -e "${RED}Error: 'miner' type requires <hf_model_path> argument.${RESET}"
    show_help
  fi
  HF_MODEL=$5
  BASE_URL=$6
else
  HF_MODEL=""
  BASE_URL=$5
fi

echo "UID at start: $BT_UID"

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
    --arg type "$PARTICIPANT_TYPE" \
    --arg repo "$HF_MODEL" \
    '{
      uid: $uid,
      message: $msg,
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
    --arg type "$PARTICIPANT_TYPE" \
    '{
      uid: $uid,
      message: $msg,
      natix_public_key: $bt_pk,
      natix_signature: $bt_sig,
      type: $type
    }'
  )
fi

# POST request
BASE_URL="${BASE_URL:-https://hydra.natix.network}"

POLL_INTERVAL="${POLL_INTERVAL_SECONDS:-3}"   # seconds
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"     # total timeout
MAX_ATTEMPTS=$(( TIMEOUT_SECONDS / POLL_INTERVAL ))

# POST request
echo -e "${GREEN}Sending registration request to Natix...${RESET}"
ENQUEUE_RES="$(curl -sS -X POST "$BASE_URL/participant/register" \
  -H "Content-Type: application/json" \
  -d "$JSON")"

if ! echo "$ENQUEUE_RES" | jq . >/dev/null 2>&1; then
  echo -e "${RED}Server response was not JSON:${RESET}"
  echo "$ENQUEUE_RES"
  exit 3
fi

DETAIL="$(echo "$ENQUEUE_RES" | jq -r '.detail // empty')"

if [[ -n "$DETAIL" ]]; then
  echo -e "${YELLOW}!  $DETAIL${RESET}"
else
  echo -e "${YELLOW}!  Unexpected response:${RESET}"
  echo "$ENQUEUE_RES" | jq
fi



# Friendly guide for miners
echo
echo -e "${CYAN}Next steps:${RESET}"
echo -e "  We will now periodically check your registration status."
echo -e "  If the registration doesn't complete during this script run,"
echo -e "  you can manually check it later with this command:"
echo
echo -e "    ${BOLD}curl -s ${BASE_URL}/participant/registration-status/${uid} | jq${RESET}"
echo
echo -e "  Replace 'jq' with 'python -m json.tool' if you don't have jq installed."
echo

echo -e "${CYAN}Polling registration status for UID ${YELLOW}$uid${CYAN} (every ${POLL_INTERVAL}s, timeout ${TIMEOUT_SECONDS}s)…${RESET}"



echo -e "${CYAN}Polling registration status for UID ${YELLOW}$BT_UID${CYAN} (every ${POLL_INTERVAL}s, timeout ${TIMEOUT_SECONDS}s)…${RESET}"

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  ((attempt++))
  STATUS_RES="$(curl -sS "$BASE_URL/participant/registration-status/$BT_UID")"

  echo $STATUS_RES

  if ! echo "$STATUS_RES" | jq . >/dev/null 2>&1; then
    echo -e "${RED}Status endpoint returned non-JSON:${RESET} $STATUS_RES"
    sleep "$POLL_INTERVAL"
    continue
  fi

  status="$(echo "$STATUS_RES" | jq -r '.status // empty')"
  error_msg="$(echo "$STATUS_RES" | jq -r '.error_message // empty')"

  case "$status" in
    succeeded|success)
      echo -e "${GREEN}Registration succeeded for UID $uid.${RESET}"
      echo "$STATUS_RES" | jq
      exit 0
      ;;
    failed|error)
      echo -e "${RED}Registration failed for UID $uid.${RESET}"
      [[ -n "$error_msg" ]] && echo -e "${RED}Reason:${RESET} $error_msg"
      echo "$STATUS_RES" | jq
      exit 1
      ;;
    pending|queued|processing|"")
      printf "\r${CYAN}…still waiting (attempt %d/%d)${RESET}   " "$attempt" "$MAX_ATTEMPTS"
      sleep "$POLL_INTERVAL"
      ;;
    *)
      echo -e "\n${YELLOW}Unknown status '$status'. Full payload:${RESET}"
      echo "$STATUS_RES" | jq
      sleep "$POLL_INTERVAL"
      ;;
  esac
done

echo -e "\n${YELLOW}⏳ Timed out after ${TIMEOUT_SECONDS}s waiting for registration to finish.${RESET}"
echo -e "${YELLOW}Tip:${RESET} Increase TIMEOUT_SECONDS or check the server logs."
exit 2

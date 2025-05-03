#!/bin/bash

# Base settings
WALLET_NAME="sync2"
WALLET_PATH="$HOME/.bittensor/wallets"
BASE_URL="http://localhost"
START_PORT=9000  # Base port to increment for each hotkey

# List of hotkeys
HOTKEYS=("net47_p14")
# "net47_p7")
#"net47_p9" )

# "net47_p12"  "net47_p14"  "net47_p10"  "net47_p23"  "net47_p8"   "net47_p15"  "net47_p21")
#  "net47_p17"  "net47_p22"  "net47_p24"  "net47_p11"  "net47_p20"  "net47_p16"  "net47_p27"  "net47_p25"  "net47_p29"  "net47_p26"  "net47_p19"  "net47_p30"  "net47_p28"  "net47_p13")

for i in "${!HOTKEYS[@]}"; do
  HOTKEY="${HOTKEYS[$i]}"
  PORT=$((START_PORT + i * 2))      # Sidecar port
  AXON_PORT=$((PORT + 1))           # Axon port

  # Set environment variables for each instance
  update-env WALLET_NAME "$WALLET_NAME"
  update-env WALLET_HOTKEY "$HOTKEY"
  update-env WALLET_PATH "$WALLET_PATH"
  update-env SIDECAR_BITTENSOR__BASE_URL "${BASE_URL}:${PORT}"
  update-env AXON_PORT "$AXON_PORT"

  # Start sidecar
  pm2 start python --name "sidecar-${HOTKEY}" -- -m uvicorn sidecar_bittensor.server:app --host 127.0.0.1 --port "$PORT"

  # Start miner
  pm2 start condense_miner/main.py --interpreter python3 --name "miner-${HOTKEY}"
done
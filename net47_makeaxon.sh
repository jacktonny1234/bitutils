#!/bin/bash

# Base settings
WALLET_NAME="cv"
WALLET_PATH="$HOME/.bittensor/wallets"
BASE_URL="http://localhost"
START_PORT=9000  # Base port to increment for each hotkey

# List of hotkeys
HOTKEYS=("net47_uid94")
#( "net47_pa3"  "net47_pa2"  "net47_pa6"  "net47_pa4"  "net47_pa1"  "net47_pa5")
#("net47_p18" "net47_p12"  "net47_p14"  "net47_p10"  "net47_p23"  "net47_p8"   "net47_p15"  "net47_p21" "net47_p17"  "net47_p22"  "net47_p24"  "net47_p11"  "net47_p20"  "net47_p16"  "net47_p27"  "net47_p25"  "net47_p29"  "net47_p26"  "net47_p19"  "net47_p30"  "net47_p28"  "net47_p13")

for i in "${!HOTKEYS[@]}"; do
  HOTKEY="${HOTKEYS[$i]}"
  update-env WALLET_HOTKEY "$HOTKEY"

 # PORT=$((START_PORT + i * 2))      # Sidecar port
  pm2 start condense_miner/main.py --interpreter python3 --name "1miner-${HOTKEY}"
  sleep 60
  pm2 stop "1miner-${HOTKEY}"
done
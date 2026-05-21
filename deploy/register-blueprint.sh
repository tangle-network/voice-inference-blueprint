#!/usr/bin/env bash
# Register the voice-inference blueprint on Tangle.
#
# Single-shot flow: deploys InferenceBSM (non-upgradeable, constructor takes
# the tsUSD payment-token address) AND calls Tangle.createBlueprint in the
# same broadcast via `contracts/script/RegisterBlueprint.s.sol`. This
# replaces the prior cargo-tangle CLI two-stage flow.
#
# Prerequisites:
#   - forge installed
#   - Deployer wallet funded on the target network
#
# Usage (Base Sepolia, against the already-deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export TSUSD_ADDRESS=0x036CbD53842c5426634e7929541eC2318f3dCF7e  # USDC sepolia
#   ./deploy/register-blueprint.sh
#
# Local anvil (LocalTestnet snapshot):
#
#   export RPC_URL=http://127.0.0.1:8545
#   ./deploy/register-blueprint.sh   # uses anvil deployer key + Tangle/tsUSD defaults
#
# Outputs (parsed by deployment scripts, do not change without coordinating):
#   DEPLOY_INFERENCE_BSM=<address>
#   DEPLOY_VOICE_BLUEPRINT_ID=<u64>

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"

# Defaults match the Base Sepolia deployment; override via env for other chains.
TANGLE_CORE="${TANGLE_CORE:-0xC9b0716a187072be0f38A5D972392C6479b9Cfe3}"
TSUSD_ADDRESS="${TSUSD_ADDRESS:-0x036CbD53842c5426634e7929541eC2318f3dCF7e}"

echo "=== Voice-Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Deployer:    $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Tangle Core: $TANGLE_CORE"
echo "tsUSD:       $TSUSD_ADDRESS"
echo ""

cd "$(dirname "$0")/../contracts"

# Deploy BSM AND register the blueprint in one forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="$TANGLE_CORE" \
    TSUSD_ADDRESS="$TSUSD_ADDRESS" \
    forge script script/RegisterBlueprint.s.sol \
        --rpc-url "$RPC_URL" \
        --broadcast --slow)

echo "$DEPLOY_OUTPUT"

# Extract the BSM address + blueprint ID for downstream scripts.
BSM_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_INFERENCE_BSM=0x[0-9a-fA-F]+' | tail -1 | cut -d= -f2)
BLUEPRINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_VOICE_BLUEPRINT_ID=[0-9]+' | tail -1 | cut -d= -f2)

if [ -z "$BSM_ADDRESS" ] || [ -z "$BLUEPRINT_ID" ]; then
    echo "ERROR: failed to extract addresses from forge output" >&2
    exit 1
fi

echo ""
echo "=== Blueprint registered ==="
echo "Blueprint ID:  $BLUEPRINT_ID"
echo "InferenceBSM:  $BSM_ADDRESS"
echo ""

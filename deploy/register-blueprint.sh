#!/usr/bin/env bash
# Register the voice-inference blueprint on Tangle.
#
# Two-stage flow:
#   1. forge create InferenceBSM (constructor arg: payment token address)
#   2. cargo tangle blueprint deploy tangle — registers the blueprint via the
#      definition file at `deploy/definition.json` with the freshly-deployed
#      BSM address patched in.
#
# Prerequisites:
#   - forge (Foundry) installed
#   - cargo-tangle CLI installed (`cargo install cargo-tangle`)
#   - jq installed
#   - Deployer wallet funded on the target network
#   - Keystore with the deployer key at ./keystore (or set KEYSTORE_PATH)
#
# Usage (Base Sepolia, against the deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export WS_URL=wss://base-sepolia-rpc.publicnode.com
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export TSUSD_ADDRESS=0x036CbD53842c5426634e7929541eC2318f3dCF7e  # USDC sepolia
#   export KEYSTORE_PATH=./keystore
#   ./deploy/register-blueprint.sh
#
# Optional:
#   BSM_ADDRESS  — skip the forge create step if the BSM is already deployed
#                   (definition.json gets patched with this address instead)

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"
: "${TANGLE_CORE:?Set TANGLE_CORE}"
: "${WS_URL:?Set WS_URL (ws://… or wss://…)}"
: "${KEYSTORE_PATH:=./keystore}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFINITION_FILE="$REPO_ROOT/deploy/definition.json"

echo "=== Voice-Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Deployer:    $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Tangle Core: $TANGLE_CORE"
echo "Definition:  $DEFINITION_FILE"
echo ""

# Stage 1 — Deploy the InferenceBSM if no address was supplied. Voice's BSM
# is a non-upgradeable contract that takes a payment token as its constructor
# argument (the tsUSD wrapper used by the shielded billing flow).
if [ -z "${BSM_ADDRESS:-}" ]; then
    : "${TSUSD_ADDRESS:?Set TSUSD_ADDRESS (payment token) when BSM_ADDRESS is not supplied}"

    echo "Stage 1: deploying InferenceBSM with tsUSD=$TSUSD_ADDRESS …"
    # NOTE: `forge create --json` interleaves compile progress on stdout, so the
    # output is not strictly parseable JSON. Grep the address out of the
    # human-readable line instead — robust to leading compile chatter.
    BSM_ADDRESS=$(forge create \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        --broadcast \
        "$REPO_ROOT/contracts/src/InferenceBSM.sol:InferenceBSM" \
        --constructor-args "$TSUSD_ADDRESS" 2>&1 \
        | grep -oE 'Deployed to: 0x[a-fA-F0-9]{40}' \
        | tail -1 \
        | awk '{print $3}')
    if ! echo "$BSM_ADDRESS" | grep -qE '^0x[a-fA-F0-9]{40}$'; then
        echo "failed to extract BSM addr from forge create output" >&2
        exit 1
    fi
    echo "InferenceBSM deployed at: $BSM_ADDRESS"
else
    echo "Stage 1 skipped — reusing existing BSM at $BSM_ADDRESS"
fi
echo ""

# Stage 2 — Patch deploy/definition.json with the BSM address and call
# cargo-tangle's canonical deploy flow. The patched file is written to a
# temp path so the in-tree file stays untouched (its `manager: 0x0…0` is
# the template).
PATCHED_DEFINITION=$(mktemp --suffix=-voice-blueprint.json)
trap 'rm -f "$PATCHED_DEFINITION"' EXIT
jq --arg mgr "$BSM_ADDRESS" '.manager = $mgr' "$DEFINITION_FILE" > "$PATCHED_DEFINITION"

echo "Stage 2: cargo tangle blueprint deploy tangle …"
cargo tangle blueprint deploy tangle \
    --network testnet \
    --definition "$PATCHED_DEFINITION" \
    --http-rpc-url "$RPC_URL" \
    --ws-rpc-url "$WS_URL" \
    --tangle-contract "$TANGLE_CORE" \
    --keystore-path "$KEYSTORE_PATH"

echo ""
echo "=== Blueprint registered ==="
echo "InferenceBSM: $BSM_ADDRESS"
echo "(blueprint ID is logged by cargo-tangle above)"

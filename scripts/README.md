# Integration checks

The pinned core dependency owns shared protocol fixtures. tcloud owns client SpendAuth signing. Resolve that dependency and run its maintained chain checks from the repository root:

```bash
CORE_DIR="$(cargo metadata --locked --format-version 1 | python3 -c 'import json,sys,pathlib; print(pathlib.Path(next(p["manifest_path"] for p in json.load(sys.stdin)["packages"] if p["name"] == "tangle-inference-core")).parent)')"
bash "$CORE_DIR/scripts/test-full-protocol.sh" "$PWD"
```

Core's protocol tests own their disposable Anvil deployments. [Native registration](../README.md#native-blueprint-registration) uses cargo-tangle and the released TNT manager base.

[The chain job integration](../operator/tests/harness_e2e.rs) exercises the speech adapter through the SDK harness and an HTTP backend peer. Run it with the required Tangle artifacts and verify that it executed. Any live SDK check needs an existing RPC, deployed billing contracts and private configuration from the approved deployment.

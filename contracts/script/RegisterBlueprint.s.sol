// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Script, console2 } from "forge-std/Script.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { InferenceBSM } from "../src/InferenceBSM.sol";

/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}

/// @title RegisterBlueprint
/// @notice Deploys the (non-upgradeable) voice InferenceBSM and registers the
///         voice-inference blueprint on Tangle in a single broadcast.
/// @dev    Run via: `forge script script/RegisterBlueprint.s.sol
///         --rpc-url $RPC_URL --broadcast --slow` from the contracts/ dir.
///
///         Mirrors the pattern proven by the llm-inference / image-gen /
///         training / vector-store / distributed-inference register scripts
///         already shipped against the Base Sepolia Tangle proxy.
contract RegisterBlueprint is Script {
    // ─────────────────────────────────────────────────────────────────────────
    // Defaults — overridable via env vars for non-anvil chains.
    // ─────────────────────────────────────────────────────────────────────────

    // Anvil well-known deployer key (default when no PRIVATE_KEY env is set).
    uint256 constant DEFAULT_DEPLOYER_KEY =
        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;

    // Tangle protocol address on a LocalTestnet anvil snapshot. For real
    // chains (Base Sepolia, mainnet) pass TANGLE_CORE via env.
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;

    // tsUSD shielded wrapper on Base Sepolia (USDC sepolia is used as the
    // underlying for the dev environment). Override via TSUSD_ADDRESS env.
    address constant DEFAULT_TSUSD = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;

    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        address tsUSD = vm.envOr("TSUSD_ADDRESS", DEFAULT_TSUSD);

        ITangle tangle = ITangle(tangleAddr);

        vm.startBroadcast(deployerKey);

        // ── Deploy InferenceBSM (non-upgradeable; constructor(_tsUSD)) ──────
        InferenceBSM bsm = new InferenceBSM(tsUSD);

        // ── Register on Tangle ──────────────────────────────────────────────
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));

        vm.stopBroadcast();

        // ── Output for bash wrapper parsing ─────────────────────────────────
        console2.log("DEPLOY_INFERENCE_BSM=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_VOICE_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Blueprint Definition builder
    // ═════════════════════════════════════════════════════════════════════════

    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/voice-inference-blueprint";
        // metadataHash is a digest of the canonical metadata JSON. Until that
        // payload is pinned via IPFS, derive it from the metadataUri so the
        // value is deterministic + traceable.
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;

        // Event-driven pricing: operators are paid per TTS job rather than on
        // a fixed subscription cadence. The Rust side reads `blueprint.json`
        // (`required_results: 1`, `execution: local`) — that maps to dynamic
        // membership + event-driven pricing here.
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 1,
            maxOperators: 0, // unbounded
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // operators negotiate price per call via RFQ
        });

        def.metadata = Types.BlueprintMetadata({
            name: "Voice Inference Blueprint",
            description: "Voice synthesis (TTS) operator with shielded billing via Tangle",
            author: "Tangle",
            category: "AI/Inference",
            codeRepository: "https://github.com/tangle-network/voice-inference-blueprint",
            logo: "",
            website: "https://tangle.network",
            license: "MIT",
            profilingData: ""
        });

        def.jobs = _buildJobs();

        def.registrationSchema = "";
        def.requestSchema = "";

        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "voice-inference-blueprint",
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(
                Types.BlueprintFetcherKind.None,
                "file:///target/release/voice-inference-blueprint",
                "./target/release/voice-inference-blueprint"
            ),
            testing: Types.TestingSource("voice-inference-blueprint-bin", "voice-inference-blueprint", "."),
            binaries: bins
        });

        def.supportedMemberships = new Types.MembershipModel[](1);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
    }

    function _buildJobs() internal pure returns (Types.JobDefinition[] memory jobs) {
        jobs = new Types.JobDefinition[](1);
        // Job 0: tts
        //   inputs:  (string text, string voice, string format)
        //   outputs: (bytes audio, uint32 sampleRate, string mime)
        // The Rust operator enforces these shapes; on-chain schemas are kept
        // empty to match the pattern used across the sibling inference
        // blueprints — params/result types live with the running operator,
        // not the Blueprint registry.
        jobs[0] = Types.JobDefinition({
            name: "tts",
            description: "Text-to-speech synthesis (input text - audio bytes)",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });
    }
}

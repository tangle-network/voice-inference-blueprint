// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external returns (address) {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        address token = vm.envAddress("TSUSD_ADDRESS");
        // The operator config is the one source of the served model, price and VRAM floor.
        string memory config = vm.readFile(vm.envOr("MODEL_CONFIG", string("../deploy/config.example.json")));
        vm.startBroadcast();
        InferenceBSM manager = new InferenceBSM(token, registry);
        // TTS bills input characters only, so the output price is zero.
        manager.configureModel(
            vm.parseJsonString(config, ".vllm.model"),
            uint32(vm.parseJsonUint(config, ".vllm.max_model_len")),
            uint64(vm.parseJsonUint(config, ".vllm.price_per_1k_chars")),
            0,
            uint32(vm.parseJsonUint(config, ".gpu.min_vram_mib"))
        );
        vm.stopBroadcast();
        console2.log("DEPLOY_MANAGER=%s", vm.toString(address(manager)));
        return address(manager);
    }
}

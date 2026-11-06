// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        address token = vm.envAddress("TSUSD_ADDRESS");
        vm.startBroadcast();
        address manager = address(new InferenceBSM(token, registry));
        vm.stopBroadcast();
        console2.log("DEPLOY_INFERENCE_BSM=%s", vm.toString(manager));
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";
import {Deploy} from "../script/Deploy.s.sol";
import {BlueprintServiceManagerBase} from "tnt-core/BlueprintServiceManagerBase.sol";



contract InferenceBSMTest is Test {
    InferenceBSM public bsm;
    address public tsUSD;

    address public tangleCore = address(0xC0DE);
    address public owner = address(0xBEEF);
    address public operator1 = address(0x1111);
    address public operator2 = address(0x2222);
    address public user = address(0x3333);

    function setUp() public {
        tsUSD = address(0xD00D);
        vm.prank(owner);
        bsm = new InferenceBSM(address(tsUSD), tangleCore);

        // Initialize blueprint
        vm.prank(tangleCore);
        bsm.onBlueprintCreated(1, owner, tangleCore);

        // Configure a model
        vm.prank(owner);
        bsm.configureModel(
            "meta-llama/Llama-3.1-8B-Instruct",
            8192, // max context
            1, // 1 base unit per input token
            2, // 2 base units per output token
            16_000 // 16GB VRAM minimum
        );
    }

    // ─── Initialization ──────────────────────────────────────────────────

    function test_initialization() public view {
        assertEq(bsm.blueprintId(), 1);
        assertEq(bsm.blueprintOwner(), owner);
        assertEq(bsm.tangleCore(), tangleCore);
        assertEq(bsm.tsUSD(), address(tsUSD));
    }

    function test_cannotReinitialize() public {
        vm.expectRevert(BlueprintServiceManagerBase.AlreadyInitialized.selector);
        vm.prank(tangleCore);
        bsm.onBlueprintCreated(2, owner, tangleCore);
    }

    // ─── Model Configuration ─────────────────────────────────────────────

    function test_deployConfiguresModelFromOperatorConfig() public {
        vm.setEnv("TANGLE_CORE", vm.toString(address(this)));
        vm.setEnv("TSUSD_ADDRESS", vm.toString(tsUSD));
        InferenceBSM manager = InferenceBSM(payable(new Deploy().run()));
        manager.onRegister(operator1, abi.encode("Qwen/Qwen3-TTS-1.7B", uint32(1), uint32(16_000), "L4", "https://op"));
        assertTrue(manager.isOperatorActive(operator1));
        (uint64 inputPrice, uint64 outputPrice,) = manager.getOperatorPricing(operator1);
        assertEq(inputPrice, 10);
        assertEq(outputPrice, 0);
    }

    function test_configureModel_onlyOwner() public {
        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyBlueprintOwnerAllowed.selector, user, owner)
        );
        bsm.configureModel("test-model", 4096, 1, 1, 8000);
    }

    function test_disableModel() public {
        vm.prank(owner);
        bsm.disableModel("meta-llama/Llama-3.1-8B-Instruct");

        InferenceBSM.ModelConfig memory mc = bsm.getModelConfig("meta-llama/Llama-3.1-8B-Instruct");
        assertFalse(mc.enabled);
    }

    // ─── Operator Registration ───────────────────────────────────────────

    function test_registerOperator() public {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct", uint32(2), uint32(48_000), "NVIDIA A100", "https://op1.example.com"
        );

        vm.prank(tangleCore);
        bsm.onRegister(operator1, regData);

        assertTrue(bsm.isOperatorActive(operator1));
        assertEq(bsm.getOperatorCount(), 1);
    }

    function test_registerOperator_unsupportedModel() public {
        bytes memory regData =
            abi.encode("nonexistent-model", uint32(1), uint32(16_000), "NVIDIA A100", "https://op1.example.com");

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.ModelNotSupported.selector, "nonexistent-model"));
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_insufficientVram() public {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct",
            uint32(1),
            uint32(8_000), // Only 8GB, need 16GB
            "NVIDIA RTX 3060",
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.InsufficientGpuCapability.selector, 16_000, 8_000));
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_onlyTangle() public {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct", uint32(1), uint32(16_000), "NVIDIA A100", "https://op1.example.com"
        );

        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, user, tangleCore)
        );
        bsm.onRegister(operator1, regData);
    }

    // ─── Unregistration ──────────────────────────────────────────────────

    function test_unregisterOperator() public {
        _registerOperator(operator1);

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertFalse(bsm.isOperatorActive(operator1));
        assertEq(bsm.getOperatorCount(), 0);
    }

    // ─── Service Request ─────────────────────────────────────────────────

    function test_onRequest_validPayment() public {
        address[] memory ops = new address[](0);

        vm.prank(tangleCore);
        bsm.onRequest(1, user, ops, "", 3600, address(tsUSD), 1000);
    }

    function test_onRequest_nativePayment() public {
        address[] memory ops = new address[](0);

        vm.prank(tangleCore);
        bsm.onRequest(1, user, ops, "", 3600, address(0), 0);
    }

    function test_onRequest_invalidPaymentAsset() public {
        address[] memory ops = new address[](0);
        address wrongToken = address(0xDEAD);

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.InvalidPaymentAsset.selector, wrongToken));
        bsm.onRequest(1, user, ops, "", 3600, wrongToken, 1000);
    }

    // ─── Job Lifecycle ───────────────────────────────────────────────────

    function test_onJobCall() public {
        bytes memory inputs = abi.encode("What is Tangle?", uint32(256), uint64(7000));

        vm.prank(tangleCore);
        bsm.onJobCall(1, 0, 1, inputs);
    }

    // ─── Configuration Queries ───────────────────────────────────────────

    function test_paymentAssetAllowed_tsUSD() public {
        // Initialize a service to set up permitted assets
        address[] memory callers = new address[](0);
        vm.prank(tangleCore);
        bsm.onServiceInitialized(1, 1, 1, owner, callers, 3600);

        assertTrue(bsm.queryIsPaymentAssetAllowed(1, address(tsUSD)));
        assertFalse(bsm.queryIsPaymentAssetAllowed(1, address(0xDEAD)));
        assertTrue(bsm.queryIsPaymentAssetAllowed(1, address(0))); // native always allowed
    }

    // ─── Dynamic Membership ─────────────────────────────────────────────

    function test_canJoin_activeOperator() public {
        _registerOperator(operator1);
        assertTrue(bsm.canJoin(1, operator1));
    }

    function test_canJoin_inactiveOperator() public view {
        assertFalse(bsm.canJoin(1, operator1));
    }

    // ─── Operator Pricing Query ────────────────────────────────────────

    function test_getOperatorPricing() public {
        _registerOperator(operator1);

        (uint64 inputPrice, uint64 outputPrice, string memory endpoint) = bsm.getOperatorPricing(operator1);
        assertEq(inputPrice, 1);
        assertEq(outputPrice, 2);
        assertEq(keccak256(bytes(endpoint)), keccak256(bytes("https://operator.example.com")));
    }

    function test_getOperatorPricing_unregistered() public {
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.OperatorNotRegistered.selector, operator1));
        bsm.getOperatorPricing(operator1);
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    function _registerOperator(address op) internal {
        bytes memory regData = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct", uint32(2), uint32(48_000), "NVIDIA A100", "https://operator.example.com"
        );
        vm.prank(tangleCore);
        bsm.onRegister(op, regData);
    }
}

from __future__ import annotations

import unittest
from collections import Counter

from benchmarks.formal_data import load_semantic_cases
from benchmarks.formal_protocol import (
    BMC_COUNTEREXAMPLE_DEPTH,
    INDEPENDENT_SIMULATION_CYCLES,
    INDEPENDENT_SIMULATION_SEEDS,
    FormalResult,
    FormalStatus,
    InitializationContract,
    compatible_interfaces,
    generate_equivalence_wrapper,
    initialization_contract,
    normalize_prob151,
    parse_module,
    rename_top_module,
)


class ProtocolAuditTest(unittest.TestCase):
    def test_case_classification_matches_preregistered_audit(self):
        cases = load_semantic_cases()
        kinds = Counter(
            initialization_contract(case.seed_id, case.canonical_rtl).kind
            for case in cases
        )
        self.assertEqual(kinds, {"combinational": 17, "reset": 60, "preamble": 8})
        reset_styles = Counter(
            initialization_contract(case.seed_id, case.canonical_rtl).reset_synchronous
            for case in cases
            if initialization_contract(case.seed_id, case.canonical_rtl).kind == "reset"
        )
        self.assertEqual(reset_styles, {True: 37, False: 23})

    def test_frozen_independent_simulation_and_bmc_contract(self):
        self.assertEqual(INDEPENDENT_SIMULATION_SEEDS, tuple(range(1001, 1011)))
        self.assertEqual(INDEPENDENT_SIMULATION_CYCLES, 200)
        self.assertEqual(BMC_COUNTEREXAMPLE_DEPTH, 256)

    def test_prob151_is_the_only_deterministic_normalization(self):
        cases = [
            case for case in load_semantic_cases()
            if case.seed_id == "Prob151_review2015_fsm_ref"
        ]
        source = cases[0].canonical_rtl
        normalized, changed = normalize_prob151("Prob151_review2015_fsm_ref", source)
        self.assertTrue(changed)
        self.assertNotIn("States'(", normalized)
        self.assertIn("localparam logic [3:0]", normalized)
        self.assertNotIn("|state === 1'bx", normalized)
        other, changed = normalize_prob151("Other", source)
        self.assertFalse(changed)
        self.assertEqual(other, source)
        # The exact three known mutants and an anonymous module-name rendering
        # are the only additional audited source identities.
        for case in cases:
            _, changed = normalize_prob151(case.seed_id, case.broken_rtl)
            self.assertTrue(changed)
        anonymous, _ = rename_top_module(source, "TopModule")
        _, changed = normalize_prob151("Prob151_review2015_fsm_ref", anonymous)
        self.assertTrue(changed)

    def test_prob151_normalization_fails_closed_on_behavioral_source_drift(self):
        source = next(
            case.canonical_rtl
            for case in load_semantic_cases()
            if case.seed_id == "Prob151_review2015_fsm_ref"
        )
        drifted = source.replace("B0: next = B1;", "B0: next = B2;")
        self.assertNotEqual(drifted, source)
        with self.assertRaisesRegex(ValueError, "outside the audited"):
            normalize_prob151("Prob151_review2015_fsm_ref", drifted)

    def test_prob151_complete_rewrite_without_cast_or_x_guard_is_not_normalized(self):
        rewrite = "module TopModule(input a, output y); assign y = a; endmodule\n"
        normalized, changed = normalize_prob151(
            "Prob151_review2015_fsm_ref", rewrite
        )
        self.assertFalse(changed)
        self.assertEqual(normalized, rewrite)

    def test_prob046_preserves_its_real_falling_clock_edge(self):
        case = next(
            case for case in load_semantic_cases() if case.seed_id == "Prob046_dff8p_ref"
        )
        contract = initialization_contract(case.seed_id, case.canonical_rtl)
        self.assertEqual(contract.clock_edge, "negedge")
        wrapper = generate_equivalence_wrapper(parse_module(case.canonical_rtl), contract)
        self.assertIn("always @(negedge clk)", wrapper)

    def test_prob154_masks_payload_only_while_public_done_is_low(self):
        case = next(
            case for case in load_semantic_cases()
            if case.seed_id == "Prob154_fsm_ps2data_ref"
        )
        contract = initialization_contract(case.seed_id, case.canonical_rtl)
        wrapper = generate_equivalence_wrapper(
            parse_module(case.canonical_rtl), contract, seed_id=case.seed_id
        )
        self.assertIn("done_golden == done_candidate", wrapper)
        self.assertIn(
            "(!done_golden) || (out_bytes_golden == out_bytes_candidate)", wrapper
        )


class HarnessTest(unittest.TestCase):
    def test_sequential_assertion_is_continuous_after_preamble(self):
        source = """module D(input clk, input reset, input d, output reg q);
always @(posedge clk) if (reset) q <= 0; else q <= d;
endmodule
"""
        info = parse_module(source)
        contract = initialization_contract("D", source)
        wrapper = generate_equivalence_wrapper(info, contract)
        self.assertIn("formal_step >= 2", wrapper)
        self.assertIn("always @* if", wrapper)
        self.assertIn("assume (reset == 1'b1)", wrapper)

    def test_named_port_reordering_is_compatible(self):
        first = parse_module("module A(input a, input b, output y); endmodule")
        second = parse_module("module B(output y, input b, input a); endmodule")
        self.assertEqual(compatible_interfaces(first, second), (True, ""))

    def test_formal_result_json_roundtrip(self):
        contract = InitializationContract("combinational", "none")
        result = FormalResult(
            FormalStatus.PROVED,
            "sat",
            contract,
            0.1,
            "c",
            "g",
        )
        self.assertEqual(FormalResult.from_dict(result.to_dict()), result)


if __name__ == "__main__":
    unittest.main()

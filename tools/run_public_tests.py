#!/usr/bin/env python3
"""Run every test that is satisfiable from the public standalone release.

Four protocol-amendment tests intentionally bind private, pre-amendment files
that are not redistributed.  They remain in the source suite and run under
``make test-full`` when those historical files are available.  This runner
excludes only their exact unittest IDs and fails if that expected set drifts.
"""

from __future__ import annotations

import sys
import unittest


PRIVATE_BINDING_TESTS = {
    "test_vericodegen_eval.GateAndConfigTests.test_calibration_gate_is_exact_and_fail_closed",
    "test_vericodegen_eval.GateAndConfigTests.test_calibration_gate_rejects_stale_protocol_or_image_provenance",
    "test_vericodegen_eval.GateAndConfigTests.test_calibration_gate_rejects_subset_and_results_byte_drift",
    "test_vericodegen_eval.GateAndConfigTests.test_checked_in_protocol_amendment_and_redaction_drop_are_valid",
}


def flatten(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main() -> int:
    discovered = unittest.defaultTestLoader.discover(
        "tests",
        pattern="test_*.py",
        top_level_dir=None,
    )
    tests = list(flatten(discovered))
    by_id = {test.id(): test for test in tests}
    missing = sorted(PRIVATE_BINDING_TESTS - set(by_id))
    if missing:
        print(
            "public-test exclusion set drifted; expected test IDs are missing: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    public = unittest.TestSuite(
        test for test in tests if test.id() not in PRIVATE_BINDING_TESTS
    )
    print(
        f"public standalone suite: {public.countTestCases()} tests; "
        f"{len(PRIVATE_BINDING_TESTS)} exact private-binding tests excluded",
        flush=True,
    )
    result = unittest.TextTestRunner(verbosity=2).run(public)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

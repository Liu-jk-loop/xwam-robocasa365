from __future__ import annotations

import unittest

from project_tools.starlight_environment import evaluate_packages, version_key, version_satisfies


class StarlightEnvironmentTest(unittest.TestCase):
    def test_version_key_handles_cuda_and_build_suffixes(self) -> None:
        self.assertEqual(version_key("2.8.0+cu128"), (2, 8, 0, 0))
        self.assertEqual(version_key("4.51.3.dev0"), (4, 51, 3, 0))

    def test_version_bounds(self) -> None:
        spec = {"min": "1.23.5", "max_exclusive": "1.26.0"}
        self.assertTrue(version_satisfies("1.23.5", spec)[0])
        self.assertTrue(version_satisfies("1.25.2", spec)[0])
        self.assertFalse(version_satisfies("1.22.4", spec)[0])
        self.assertFalse(version_satisfies("1.26.0", spec)[0])

    def test_inclusive_upper_bound(self) -> None:
        spec = {"min": "4.49.0", "max_inclusive": "4.51.3"}
        self.assertTrue(version_satisfies("4.51.3", spec)[0])
        self.assertFalse(version_satisfies("4.52.0", spec)[0])

    def test_package_evaluation_separates_errors_and_warnings(self) -> None:
        versions = {"torch": "2.8.0+cu128", "ninja": None, "numpy": "1.26.1"}
        specs = [
            {"distribution": "torch", "module": "torch", "min": "2.4.0", "tested": "2.8.0"},
            {"distribution": "ninja", "module": "ninja", "required": False},
            {"distribution": "numpy", "module": "numpy", "max_exclusive": "1.26.0"},
        ]
        results, errors, warnings = evaluate_packages(
            specs,
            version_lookup=versions.get,
            run_imports=False,
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(any("numpy" in error for error in errors))
        self.assertTrue(any("ninja" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()

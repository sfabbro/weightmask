"""The MegaCam science gate skips when its data directory is empty.

A skip is not a pass, and this test does not run the MegaCam benches.
"""

import os
import tempfile
import unittest

from benchmarks.science_gate import main


class TestScienceGateSkip(unittest.TestCase):
    def test_missing_fits_is_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "science-gate.md")
            code = main(["--data-dir", tmp, "--out", out])
            self.assertEqual(code, 0)
            text = open(out).read()
        self.assertIn("status: skipped", text)
        self.assertIn("not a pass", text)
        self.assertNotIn("status: passed", text)


if __name__ == "__main__":
    unittest.main()

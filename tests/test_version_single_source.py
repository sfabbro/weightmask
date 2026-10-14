"""The package version has exactly one source of truth.

It used to be written out in five places: ``pyproject.toml``,
``weightmask/__init__.py``, ``weightmask/contract.py``, ``tests/test_contract.py``
and ``docs/api.md``. A release had to update all five, and the one in
``contract.py`` is the dangerous one -- it stamps provenance into every output
map, so a missed update silently writes a stale version into science products.

Now ``weightmask/_version.py`` holds it and everything imports from there. The
pyproject version is asserted to match, because a packaging tool cannot read a
Python module for the project table, so that one duplication is checked rather
than trusted.

``docs/api.md`` is generated-facing prose and is checked textually below, since a
stale version quoted in the docs is a small but real trap for a reader.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SOURCE = REPO / "weightmask" / "_version.py"
INIT = REPO / "weightmask" / "__init__.py"
PYPROJECT = REPO / "pyproject.toml"
API_DOC = REPO / "docs" / "api.md"


def _declared_version() -> str:
    import weightmask

    return weightmask.__version__


class TestSingleSourceOfVersion(unittest.TestCase):
    def test_version_is_declared_once(self):
        self.assertTrue(SOURCE.exists(), "weightmask/_version.py must exist as the single declaration")
        text = SOURCE.read_text()
        self.assertEqual(
            len(re.findall(r'__version__\s*=\s*"', text)),
            1,
            "_version.py must declare __version__ exactly once",
        )

    def test_pyproject_matches_the_module(self):
        declared = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.M)
        self.assertIsNotNone(declared, "pyproject.toml must pin a version")
        self.assertEqual(
            declared.group(1),
            _declared_version(),
            "pyproject.toml and weightmask must agree; packaging cannot read the module for us",
        )

    def test_contract_stamps_the_module_version_not_a_literal(self):
        """The dangerous copy: this one reaches science products."""
        import inspect

        from weightmask.contract import ProducerMetadata

        self.assertIsNotNone(ProducerMetadata().version)
        source = inspect.getsource(inspect.getmodule(ProducerMetadata))
        self.assertIsNone(
            re.search(r'version:\s*str\s*=\s*"[\d.]+"', source),
            "ProducerMetadata.version must be derived, not a hardcoded literal",
        )

    def test_init_reads_the_single_source(self):
        text = INIT.read_text()
        self.assertIn("_version", text, "__init__ should import from weightmask._version")
        self.assertNotRegex(
            text,
            r'__version__\s*=\s*"[\d.]+"',
            "__init__ must not carry its own literal version",
        )

    def test_docs_quote_the_current_version(self):
        quoted = set(re.findall(r"`(\d+\.\d+\.\d+)`", API_DOC.read_text()))
        self.assertIn(
            _declared_version(),
            quoted,
            f"docs/api.md quotes {sorted(quoted)} but the package is {_declared_version()}",
        )


if __name__ == "__main__":
    unittest.main()

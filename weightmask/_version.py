"""The single declaration of the package version.

Everything that reports a version imports from here: ``weightmask.__version__``
and ``ProducerMetadata.version``, which stamps every output map. The copy in
``pyproject.toml`` cannot be avoided -- packaging cannot read a Python module
for the project table -- so ``tests/test_version_single_source.py`` asserts the
two agree rather than trusting a human to update both.
"""

__version__ = "0.2.0"

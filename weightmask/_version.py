"""The runtime package version.

Everything that reports a version imports from here: ``weightmask.__version__``
and ``ProducerMetadata.version``, which stamps every output map. The copy in
``pyproject.toml`` is declared separately for build metadata;
``tests/test_version_single_source.py`` asserts the two agree.
"""

__version__ = "0.2.1"

# API

Supported Python imports are the package root names in `weightmask.__all__`,
the names listed below in `weightmask.process`, `weightmask.contract`,
`weightmask.config`, `weightmask.background`, `weightmask.reconstruct_sky`, and
`weightmask.torchfits_adapter`. Other modules and names are callable internals,
not a stability promise. Science methods are in [algorithms.md](algorithms.md).
Config keys are in the canonical [`weightmask.yml`](../weightmask.yml).

## `weightmask.config`

The canonical YAML is packaged with wheel and sdist artifacts. Use the resource
directly or copy it to a writable working directory:

```python
from weightmask.config import (
    clean_config_dict,
    copy_default_config,
    default_config_bytes,
    default_config_path,
    default_config_resource,
    default_config_sha256,
    default_config_text,
)

resource = default_config_resource()
copy_default_config("weightmask.yml")
config = clean_config_dict({"output_params": {"compress": "true"}})
```

`default_config_resource()` returns an `importlib.resources.abc.Traversable`.
`default_config_path()` is a context manager for a temporary filesystem path
when the package is stored in an archive. `copy_default_config()` uses exclusive
creation by default, rejects existing or dangling symlinks, and never follows a
target symlink. With `overwrite=True`, an existing regular file is replaced
atomically; symlink targets are still rejected.
`default_config_bytes()`, `default_config_text()`, and
`default_config_sha256()` expose the packaged bytes, UTF-8 text, and digest.
`clean_config_dict()` converts YAML scalar strings to their typed values before
validation.

The reconstruction and torchfits names are module imports, not package-root
re-exports. In particular, `weightmask.sky_to_mesh` and
`weightmask.TorchfitsArrayHeaderIO` are not supported paths.

## `weightmask.process`

```python
from weightmask.process import process_image, validate_config

if not validate_config(config):
    raise ValueError("Invalid weightmask configuration")
mask, ivar, weight, confidence, sky, header_info = process_image(data, hdr, flat_data, config, tile_size=1024)
```

Single-array entry: no dark, keep-map, or MEF dead-CCD veto. Pass a real flat
or flat-based `BAD` is skipped (`F = 1`). Use the CLI (`weightmask.cli.run_pipeline`)
for those extra `BAD` sources and MEF orchestration.

`process_image()` returns `(mask, ivar, weight, confidence, sky, header_info)`
(or a 6-tuple of `None` on failure), where `header_info` includes:

| Key | Contents |
|---|---|
| `individual_masks` | Component boolean maps (`bad`, `sat`, `cr`, `obj`, `streak`, `nodata`) |
| `contract_product` | `WeightMaskProduct` |
| `sky_cards` | `SKYMESH` header cards when `output_params.sky_format: mesh` |
| `timings` | Per-stage wall-clock seconds |

Science input must be a nonempty 2-D array; supplied bad masks must match its
shape and are interpreted as boolean pixel masks. Invalid shapes raise
`ValueError`. Non-finite science pixels receive `NO_DATA` and zero weight.
`tile_size` caps flat-mask tiles, with smaller tiles on small images.

## Bits and polarity

```python
from weightmask import MASK_BITS, QUALITY_BITS, MASK_DTYPE
```

`MASK_BITS` is an alias of `QUALITY_BITS`:

| Name | Value |
|---|---|
| `BAD` | 1 |
| `SAT` | 2 |
| `CR` | 4 |
| `DETECTED` | 8 |
| `STREAK` | 16 |
| `INVALID_VARIANCE` | 32 |
| `NO_DATA` | 64 |

Polarity is `set_means_flagged` (`weightmask.contract.MASK_POLARITY`).
`DETECTED` does not zero weight unless `output_params.mask_detected_in_weight`
is true. `MASK_DTYPE` is `"uint32"` in memory; FITS masks are written as
uint16 (`output_params.mask_bitpix: 16`) because values 0–127 fit.

`weightmask.__version__` imports the runtime version from `weightmask/_version.py`.
Packaging also declares it in `pyproject.toml`; `tests/test_version_single_source.py`
checks that the two values agree.

## `weightmask.contract`

```python
from weightmask.contract import (
    WeightMaskProduct,
    build_weight_product,
    QUALITY_BITS,
    MASK_POLARITY,
    INVERSE_VARIANCE_SEMANTICS,
)
```

`build_weight_product(inverse_variance, quality_mask=None, *, exclude_detected=False, confidence_percentile=99.0, producer=None, provenance=None)`
returns a `WeightMaskProduct` with quality flags, non-negative inverse variance
and weight. Confidence from this function is always in `[0, 1]`; the CLI
product may then multiply by 100 if `confidence_params.scale_to_100` is true.
Non-finite or non-positive inverse variance, including overflow or underflow
when converted to float32, is marked `INVALID_VARIANCE` and zeroed.
`INVERSE_VARIANCE_SEMANTICS` is `"inverse_variance_adu^-2"`; the estimator and
flat convention are selected in `variance` (see [algorithms.md](algorithms.md)).
Missing provenance is allowed; malformed or truncated provenance warns and
reports an unknown producer.
`CONTRACT_VERSION` (`"1.0"`) is the array-schema version; the package version
is `weightmask.__version__` (`0.2.1`).

`ArrayHeaderIO` is an optional read/write protocol at
`weightmask.contract.ArrayHeaderIO`. The optional adapter is exposed at
`weightmask.torchfits_adapter`:

```python
from weightmask.torchfits_adapter import (
    TorchfitsArrayHeaderIO,
    TorchfitsUnavailableError,
    torchfits_available,
)
```

`TorchfitsArrayHeaderIO` implements `ArrayHeaderIO` when torchfits and torch
are installed; neither is a required dependency. `torchfits_available()` is
the supported availability probe, and constructing or using the adapter raises
`TorchfitsUnavailableError` when the optional dependencies are unavailable.

## `weightmask.reconstruct_sky`

Module and CLI for compact sky meshes (`output_params.sky_format: mesh`).

```python
from weightmask.reconstruct_sky import reconstruct_sky_fits
from weightmask.background import (
    parse_sky_mesh_header,
    reconstruct_sky_from_header,
    reconstruct_sky_mesh,
    sky_to_mesh,
)
```

`reconstruct_sky_fits(input_path, output_path, hdu=None)` rebuilds a FITS file
and returns an exit code. Input/output aliases are rejected. Node layout: `n = (size - 1) // box + 1` at
`clip(rint((k + 0.5) * box), 0, size - 1)`; reconstruction is a natural cubic spline. See
[algorithms.md](algorithms.md).

`sky_to_mesh`, `reconstruct_sky_mesh`, `parse_sky_mesh_header`, and
`reconstruct_sky_from_header` are the supported array/header helpers in
`weightmask.background`. The CLI entry point is
`weightmask.reconstruct_sky.main`, installed as
`weightmask-reconstruct-sky`.

## Console scripts

| Script | Entry |
|---|---|
| `weightmask` | `weightmask.cli:run_pipeline` |
| `weightmask-reconstruct-sky` | `weightmask.reconstruct_sky:main` |

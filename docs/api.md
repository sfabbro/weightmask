# API

Supported public surface: package `__all__`, `weightmask.contract`,
`weightmask.process`, `weightmask.reconstruct_sky`, and the two console scripts.
Other modules are callable internals, not a stability promise. Science methods
are in [algorithms.md](algorithms.md). Config keys are in
[`weightmask.yml`](../weightmask.yml).

## `weightmask.process`

```python
from weightmask.process import process_image, validate_config

if not validate_config(config):
    raise ValueError("Invalid weightmask configuration")
mask, ivar, weight, confidence, sky, header_info = process_image(
    data, hdr, flat_data, config, tile_size=1024
)
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

`ArrayHeaderIO` is an optional read/write protocol. `TorchfitsArrayHeaderIO`
implements it when torchfits is installed; torchfits is not a required
dependency.

## `weightmask.reconstruct_sky`

Module and CLI for compact sky meshes (`output_params.sky_format: mesh`).

```python
from weightmask.reconstruct_sky import reconstruct_sky_fits
from weightmask.background import reconstruct_sky_mesh, reconstruct_sky_from_header
```

`reconstruct_sky_fits(input_path, output_path, hdu=None)` rebuilds a FITS file
and returns an exit code. Input/output aliases are rejected. Node layout: `n = (size - 1) // box + 1` at
`clip(rint((k + 0.5) * box), 0, size - 1)`; reconstruction is a natural cubic spline. See
[algorithms.md](algorithms.md).

## Console scripts

| Script | Entry |
|---|---|
| `weightmask` | `weightmask.cli:run_pipeline` |
| `weightmask-reconstruct-sky` | `weightmask.reconstruct_sky:main` |

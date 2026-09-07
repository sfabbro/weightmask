# API (0.1)

Supported public surface: package `__all__`, `weightmask.contract`,
`weightmask.reconstruct_sky`, and the two console scripts. Other modules are
callable internals, not a stability promise. Science methods are in
[algorithms.md](algorithms.md). Config keys are in
[`weightmask.yml`](../weightmask.yml).

## `weightmask.WeightMapGenerator`

```python
from weightmask import WeightMapGenerator

gen = WeightMapGenerator(config)          # raises ValueError if config is invalid
out = gen.process(data, header=None, flat_data=None, tile_size=1024)
```

`process()` returns a dict:

| Key | Contents |
|---|---|
| `weight_map` | Masked inverse variance |
| `flag_map` | Integer quality mask |
| `inv_variance_map` | Inverse-variance plane |
| `confidence_map` | Percentile-normalized weight in `[0, 1]` |
| `sky_map` | Background map |
| `individual_masks` | Component boolean maps (`bad`, `sat`, `cr`, `obj`, `streak`) |
| `contract_product` | `WeightMaskProduct` (when the contract path ran) |
| `artifact_metadata` | Metadata from that product |

Empty dict if processing failed.

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

Polarity is `set_means_flagged` (`weightmask.contract.MASK_POLARITY`).
`DETECTED` does not zero weight unless `output_params.mask_detected_in_weight`
is true. `MASK_DTYPE` is `"uint32"` in memory; FITS masks are written as
uint16 (`output_params.mask_bitpix: 16`) because values 0–63 fit.

`weightmask.__version__` is the installed package version (fallback `"0.1.0"`).

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
and weight, and confidence in `[0, 1]`. Non-finite or non-positive inverse
variance is marked `INVALID_VARIANCE` and zeroed. Inverse-variance semantics
are `elixir_style_flat2_coadd_weight`.

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
and returns an exit code. Node layout: `n = (size - 1) // box + 1` at clipped
`(k + 0.5) * box`; reconstruction is a natural cubic spline. See
[algorithms.md](algorithms.md).

## Console scripts

| Script | Entry |
|---|---|
| `weightmask` | `weightmask.cli:run_pipeline` |
| `weightmask-reconstruct-sky` | `weightmask.reconstruct_sky:main` |

`weightmask reconstruct-sky ...` is a thin compatibility dispatch to the
second program. It is not a `weightmask` flag.

# Examples

Synthetic end-to-end run of the WeightMask pipeline.

## Run

From the repository root, with the pixi environment:

```bash
pixi run example-complex
```

That executes `complex_simulation_example.py`: a Poisson-plus-read-noise image
with a spatially varying sky, a variable PSF, and dashed satellite tracks.
It writes FITS products under `test_outputs/` (gitignored) and prints
precision/recall against the injected truth.

`cfht_megaprime_example.py` is a thinner MegaPrime-like walkthrough.
`real_world_robustness.py` and `test_real_mef.py` are extra scripts, not the
default pixi task.

## Quality bits

Bits match the package contract. See the table in the root README and the
per-artefact methods in [docs/algorithms.md](../docs/algorithms.md).

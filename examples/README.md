# Examples

Synthetic detection demo (mask bits vs injected truth), not a full
weight/invvar/sky product run.

## Run

From the repository root, with the pixi environment:

```bash
pixi run example-complex
```

That executes `complex_simulation_example.py`: a Poisson-plus-read-noise image
with a spatially varying sky, a variable PSF, and dashed satellite tracks.
It writes mask FITS under `test_outputs/` (gitignored) and prints
precision/recall against the injected truth.

`cfht_megaprime_example.py` is a thinner MegaPrime-like walkthrough.
`real_world_robustness.py` is the same staged detector test on a denser field.
`test_real_mef.py` needs external MegaCam files under `benchmark_data/`.

## Quality bits

Bits match the package contract. See the table in the root README and the
per-artefact methods in [docs/algorithms.md](../docs/algorithms.md).

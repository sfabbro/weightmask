# Weightmask Examples

This directory contains standalone examples demonstrating the capabilities of the `weightmask` library, particularly its robustness in complex astronomical regimes.

## Available Examples

### 1. `cfht_megaprime_example.py`
Demonstrates basic usage on data similar to CFHT MegaPrime images.

### 2. `complex_simulation_example.py`
**Recommended for seeing new features.**
This script generates a synthetic image with:
- **Spatially Variable Background**: Undulating sky levels.
- **Poisson + Read Noise Model**: Realistic noise statistics.
- **Variable PSF**: Star sharpness changing across the field.
- **Complex Streaks**: Tumbling satellite tracks (dashed/dotted).

It runs the full `weightmask` pipeline and produces a summary of precision and recall for different artifacts.

## How to Run

Ensure you are in the project root and have the virtual environment activated:

```bash
pixi run example-complex
```

## What it Produces

The examples typically output:
1. **FITS files**: The original simulated image and the final mask bits.
2. **Terminal logs**: Detailed steps of the detection pipeline (background subtraction, satdet-style Hough candidates, strip refinement, RANSAC fallback, etc.).
3. **Metrics**: Precision and Recall against the "ground truth" simulation model.

Generated example products are written to `test_outputs/` and are ignored by git.

---

## Mask Bit Definitions

The output masks use a bitmask system:
- `1`: Bad Pixels (Flat outliers)
- `2`: Saturation (Saturation + Bleed Trails)
- `4`: Cosmic Rays
- `8`: Detected Objects (Stars/Galaxies)
- `16`: Streaks (Satellites/Meteors)

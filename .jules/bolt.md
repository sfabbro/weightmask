
## 2024-05-18 - Avoid Python loops for pixel masking
**Learning:** Pure Python iteration to draw thousands of ellipses using `skimage.draw.ellipse` creates a significant bottleneck on dense star fields. Vectorized C-extensions like `sep.mask_ellipse` offer ~50-100x speedups.
**Action:** Always prefer C-level vectorized operations (`sep.mask_ellipse`, `skimage.draw.polygon2mask`, etc.) over manual pixel-level `for` loops when generating masks in Python.

## 2024-05-18 - Pre-clean Configuration Dicts
**Learning:** Parsing numeric and boolean values out of a configuration dictionary (e.g. from YAML strings) inside a frequently called function (`detect_objects`) introduces unnecessary overhead. For dictionaries with 10 keys, this repeated casting took ~0.94 seconds for 100k calls, compared to ~0.0067 seconds when pre-cleaned.
**Action:** Always parse and clean configurations at the entry point of the application (e.g. `cli.py` immediately after file load) using a recursive utility function. Do not perform dictionary type-casting inside performance-critical processing pipelines.

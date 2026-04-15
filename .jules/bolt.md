
## 2024-05-18 - Avoid Python loops for pixel masking
**Learning:** Pure Python iteration to draw thousands of ellipses using `skimage.draw.ellipse` creates a significant bottleneck on dense star fields. Vectorized C-extensions like `sep.mask_ellipse` offer ~50-100x speedups.
**Action:** Always prefer C-level vectorized operations (`sep.mask_ellipse`, `skimage.draw.polygon2mask`, etc.) over manual pixel-level `for` loops when generating masks in Python.
## 2024-05-18 - [CLI Config Fallback Testing]
**Learning:** Learned that `cli.py` has a fallback logic for config paths. Need to test each fallback path with mocks ensuring testing without filesystem side effects.
**Action:** Use `unittest.mock.patch('os.path.exists')` and mock `sys.argv` to effectively verify fallback logics.

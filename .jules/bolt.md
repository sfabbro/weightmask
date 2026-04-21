
## 2024-05-18 - Avoid Python loops for pixel masking
**Learning:** Pure Python iteration to draw thousands of ellipses using `skimage.draw.ellipse` creates a significant bottleneck on dense star fields. Vectorized C-extensions like `sep.mask_ellipse` offer ~50-100x speedups.
**Action:** Always prefer C-level vectorized operations (`sep.mask_ellipse`, `skimage.draw.polygon2mask`, etc.) over manual pixel-level `for` loops when generating masks in Python.

## 2026-04-12 - Replaced slow skimage.draw.ellipse with fast sep.mask_ellipse
**Learning:** `skimage.draw.ellipse` was used in `weightmask/objects.py` in a tight loop to draw masks for up to 100k objects detected by SEP. This loop can take over 12 seconds in python, whereas SEP has a built-in vectorized C-extension function `sep.mask_ellipse` that accomplishes the exact same thing in ~0.2s.
**Action:** Replaced the for loop and `skimage.draw.ellipse` call with `sep.mask_ellipse(object_mask_ui8, objects['x'], objects['y'], scaled_a, scaled_b, objects['theta'], r=base_k)` (with boolean mask + write-back where appropriate).

## 2026-04-14 - Subsampling for robust statistics
**Learning:** Calculating `np.median` or `np.percentile` on full high-resolution image arrays (e.g., 4000x4000) for global background or MAD estimates can introduce significant delays (>0.3s per call) without yielding more accurate global statistics.
**Action:** Always subsample large arrays (e.g., `data[::10, ::10]` or `data.ravel()[::step]`) when calculating global robust statistics like median or MAD to achieve massive speedups.

## 2024-05-18 - [Vectorize region growing]
**Learning:** Pixel-by-pixel Python `for` loops in image processing algorithms like bleed trail growth cause massive performance bottlenecks.
**Action:** Replace sequential boundary limit checks with NumPy vectorized operations (slicing and `np.argmin` on boolean arrays) to determine termination points instantly. Reverse slicing can be elegantly handled using a `-1` end index condition to capture the 0th array element cleanly.

## 2024-05-18 - Mocking Context Managers with Dunder Methods
**Learning:** When mocking `fitsio.FITS` used as a context manager (e.g. `with fitsio.FITS(...) as f`), it is necessary to mock both the `__enter__` return value and any specific dunder methods used within the block (like `__len__` for checking file length).
**Action:** Use `mock_fits.return_value.__enter__.return_value = mock_fits_instance` and `mock_fits_instance.__len__.return_value = ...` to properly simulate different FITS file states (empty vs. non-empty).

## 2024-05-18 - Pre-clean Configuration Dicts
**Learning:** Parsing numeric and boolean values out of a configuration dictionary (e.g. from YAML strings) inside a frequently called function (`detect_objects`) introduces unnecessary overhead. For dictionaries with 10 keys, this repeated casting took ~0.94 seconds for 100k calls, compared to ~0.0067 seconds when pre-cleaned.
**Action:** Always parse and clean configurations at the entry point of the application (e.g. `cli.py` immediately after file load) using a recursive utility function. Do not perform dictionary type-casting inside performance-critical processing pipelines.

## 2024-05-18 - [CLI Config Fallback Testing]
**Learning:** Learned that `cli.py` has a fallback logic for config paths. Need to test each fallback path with mocks ensuring testing without filesystem side effects.
**Action:** Use `unittest.mock.patch('os.path.exists')` and mock `sys.argv` to effectively verify fallback logics.

## 2024-05-19 - Replace pure Python loops with NumPy 2D array slicing
**Learning:** Pure Python iteration to assign array elements row by row or column by column within a bounding box introduces noticeable overhead. Native C-level NumPy 2D slicing can directly assign values to subarrays, achieving ~2x speedups.
**Action:** Always prefer 2D NumPy array slicing over manual Python `for` loops when assigning values to regular geometric regions like rectangles or lines within an image/mask.

## 2023-10-25 - Parallelizing heavy skimage operations
**Learning:** Nested loops on image blocks mapping sequentially to expensive skimage functions (like `skimage.filters.frangi`) can be a significant bottleneck. Python's `concurrent.futures.ThreadPoolExecutor` handles block parallelization very efficiently here, dropping the filter time significantly (e.g. 4x speedup).
**Action:** Always parallelize block processing on image data using `ThreadPoolExecutor` or `ProcessPoolExecutor` where operations on blocks are independent.
## 2024-05-24 - Vectorize spatial patch processing with NaN functions
**Learning:** Nested Python coordinate loops for processing spatial image patches are exceptionally slow on large arrays (e.g. 10000x10000 pixels).
**Action:** When calculating patch statistics (like median or MAD), truncate dimensions to perfectly fit the patch size, reshape and transpose the arrays into `(num_patches, patch_size * patch_size)`, replace masked pixels with `np.nan` using `np.where`, and apply vectorized functions like `np.nanmedian` along the axis. Process any remaining non-divisible edge patches using a minimal fallback loop.

## 2024-05-24 - Avoid 1D looping for contiguous segments
**Learning:** Looping over 1D arrays to find contiguous segments (e.g., column-by-column saturation masking) using `scipy.ndimage.label` incurs massive Python overhead when there are many features.
**Action:** Perform a single 2D `scipy.ndimage.label` on a sliced 2D subset using a strict directional structuring element (like a vertical 3x3 array `[[0, 1, 0], [0, 1, 0], [0, 1, 0]]` for columns), followed by `scipy.ndimage.find_objects` for efficient bounding box retrieval.
## 2024-05-18 - Extract loop invariants from hot-path algorithms
**Learning:** Performing dictionary lookups (`config.get`) and allocating fallback arrays (`np.zeros`) inside hot-path inner loops (such as iterating over saturated segments in `grow_bleed_trails`) introduces redundant overhead.
**Action:** Always extract invariant configuration lookups and pre-allocate fallback arrays outside the loop to improve execution speed in performance-critical sections.

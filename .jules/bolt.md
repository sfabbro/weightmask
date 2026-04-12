## 2026-04-12 - Replaced slow skimage.draw.ellipse with fast sep.mask_ellipse
**Learning:** `skimage.draw.ellipse` was used in `weightmask/objects.py` in a tight loop to draw masks for up to 100k objects detected by SEP. This loop can take over 12 seconds in python, whereas SEP has a built-in vectorized C-extension function `sep.mask_ellipse` that accomplishes the exact same thing in ~0.2s.
**Action:** Replaced the for loop and `skimage.draw.ellipse` call with `sep.mask_ellipse(object_mask_ui8, objects['x'], objects['y'], scaled_a, scaled_b, objects['theta'], r=base_k)`

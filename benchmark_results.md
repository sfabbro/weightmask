# Weightmask Benchmark Report

| Regime | Saturation P | Saturation R | Saturation F1 | Cosmics P | Cosmics R | Cosmics F1 | Objects P | Objects R | Objects F1 | Streaks P | Streaks R | Streaks F1 | CR-Star Overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ideal (Low Noise, Sparse) | 0.168 | 1.000 | 0.288 | 1.000 | 0.981 | 0.990 | 0.272 | 0.867 | 0.415 | 0.440 | 0.924 | 0.596 | 0 |
| Noisy Sparse (High Noise) | 0.388 | 1.000 | 0.559 | 0.182 | 0.983 | 0.307 | 0.319 | 0.891 | 0.470 | 0.021 | 0.028 | 0.024 | 0 |
| Galactic Plane (Crowded) | 0.168 | 1.000 | 0.288 | 0.875 | 0.961 | 0.916 | 0.271 | 0.989 | 0.426 | 0.367 | 0.861 | 0.514 | 0 |
| Ultra-Faint Artifacts | 0.217 | 1.000 | 0.357 | 0.268 | 0.941 | 0.417 | 0.254 | 0.944 | 0.400 | 0.120 | 0.791 | 0.208 | 0 |
| Complex Ideal (Var Bkg/PSF) | 0.169 | 1.000 | 0.289 | 1.000 | 0.979 | 0.989 | 0.239 | 0.904 | 0.378 | 0.027 | 0.049 | 0.035 | 0 |
| Complex Noisy | 0.391 | 1.000 | 0.563 | 0.203 | 0.925 | 0.333 | 0.178 | 0.890 | 0.296 | 0.872 | 0.852 | 0.862 | 0 |
| Complex Crowded | 0.197 | 1.000 | 0.329 | 0.933 | 0.824 | 0.875 | 0.241 | 0.992 | 0.387 | 0.201 | 0.760 | 0.318 | 0 |
| Extreme Gradient Bkg | 0.169 | 1.000 | 0.289 | 1.000 | 0.889 | 0.941 | 0.215 | 0.954 | 0.351 | 0.218 | 0.596 | 0.319 | 0 |
| Elliptical PSF/Tracking err | 0.181 | 1.000 | 0.307 | 0.943 | 0.926 | 0.935 | 0.139 | 0.942 | 0.242 | 0.256 | 0.922 | 0.401 | 0 |
| Thick Satellite Streak | 0.180 | 1.000 | 0.305 | 0.927 | 0.962 | 0.944 | 0.194 | 0.955 | 0.323 | 0.503 | 0.558 | 0.529 | 0 |
| Extreme Poisson Crowded | 0.369 | 1.000 | 0.539 | 0.153 | 0.953 | 0.264 | 0.277 | 0.992 | 0.433 | 0.024 | 0.024 | 0.024 | 0 |

## Benchmark Gate Failures

- Average streak F1 0.348 < 0.650
- Complex Ideal (Var Bkg/PSF) streak F1 0.035 < 0.450
- Extreme Poisson Crowded streak F1 0.024 < 0.250
- Average object F1 0.375 < 0.400

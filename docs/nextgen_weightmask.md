# Next-Generation WeightMask

**Status:** research and implementation plan, July 2026  
**Scope:** astronomical imaging, multi-extension FITS, calibration frames, and 2-D spectra  
**Primary principle:** preserve the measured pixels and attach a calibrated defect/noise model; do not silently replace scientific data with a neural reconstruction.

## Executive summary

WeightMask already provides a useful classical pipeline for astronomical FITS images: background estimation, bad-pixel and saturation handling, cosmic-ray detection, satellite-trail detection, object masks, inverse-variance maps, confidence maps, bitmasks, MEF processing, and synthetic/real-data benchmarks. The next step should not be “a larger denoising network.” It should be a **calibration-conditioned, probabilistic detector-quality and empirical-likelihood system** that can operate when the upstream detrending pipeline is unavailable or opaque.

The project should estimate the statistics of the *delivered image product* rather than pretending that the raw detector variance has been propagated exactly. It should combine:

- classical and learned defect detectors;
- dark, bias, flat, and other calibration-image sequences;
- self-supervised and weakly supervised learning;
- physically motivated synthetic defect injection;
- open-set anomaly detection;
- an empirical signal-dependent noise model;
- a compact representation of correlated noise;
- explicit out-of-distribution and abstention signals;
- validation through PSF photometry, profile fitting, coaddition, spectral extraction, and shape measurement.

The central output is not a single binary mask and not a single denoised image. It is a collection of products describing which pixels are unusable, which are merely suspicious, what additive contamination may be present, and what likelihood should be used when fitting the original pixels.

A concise project definition is:

> **Next-generation WeightMask estimates calibrated defect probabilities, additive-contamination terms, effective post-detrending variance, and compact correlated-noise models for astronomical images and 2-D spectra, using science and calibration data with minimal human labelling.**

---

## 1. Astronomical context

### 1.1 Why masks and weights matter

Astronomical measurements are rarely made from isolated high-S/N pixels. Fluxes, centroids, sizes, ellipticities, line strengths, and transient detections are inferred by fitting a model across many pixels. A small number of unrecognized cosmic rays or trail pixels can dominate a fit, while an overly aggressive mask discards useful information and reduces depth.

The usual data model contains some combination of:

- a science image;
- a data-quality or integer mask plane;
- a variance, RMS, or inverse-variance plane;
- optional calibration and provenance metadata.

This is a sound interface, but in practice several complications occur:

1. the science product may already have passed through an inaccessible detrending pipeline;
2. the supplied variance may omit calibration uncertainty or post-processing correlations;
3. bad pixels may have been interpolated without preserving provenance;
4. weak artifacts may not justify a hard mask but still require a non-Gaussian likelihood;
5. the same detector signature can vary with time, temperature, amplifier, exposure history, filter, or readout mode;
6. raw, detrended, resampled, and coadded images require different statistical interpretations.

The practical target is therefore an **effective likelihood in the coordinate system and units of the delivered image**.

### 1.2 Data regimes to support

The design should accommodate the following without requiring separate projects:

| Regime | Typical available information | Main statistical difficulty |
|---|---|---|
| Raw CCD/CMOS image | overscan, bias, dark, gain, amplifier geometry | full detector model is possible but instrument metadata vary |
| Detrended single exposure | science image, partial header, sometimes weight/DQ | upstream transforms and uncertainty propagation may be unknown |
| MEF mosaic | multiple detectors/amplifiers, extension-specific headers | heterogeneous detector states and scales |
| Resampled image | WCS-aligned image and perhaps weight map | spatial covariance from interpolation |
| Coadd | combined science and weight products | mixture of PSFs, masks, resampling, rejection, and exposure counts |
| 2-D spectrum | wavelength/spatial axes, slit/order geometry, sky residuals | anisotropic structure; emission lines must not be confused with defects |
| Calibration sequence | biases, darks, flats, persistence or lamp frames | repeated detector-only information but no astronomical scene |

A universal model should mean **safe adaptation across these regimes**, not a claim that one fixed network is always correct.

---

## 2. Defect and noise taxonomy

The outputs should distinguish physical classes because downstream treatment differs.

### 2.1 Hard-invalid pixels

These generally have no defensible science likelihood and should receive zero statistical weight:

- missing data or NaNs;
- saturated pixels when the response is clipped or nonlinear;
- severe bleed regions with destroyed charge information;
- dead pixels;
- pixels replaced by an unknown interpolation procedure;
- detector regions outside the valid imaging area;
- unrecoverable telemetry/readout failures.

### 2.2 Sparse transient contamination

These are often well described by a probability of contamination and a broad outlier likelihood:

- cosmic rays;
- satellite, aircraft, meteor, or moving-object trails;
- transient electronic spikes;
- ramp jumps in infrared detectors;
- isolated hot-pixel excursions;
- short-lived persistence events.

### 2.3 Persistent or slowly varying detector defects

Calibration sequences are especially informative for:

- stable and unstable hot pixels;
- dead or low-response pixels;
- bad columns and rows;
- random-telegraph pixels;
- amplifier boundaries and bias steps;
- dark-current structure;
- evolving radiation damage;
- persistence histories.

### 2.4 Structured additive signatures

These may be estimated and subtracted only when their uncertainty is also estimated:

- crosstalk ghosts;
- amplifier glow;
- bias striping and herringbone patterns;
- 1/f noise;
- fringe residuals;
- scattered light;
- pupil ghosts and internal reflections;
- CTI trails;
- sky-subtraction residuals in 2-D spectra.

### 2.5 Correlated stochastic noise

A diagonal inverse-variance map is insufficient after:

- interpolation or resampling;
- PSF matching;
- inter-pixel capacitance;
- row/column common-mode subtraction;
- destriping;
- convolution or deconvolution;
- image combination.

The practical representation should be a diagonal variance plane plus a local covariance kernel, power spectrum, whitening filter, or low-rank covariance component.

---

## 3. Current WeightMask baseline

The existing code is a good classical foundation and should remain usable without Torch. It already exposes:

- FITS and MEF processing through the CLI;
- a combined bitmask;
- inverse-variance, weight, confidence, sky, and component-mask outputs;
- bad-pixel and flat handling;
- saturation and bleed masking;
- L.A.Cosmic-style cosmic-ray detection;
- multi-stage satellite-trail detection;
- SEP-based background and source segmentation;
- synthetic and real-data benchmark harnesses.

The current `WeightMapGenerator.process` interface returns `weight_map`, `flag_map`, `inv_variance_map`, `confidence_map`, `sky_map`, and individual masks. This API should be extended rather than replaced.

### 3.1 Main limitations to address

1. **A hard mask is currently the dominant representation.** Suspicious pixels are generally either kept or zero-weighted.
2. **The inverse variance is mainly diagonal.** It cannot fully describe post-resampling or amplifier-correlated noise.
3. **The variance model assumes more knowledge of the detector pipeline than is usually available.** In many archives, the delivered image is detrended but the exact sequence and uncertainty propagation are inaccessible.
4. **Calibration sequences are not yet first-class inputs.** A single flat can be supplied, but time-series dark/bias/flat information is not modelled.
5. **There is no open-set anomaly channel.** Known classes are handled, but unfamiliar additive signatures do not produce a calibrated abstention score.
6. **The benchmark is mostly pixel-centric.** It should add forced photometry, PSF fitting, spectral extraction, and shape/profile bias tests.

---

## 4. Relationship to AstroSURE

The AstroSURE draft should remain a distinct, narrower contribution. It evaluates target-free denoising methods for *source detection*, comparing Noise2Noise, SURE, and blind-spot approaches on simulations and HST/CFHT data. Its current framing correctly limits claims about photometry, morphology, and PSF-sensitive measurements.

The main lessons for WeightMask are:

- self-supervision can improve detection in a domain-consistent setting;
- transfer from space-like simulations to seeing-limited CFHT images is weak;
- assumed Poisson-Gaussian likelihoods are fragile when applied to opaque processed products;
- blind-spot methods can suppress compact astronomical signal or introduce structured artifacts;
- a denoised image can be useful for detection but should not automatically replace the original image in precision measurements.

The two projects should interact as follows:

- AstroSURE provides denoising and detection-oriented auxiliary products;
- next-generation WeightMask provides defect probabilities, empirical likelihoods, and masks for quantitative inference;
- both share simulation, patch extraction, domain-adaptation, and benchmark infrastructure;
- photometry and shape fitting always operate on the original delivered pixels unless a specific restoration method has passed dedicated bias and coverage validation.

---

## 5. Literature and community review

This review is intentionally broader than L.A.Cosmic and MaxiMask and includes astronomy, inverse problems, self-supervised denoising, anomaly detection, robust statistics, and remote-sensing ideas.

### 5.1 Astronomy-specific defect detection

#### L.A.Cosmic and Astro-SCRAPPY

L.A.Cosmic remains the standard classical reference for single-exposure cosmic-ray detection. It uses Laplacian edge information to separate sharp non-PSF events from astronomical sources. Astro-SCRAPPY provides a fast maintained implementation. It is indispensable as a baseline, but it requires parameter choices and does not provide a general post-detrending likelihood or open-set detector.

- van Dokkum, 2001, *Cosmic-Ray Rejection by Laplacian Edge Detection*, PASP, 113, 1420.
- Astro-SCRAPPY: <https://github.com/astropy/astroscrappy>

#### deepCR

`deepCR` demonstrated that learned CR masks and inpainting can outperform classical methods on HST detectors. It is important evidence that dense learned segmentation is useful, but pretrained models are instrument-specific and the cleaned image is not a substitute for a calibrated measurement likelihood.

- Zhang & Bloom, 2020, *deepCR: Cosmic Ray Rejection with Deep Learning*, ApJ, 889, 24.
- <https://github.com/profjsb/deepCR>

#### Cosmic-CoNN

Cosmic-CoNN is the strongest direct precedent for cross-instrument CR generalization. It used a large, diverse ground-based data set and reported high precision at fixed recall on unseen Gemini configurations. The relevant design lesson is that broad instrument diversity and probability outputs can generalize better than per-instrument threshold tuning.

- Xu et al., 2021, *Cosmic-CoNN: A Cosmic Ray Detection Deep-Learning Framework, Dataset, and Toolkit*, <https://arxiv.org/abs/2106.14922>
- <https://github.com/cy-xu/cosmic-conn>

#### MaxiMask and MaxiTrack

MaxiMask/MaxiTrack are the closest prior work in breadth of contaminant classes. They address cosmic rays, hot/bad pixels, persistence, satellite trails, fringe residuals, saturation, diffraction spikes, and tracking errors. Their limitation relative to this project is that they primarily produce semantic masks rather than a calibration-conditioned likelihood and covariance model.

- Paillassa et al., 2020, *MaxiMask and MaxiTrack: two new tools for identifying contaminants in astronomical images using convolutional neural networks*, A&A, 637, A13.
- <https://arxiv.org/abs/1907.08298>

#### Satellite-trail detection

Classical Hough and Radon methods remain highly competitive because trails are long, anisotropic, and may have mean surface brightness below the per-pixel noise. The modified Radon-transform work for ACS/WFC is especially relevant: it reports sensitivity to faint trails, identifies unreliable regions of parameter space, and provides an operational implementation in `acstools`. Recent U-Net variants continue to improve instrument-specific segmentation, but they do not remove the need for geometric methods.

- Stark et al., 2026, *Improved Identification of Satellite Trails in ACS/WFC Imaging Using a Modified Radon Transform*, <https://arxiv.org/abs/2602.17816>
- Yu, Zheng & Fang, 2025, *Using Deep Learning to Identify Artificial Satellite Trails in Multi-band Photometric Astronomical Images*, <https://arxiv.org/abs/2509.04081>
- `acstools.satdet`: <https://acstools.readthedocs.io/en/latest/satdet.html>

A hybrid system is preferable: learned local evidence plus Radon/Hough/global-line consistency.

### 5.2 Production astronomy data models

Mission and survey pipelines provide a useful output contract even when their internals are instrument-specific.

- JWST products separate science, DQ, error, and multiple variance components such as Poisson, read-noise, and flat-field contributions.
- PypeIt stores processed images, inverse variance, bad-pixel masks, sky/object models, slit geometry, and detector metadata for 2-D spectra.
- Astropy `CCDData` and `NDData` distinguish data, masks, flags, and uncertainty objects.

The next-generation WeightMask contract should be interoperable with these conventions while adding probability and covariance products.

### 5.3 Proper coaddition and sufficient statistics

Proper image coaddition and subtraction work emphasizes that statistically valid products should retain the information needed for downstream inference and should control noise correlations rather than merely creating visually attractive images.

- Zackay & Ofek, 2017, *How to Coadd Images? I. Optimal Source Detection and Photometry Using Ensembles of Images*, ApJ, 836, 187.
- Zackay, Ofek & Gal-Yam, 2016, *Proper Image Subtraction—Optimal Transient Detection, Photometry, and Hypothesis Testing*, ApJ, 830, 27.

These papers support the core project principle: output a likelihood-compatible representation, not only a cleaned image.

### 5.4 Self-supervised denoising under realistic noise

The foundational Noise2Noise, Noise2Void, Noise2Self, and SURE methods remain useful references, but they should no longer define the research frontier.

#### Correlated-noise blind spots

MASH explicitly studies self-supervised denoising under unknown correlated noise, adjusting the blind region and shuffling local pixels to weaken correlations. TBSN shows that attention and multiscale channel operations can leak target-pixel information unless the architecture is carefully constrained.

- Chihaoui & Favaro, 2024, *Masked and Shuffled Blind Spot Denoising for Real-World Images*, <https://arxiv.org/abs/2404.09389>
- Li, Zhang & Zuo, 2024, *Rethinking Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising*, <https://arxiv.org/abs/2404.07846>

The astronomical implication is direct: a nominal blind spot is not valid when detector noise is correlated or when resampling has mixed neighboring pixels.

#### Low-assumption and cross-scale self-supervision

LoTA-N2N introduces a trace-constrained zero-shot adaptation objective intended to reduce dependence on explicit noise assumptions. Next-Scale Prediction separates noise decorrelation from detail preservation through cross-scale targets. These are more relevant baselines than simply replacing a U-Net with a larger transformer.

- Hu et al., 2024, *Low-Trace Adaptation of Zero-shot Self-supervised Blind Image Denoising*, <https://arxiv.org/abs/2403.12382>
- Shan et al., 2025, *Next-Scale Prediction: A Self-Supervised Approach for Real-World Image Denoising*, <https://arxiv.org/abs/2512.21038>

#### Joint signal and noise inference

Gibbs Diffusion treats unknown colored-noise parameters as latent variables and alternates signal and noise inference. This is conceptually closer to the WeightMask problem than fixed-noise SURE because the effective covariance of a detrended image may be unknown.

- Heurtel-Depeiges et al., 2024, *Listening to the Noise: Blind Denoising with Gibbs Diffusion*, <https://arxiv.org/abs/2402.19455>

Blind-Spot Guided Diffusion combines a blind-spot branch with a diffusion model for the real noise distribution. It is relevant as an experimental restoration baseline, though its computational cost and natural-image priors may be unsuitable for production astronomy.

- Cheng et al., 2025, *Blind-Spot Guided Diffusion for Self-supervised Real-World Denoising*, <https://arxiv.org/abs/2509.16091>

YeTI learns signal-dependent processed-image noise from only two noisy observations and without camera metadata. Its sRGB camera setting is not astronomical, but its black-box processing-domain formulation is highly relevant to archives where the detrending code is unavailable.

- Ko et al., 2026, *YeTI: You Only Need Two Noisy Images for Real-World sRGB Noise Generation*, <https://arxiv.org/abs/2607.09193>

### 5.5 Open-set and few-shot anomaly detection

AnomalyDINO shows that frozen DINOv2 patch features and a nominal-feature memory can provide strong one/few-shot anomaly localization without training a task-specific segmentation model.

- Damm et al., 2024, *AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2*, <https://arxiv.org/abs/2405.14529>

For astronomy it should be treated as a baseline, not as a default solution. Natural-image features may ignore low-level detector statistics and may misclassify compact sources. More promising variants would learn nominal detector features from calibration images or use a small astronomy-specific self-supervised encoder.

### 5.6 Low-rank plus sparse decomposition

Repeated biases, darks, flats, and science exposures naturally support a low-rank-plus-sparse model:

\[
D = L + S + E,
\]

where `L` contains stable or slowly varying detector modes, `S` contains sparse cosmic rays and transient defects, and `E` contains stochastic noise.

Robust principal component pursuit provides the classical statistical foundation, while learned unfolding can make the decomposition scalable and instrument-adaptive.

- Candès et al., 2011, *Robust Principal Component Analysis?*, JACM, 58, 11; <https://arxiv.org/abs/0912.3599>
- Zhou et al., 2010, *Stable Principal Component Pursuit*, <https://arxiv.org/abs/1001.2363>

Remote-sensing and hyperspectral denoising are relevant because those fields routinely separate low-rank scene structure, stripes, impulse defects, dead pixels, and spectrally/spatially correlated noise. Their exact scene priors differ from astronomy, but their structured-noise decompositions are directly useful for amplifier banding and 2-D spectra.

### 5.7 Inverse-problem libraries

DeepInverse is an actively developed PyTorch library covering differentiable physics operators, noise models, variational solvers, plug-and-play methods, unfolded optimization, self-supervised losses, and posterior sampling.

- Tachella et al., 2025, *DeepInverse: A Python package for solving imaging inverse problems with deep learning*, <https://arxiv.org/abs/2505.20160>
- <https://deepinv.github.io/deepinv/>
- <https://github.com/deepinv/deepinv>

It is a strong dependency for research models and differentiable simulation, but astronomical FITS I/O, detector geometry, calibration semantics, DQ flags, and science validation should remain in WeightMask.

### 5.8 Literature gap

No reviewed system combines all of the following:

- known-class and open-set artifact detection;
- calibration-sequence conditioning;
- support for opaque detrended products;
- effective signal-dependent variance estimation;
- compact correlated-noise output;
- raw, MEF, imaging, and 2-D spectroscopy support;
- minimal manual labels;
- science-task validation through bias and interval coverage.

That is the defensible research space for next-generation WeightMask.

---

## 6. Statistical target for opaque detrending pipelines

Let `y` be the delivered image. The unknown upstream pipeline may include bias subtraction, dark correction, flat division, nonlinearity correction, CTI correction, interpolation, background operations, and resampling. Rather than claim an exact raw-level forward model, estimate

\[
p(y \mid m, \mathcal{C}, \mathcal{M}),
\]

where:

- `m` is the astronomical model evaluated in delivered-image coordinates;
- `\mathcal{C}` is the available calibration collection;
- `\mathcal{M}` contains surviving metadata and processing-state indicators.

A practical nominal model is

\[
y = m + a + \epsilon,
\]

with

\[
\operatorname{Var}(\epsilon_i \mid m_i)
= v_{\mathrm{bg},i}
+ \alpha_i \max(m_i, 0)
+ v_{\mathrm{model},i}.
\]

Here:

- `v_bg` is the effective background/read/process variance in delivered units;
- `alpha` is an empirical signal-dependent variance coefficient, not necessarily the physical raw gain;
- `v_model` accounts for uncertainty in an additive-signature estimate;
- spatial covariance is represented separately.

This model is deliberately empirical. It can be calibrated from delivered science and calibration images even when the raw pipeline is unknown.

---

## 7. Using calibration images without dense labels

### 7.1 Dark sequences

Darks are especially valuable because they contain detector behavior and cosmic rays without astronomical sources. Model a dark sequence as

\[
D_t = \mu + \sum_k c_{tk} B_k + S_t + \epsilon_t,
\]

where:

- `mu` is the persistent dark/hot-pixel pattern;
- `B_k` are low-rank temporal, amplifier, or temperature modes;
- `S_t` is sparse transient contamination;
- `epsilon_t` is stochastic noise.

From this decomposition, derive pseudo-labels for:

- stable hot pixels;
- unstable and random-telegraph pixels;
- bad rows/columns;
- cosmic-ray morphology;
- temporal detector drift;
- amplifier covariance and banding;
- exposure-time or temperature dependence.

A robust temporal median and sigma-clipped residual detector is the baseline. Robust PCA or a learned unfolded low-rank-plus-sparse solver is the research model.

### 7.2 Bias sequences

Biases constrain:

- read-noise amplitude;
- amplifier offsets;
- row/column covariance;
- low-frequency and 1/f structure;
- electronics instabilities;
- detector-state changes.

### 7.3 Flat sequences

Flats constrain:

- dead/low-response pixels;
- unstable response;
- bad columns;
- large-scale illumination modes;
- approximate gain/variance relationships;
- detector-coordinate features that should not move with the sky.

Flat structure must be separated from illumination mismatch and spectral-response effects. The model should therefore distinguish persistent pixel-scale defects from low-frequency flat modes.

### 7.4 Repeated science exposures

When available, repeated exposures provide the best constraint on the complete delivered-domain residual. After accounting for WCS, PSF, transparency, sky, and variability, inconsistent detector-coordinate features become weak labels for defects.

### 7.5 Single science images

For a single detrended image, use:

- robust local background statistics;
- anisotropic row/column residual models;
- learned nominal detector features;
- synthetic defect injection;
- teacher agreement from classical detectors;
- OOD/abstention scores.

Single-image self-supervision should be applied only after checking or estimating local noise correlation.

---

## 8. Proposed output contract

The product schema should distinguish hard validity, probabilities, additive estimates, diagonal variance, and covariance.

| Product | Meaning | Typical downstream use |
|---|---|---|
| `science` | original delivered pixels | all quantitative fitting |
| `dq_hard` | integer bits for invalid/nonlinear/missing pixels | zero weight / exclusion |
| `p_good` | prior probability of nominal behavior | mixture likelihood or soft diagnostics |
| `p_cr` | cosmic-ray probability | robust likelihood, optional mask |
| `p_streak` | trail probability | robust likelihood, trail mask |
| `p_hot` | hot/unstable pixel probability | mask or mixture component |
| `p_column` | bad-row/column probability | mask or structured component |
| `p_unknown` | unexplained anomaly probability | broad outlier likelihood / abstention |
| `additive_mean` | estimated additive contamination | optional correction before fit |
| `additive_variance` | uncertainty in additive estimate | add to effective variance |
| `variance_background` | effective background/process variance | nominal likelihood |
| `poisson_coefficient` | signal-dependent variance coefficient | model-dependent variance update |
| `inverse_variance_diag` | legacy diagonal approximation | existing software compatibility |
| `noise_kernel` or `noise_psd` | local correlation model | whitening / GLS |
| `low_rank_modes` | amplifier/common-mode covariance factors | structured GLS |
| `ood_score` | distance from calibrated/training domain | quality control and fallback |
| `denoised_detection` | optional restoration for detection only | source finding / visualization |
| `provenance` | model version, inputs, calibration IDs, settings | reproducibility |

The existing `weight_map`, `flag_map`, `inv_variance_map`, and `confidence_map` remain supported as compatibility views derived from this richer product.

---

## 9. Example: fitting a star with a scaled PSF

### 9.1 Model

For a postage stamp with original delivered pixels `y_i`, fit

\[
m_i(\theta)
= F p_i(x_0, y_0)
+ b_0 + b_x x_i + b_y y_i,
\]

where `p_i` is the normalized PSF evaluated at the centroid and `F` is the stellar flux.

The fit must use the **original image**, not `denoised_detection`.

### 9.2 Hard-mask generalized least squares

Remove pixels marked by `dq_hard`. If an additive component is estimated, define

\[
z = y - \hat a,
\]

and

\[
C_{\mathrm{eff}}(\theta)
= C_{\mathrm{bg}}
+ C_{\mathrm{source}}(\theta)
+ C_{\mathrm{additive}}.
\]

For a fixed centroid and design matrix

\[
X = [p, 1, x, y],
\]

the generalized least-squares solution is

\[
\hat\beta
= (X^T C_{\mathrm{eff}}^{-1} X)^{-1}
X^T C_{\mathrm{eff}}^{-1} z.
\]

The source-dependent term depends on flux, so iterate:

1. fit with background covariance;
2. update the signal-dependent variance from `F p_i`;
3. refit until stable.

### 9.3 Compatibility soft weights

For software accepting only scalar weights, use the approximation

\[
w_i
= \frac{p_{\mathrm{good},i}}
{v_{\mathrm{bg},i}
+ \alpha_i Fp_i
+ v_{\mathrm{additive},i}}.
\]

This is operationally useful but not fully probabilistic: contamination probability is not mathematically equivalent to reduced exposure time.

### 9.4 Preferred mixture likelihood

Treat class probabilities as priors in a mixture model:

\[
\begin{aligned}
p(y_i \mid \theta)
={}& \pi_{i,\mathrm{good}}
\,\mathcal{N}(y_i \mid m_i(\theta)+a_{i,\mathrm{good}}, v_{i,\mathrm{good}}) \\
&+ \sum_c \pi_{ic}
\,\mathcal{N}(y_i \mid m_i(\theta)+a_{ic}, v_i+s_{ic}^2) \\
&+ \pi_{i,\mathrm{unknown}}\,g_i(y_i),
\end{aligned}
\]

where `g_i` is a broad Student-t or learned outlier distribution.

The posterior nominal-pixel probability becomes

\[
r_i(\theta) = P(\mathrm{good} \mid y_i, \theta).
\]

A diagonal iteratively reweighted approximation uses

\[
w_i^{\mathrm{IRLS}} = \frac{r_i(\theta)}{v_i(\theta)}.
\]

This permits suspicious pixels to be rescued when they agree with the PSF and nominal pixels to be downweighted when they strongly disagree.

### 9.5 Correlated noise

When a local covariance model is available, whiten the data and model:

\[
\tilde y = L(y-\hat a),\qquad
\tilde m = Lm,\qquad
LCL^T \approx I.
\]

The first production implementation should combine:

- hard masks for severe defects;
- local whitening for retained pixels;
- pixelwise mixture probabilities for sparse questionable pixels.

A fully multivariate mixture model can follow later.

### 9.6 Validation outputs from the fitter

The benchmark fitter should return:

- flux, centroid, and background parameters;
- covariance or posterior intervals;
- posterior good-pixel probabilities;
- effective number of contributing pixels;
- whitened residual diagnostics;
- OOD score;
- comparison with a conservative hard-mask fit;
- sensitivity to additive-contamination correction.

This creates a direct science-facing acceptance test for every WeightMask model.

---

## 10. Proposed architecture

### 10.1 Keep a strong classical core

The existing algorithms remain the default and provide:

- deterministic fallbacks;
- pseudo-label teachers;
- physically interpretable features;
- CPU-only operation;
- regression tests.

### 10.2 Calibration encoder

A calibration module should summarize a variable-length set of biases, darks, and flats into:

- detector-coordinate defect priors;
- persistent and transient components;
- low-rank modes;
- local covariance or PSD parameters;
- calibration quality and age indicators.

Initial implementation: robust NumPy/SciPy decomposition.  
Research implementation: Torch set/sequence encoder or unfolded low-rank-plus-sparse model.

### 10.3 Science-image encoder

Use a multiscale encoder with anisotropic context. Important features include:

- local PSF-scale morphology;
- long-range line evidence;
- row/column structure;
- detector-coordinate embeddings;
- optional wavelength/spatial-axis semantics for spectra;
- optional metadata embeddings.

A U-Net-like backbone remains a valid starting point. Radon/Hough evidence and row/column projections should be explicit auxiliary channels rather than expecting a generic transformer to discover every geometry.

### 10.4 Prediction heads

Use separate heads for:

1. known-class probabilities;
2. open-set anomaly probability;
3. additive-contamination mean;
4. additive-contamination variance;
5. background/effective variance;
6. signal-dependent variance coefficient;
7. local covariance/PSD parameters;
8. OOD/domain score.

Multi-task sharing is useful, but these outputs should remain semantically distinct.

### 10.5 DeepInverse integration

DeepInverse is most useful for:

- differentiable detector and artifact injection;
- Poisson-Gaussian and correlated-noise experiments;
- masked/incomplete-data operators;
- IPC, convolution, and resampling operators;
- plug-and-play or unfolded low-rank/sparse solvers;
- self-supervised loss implementations;
- posterior-sampling prototypes.

WeightMask should own:

- FITS/MEF and spectrum I/O;
- detector/amplifier geometry;
- calibration-set discovery;
- DQ-bit semantics;
- product serialization;
- model registry and provenance;
- astronomy-specific validation.

`deepinv` should be an optional research dependency, not required by the classical package.

---

## 11. Training with little human labelling

Use several supervision channels simultaneously.

### 11.1 Physics-based injections

Inject defects into real apparently clean images and simulations:

- CR charge-deposition morphologies;
- trails with varying width, PSF, brightness, fragmentation, and acceleration;
- hot/dead pixels and columns;
- saturation and bleed;
- crosstalk ghosts;
- persistence with temporal decay;
- bias steps, banding, and amplifier glow;
- fringe modes;
- CTI-like trails;
- spectral sky residuals.

The injection process provides exact class masks and additive images.

### 11.2 Calibration-derived pseudo-labels

Use temporal medians, robust residuals, low-rank-plus-sparse decompositions, and change-point statistics on calibration stacks. These labels are imperfect but abundant.

### 11.3 Teacher ensemble

Combine:

- current WeightMask outputs;
- Astro-SCRAPPY;
- Cosmic-CoNN;
- Radon/Hough/MRT detections;
- instrument DQ planes;
- temporal outlier rejection;
- calibration-derived maps.

Use high-agreement pixels as high-precision labels and retain disagreement as an uncertainty target.

### 11.4 Cross-exposure consistency

When repeated observations exist, compare in sky and detector coordinates. Astrophysical signal follows the WCS; detector artifacts follow detector coordinates; moving objects and variable sources require explicit exceptions.

### 11.5 Small adjudicated benchmark

Human labelling remains necessary for evaluation and probability calibration. The target should be a compact, difficult, instrument-diverse benchmark rather than millions of routine labels.

---

## 12. Evaluation

### 12.1 Pixel and object metrics

- precision/recall and PR curves per defect class;
- intersection over union for extended masks;
- trail centerline and width errors;
- connected-component recovery;
- false positives near stars, galaxy cores, emission lines, and diffraction spikes;
- runtime and memory.

### 12.2 Probability calibration

- reliability diagrams;
- Brier score;
- negative log likelihood;
- expected calibration error;
- calibration versus S/N, detector region, and source crowding;
- leave-one-instrument-out performance;
- OOD score versus actual failure severity.

### 12.3 Noise-model diagnostics

For nominal background pixels:

- standardized-residual mean and variance;
- tail probabilities;
- spatial autocorrelation;
- row/column power;
- whitened power spectrum;
- stability across amplifier, detector, exposure time, and observing date.

### 12.4 Science-facing metrics

#### PSF photometry

- flux and centroid bias;
- reported-uncertainty coverage;
- catastrophic-fit rate;
- sensitivity to CR/star overlap;
- behavior under faint trails and bad columns.

#### Galaxy/profile fitting

- total flux, radius, Sérsic index, axis ratio, and position-angle bias;
- residual structure around compact nuclei;
- covariance coverage.

#### Weak-lensing shapes

- multiplicative and additive shear bias;
- selection bias induced by mask probabilities;
- sensitivity to correlated noise.

#### Coaddition and subtraction

- depth and point-source S/N;
- false detections;
- photometric bias around masked regions;
- residual trail/cosmic artifacts;
- noise whiteness.

#### 2-D spectroscopy

- line flux, centroid, and width bias;
- extracted-spectrum covariance;
- false CR classification on narrow emission lines;
- sky-residual handling.

The primary model-selection criterion should be downstream bias and uncertainty coverage, not only segmentation F1.

---

## 13. Code integration plan

### 13.1 Package structure

Proposed additions:

```text
weightmask/
    products.py             # dataclasses and product contract
    calibration.py          # calibration-set statistics and robust decomposition
    covariance.py           # kernels, PSDs, whitening, low-rank covariance
    likelihood.py           # diagonal and mixture likelihood helpers
    adapters/
        deepinv.py           # optional differentiable physics integration
    ml/
        models.py            # Torch architectures and heads
        inference.py         # tiled inference and model registry
        losses.py            # supervised, weak, self-supervised objectives
        features.py          # Radon/row-column/calibration features
    fitting/
        psf.py               # reference robust PSF fitter for validation
    benchmarks/
        photometry.py
        profiles.py
        spectra.py
        calibration.py
```

Torch and DeepInverse should be optional extras:

```toml
[project.optional-dependencies]
ml = ["torch", "torchvision"]
deepinv = ["deepinv"]
```

Do not force ML dependencies on users who only need the classical CLI.

### 13.2 Product dataclass

Introduce a stable container while retaining dictionary compatibility:

```python
@dataclass
class WeightMaskProducts:
    science: np.ndarray
    dq_hard: np.ndarray
    inverse_variance_diag: np.ndarray | None
    sky: np.ndarray | None
    probabilities: dict[str, np.ndarray]
    additive_mean: np.ndarray | None = None
    additive_variance: np.ndarray | None = None
    variance_background: np.ndarray | None = None
    poisson_coefficient: np.ndarray | None = None
    noise_kernel: np.ndarray | None = None
    noise_psd: np.ndarray | None = None
    low_rank_modes: np.ndarray | None = None
    ood_score: np.ndarray | float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
```

Legacy maps are generated from this object by policy functions.

### 13.3 Configuration sketch

```yaml
nextgen:
  enabled: true

  calibration:
    darks: null
    biases: null
    flats: null
    robust_low_rank:
      enabled: true
      rank: auto
      sparse_threshold: auto

  probabilities:
    classes:
      - cr
      - streak
      - hot
      - bad_column
      - saturation
      - persistence
      - unknown
    calibration: temperature_scaling

  likelihood:
    mode: empirical_post_detrending
    background_variance: local_robust
    signal_dependent_term: fit
    outlier_distribution: student_t

  covariance:
    representation: local_kernel
    kernel_size:  nine_by_nine
    estimate_from:
      - biases
      - darks
      - science_background

  ml:
    enabled: false
    model: null
    device: auto
    use_deepinv: false

  output:
    write_probability_planes: true
    write_additive_planes: true
    write_noise_kernel: true
    write_legacy_weight: true
```

### 13.4 Python API sketch

```python
from weightmask import WeightMapGenerator
from weightmask.calibration import CalibrationSet

cal = CalibrationSet.from_files(
    darks=dark_paths,
    biases=bias_paths,
    flats=flat_paths,
)

generator = WeightMapGenerator(config)
products = generator.process(
    data=science,
    header=header,
    calibration=cal,
)

flux_result = products.fit_psf(
    psf=psf_model,
    x=source_x,
    y=source_y,
    likelihood="mixture",
)
```

The first implementation need not expose `fit_psf` as a production API; it can live in the benchmark module until validated.

---

## 14. Phased roadmap

### Phase 0 — Freeze the benchmark and product schema

- add this design document;
- define DQ-bit and probability-plane names;
- create a versioned `WeightMaskProducts` schema;
- add real-data manifests for multiple instruments and 2-D spectra;
- integrate external baselines where licenses permit;
- add PSF-photometry injection tests.

### Phase 1 — Empirical post-detrending variance

- robust local background variance;
- empirical signal-dependent variance coefficient;
- standardized-residual calibration;
- local covariance kernel/PSD estimation;
- whitening utilities;
- validation with blank sky and injected stars.

This phase is valuable even without ML.

### Phase 2 — Calibration-sequence engine

- dark/bias/flat collection API;
- temporal robust statistics;
- persistent/unstable-pixel maps;
- low-rank-plus-sparse decomposition;
- calibration-derived cosmic-ray pseudo-labels;
- detector-state and OOD summaries.

### Phase 3 — Torch probability model

- optional Torch dependency;
- CR/streak/hot/column heads;
- explicit Radon and row/column features;
- synthetic injection pipeline;
- probability calibration;
- leave-one-instrument-out evaluation.

The first learned targets should be the current benchmark weaknesses: low-surface-brightness trails and cosmic-ray false positives in high-noise/Poisson-dominated regions.

### Phase 4 — Open-set additive signatures

- nominal calibration/science feature model;
- unknown anomaly head;
- additive mean/variance heads;
- crosstalk, persistence, fringe, banding, and amplifier-glow prototypes;
- abstention/fallback behavior.

### Phase 5 — Spectroscopy and raw ramps

- anisotropic 2-D spectral model;
- slit/order geometry and emission-line protection;
- infrared ramp-jump support;
- mode-specific adapters while preserving the common product schema.

### Phase 6 — Science validation and release

- PSF and galaxy-profile benchmarks;
- spectral extraction tests;
- coadd/difference-imaging tests;
- shape-measurement challenge;
- model cards and failure-domain documentation;
- stable FITS serialization.

---

## 15. Research and publication strategy

### AstroSURE paper

Keep the current paper focused on detection-oriented target-free denoising. Update its literature review with modern correlated-noise and low-assumption self-supervision, state the likelihood limitations more explicitly, and avoid implying that larger architectures alone solve domain transfer.

### WeightMask methods paper

A first WeightMask paper can be predominantly statistical:

> **Calibration-conditioned empirical likelihoods and probabilistic defect masks for post-detrending astronomical images.**

It should demonstrate:

- calibration-derived defect maps with little manual labelling;
- effective variance and covariance estimation in delivered-image units;
- robust PSF photometry under cosmic rays, trails, hot pixels, and bad columns;
- leave-one-instrument-out tests;
- comparisons against classical and learned baselines.

### Later ML paper

A second paper can focus on open-set and multi-modal learning:

> **Open-set detector anomaly segmentation for imaging and 2-D spectroscopy using calibration-conditioned weak supervision.**

---

## 16. Risks and safeguards

| Risk | Safeguard |
|---|---|
| network removes real compact sources | fit original pixels; use restoration only for detection; inject PSF sources in validation |
| opaque pipeline invalidates Poisson-Gaussian assumptions | learn effective delivered-domain likelihood and covariance |
| calibration frames differ from science processing | record processing state; use detector priors but recalibrate output-domain noise on science background |
| synthetic defects do not match reality | mix physics injections, teacher labels, temporal consistency, and a small human benchmark |
| natural-image foundation model fails on detector statistics | retain it only as a baseline; train astronomy/calibration-specific features |
| one model fails on an unseen instrument | OOD score, test-time calibration, classical fallback, documented abstention |
| probability maps are misused as inverse variance | keep probability and variance planes separate; provide explicit conversion policies |
| additive subtraction biases photometry | subtract only with calibrated additive variance; otherwise use an outlier likelihood |
| covariance products are too large | use local kernels, PSDs, or low-rank factors rather than dense matrices |

---

## 17. Immediate next actions

1. Implement `WeightMaskProducts` without changing existing CLI behavior.
2. Add a PSF-fitting benchmark that uses original pixels, hard masks, soft weights, and a mixture likelihood.
3. Add a calibration-set loader for dark, bias, and flat sequences.
4. Build a robust temporal dark decomposition and compare its CR/hot-pixel maps with Astro-SCRAPPY and Cosmic-CoNN.
5. Add local covariance-kernel estimation from bias/dark/science-background regions.
6. Expand the real-data manifest to at least one ground imager, one space imager, one IR detector, and one 2-D spectrograph.
7. Define acceptance thresholds in terms of flux bias, interval coverage, trail recovery, and OOD behavior.
8. Only then select the first Torch architecture and DeepInverse components.

The strongest near-term deliverable is therefore not a general denoised image. It is a **calibration-aware product schema, empirical delivered-domain likelihood, and robust PSF-photometry validation loop** built on the existing WeightMask pipeline.

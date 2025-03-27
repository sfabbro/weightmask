# Astronomical Image Effects, Noise Sources, and Processing Systematics

This document describes various effects, contaminants, and noise sources encountered in astronomical CCD imaging, including those introduced or propagated through typical detrending pipelines. Understanding these is crucial for accurate data analysis, masking, variance estimation, and image stacking.

## 1. Real Instrumental / Observational Effects / Residual after Basic Detrending

These are typically physical artifacts or characteristics originating from the instrument, optics, detector, or observing conditions that might persist even after standard detrending (bias/dark subtraction, flat-fielding).

* **Cosmic Rays**
    high-energy particles that strike the detector and deposit charge in a compact, often irregular pattern, typically affecting 1-3 adjacent pixels. these appear as sharp, isolated features with high signal-to-noise ratio and no PSF-like profile. cosmic rays must be identified and masked to prevent false source detections and corrupted photometry.

* **Blooming / Charge Bleeding**
    when a pixel collects more charge than it can hold (saturates), the excess charge spills into adjacent pixels. this typically occurs along the column direction in CCDs due to the structure of the potential wells. blooming creates linear streaks that can extend hundreds of pixels from very bright sources, contaminating other objects and creating false detections.

* **Linear Streaks**
    trails across the image caused by moving objects. common sources include artificial satellites, airplanes, meteors, and sometimes fast-moving asteroids. they vary greatly in brightness, width, length, and orientation, but generally form continuous straight paths across portions of the image.

* **Persistence / Latent Images**
    a faint residual image left behind from bright illumination in a *previous* exposure. this occurs due to charge carriers becoming trapped in the detector substrate and slowly releasing over time. persistence can last for multiple subsequent exposures, gradually decaying, and is especially problematic in infrared detectors.

* **CTI Trails (Charge Transfer Inefficiency)**
    during CCD readout, charge packets are shifted pixel by pixel towards the amplifier. if this transfer is imperfect (due to traps or defects, often exacerbated by radiation damage), some charge gets left behind, creating faint trails in the readout direction. the effect increases with distance from the amplifier and particularly affects low-signal pixels following high-signal pixels.

* **IPC Effect (Inter-pixel Capacitance)**
    electrical coupling between adjacent pixels causes a small fraction of the charge collected in one pixel to induce a signal in its immediate neighbours. this effectively broadens the detector's intrinsic PSF and introduces correlations in the pixel noise. particularly significant in infrared arrays and modern thick, fully-depleted CCDs.

* **Cross-talk**
    electronic signal coupling between different amplifiers or readout channels on a multi-amplifier detector. when one amplifier reads a bright source, it can induce ghost images or signal distortions in the data being read simultaneously by other amplifiers. these ghosts appear at predictable positions relative to the original source.

* **Amplifier Glow**
    infrared and near-infrared emission from the detector readout electronics, particularly visible in the corners or edges nearest to amplifiers. this creates a characteristic pattern of elevated background that increases with exposure time. particularly problematic in infrared arrays but can also affect thinned CCDs at longer wavelengths.

* **Ghosts / Reflections**
    spurious images formed by light reflecting internally between optical elements (lenses, filters, corrector plates, detector window) or the detector surface itself. these often appear as out-of-focus "donut" shapes at positions that depend on the optical design and the location of the original bright source.

* **Scattered Light**
    light from bright sources (either within or outside the field of view) scattering off telescope structures, baffles, dust on optics, or within the atmosphere. this creates extended, low-surface-brightness features that can significantly affect background estimation and faint source detection, especially near field edges.

* **Diffraction Spikes**
    linear features extending from bright sources caused by light diffracting around support structures within the telescope, typically the vanes holding the secondary mirror. these form characteristic patterns (often X or cross shapes) and can extend hundreds of pixels from very bright stars.

* **Fringing**
    interference patterns caused by reflections between parallel surfaces within the detector, particularly in thinned back-illuminated CCDs at red wavelengths. this produces wave-like patterns across the image whose strength varies with wavelength. fringing is strongly dependent on the spectral distribution of the incoming light and thus varies between science target and calibration source.

* **Edge Effects**
    physical boundaries of the detector exhibit different behaviour from the central regions. this includes different noise characteristics, sensitivity rolloff, vignetting, and often increased defect density. these effects can extend tens of pixels inward from the physical edges.

* **Correlated Read Noise**
    in many modern detectors, particularly infrared arrays and some CCD controllers, the read noise is not purely random but shows patterns or correlations between pixels. this can manifest as horizontal stripes, herringbone patterns, or 1/f noise that impacts background estimation and faint source detection.

* **Bad Pixels/Columns**
    detector elements with anomalous response characteristics, including hot pixels (abnormally high dark current), dead pixels (no response), traps (nonlinear response), and similar defects affecting entire columns. these require specific masking and can evolve over the detector lifetime.

This table focuses on physical artifacts or characteristics that might remain after standard detrending and often require specific masking or modeling.

| effect             | algorithms / process                      | implemented algorithm  | existing implementations            |
| :----------------- | :---------------------------------------- | :--------------------------------- | :---------------------------------------------- |
| **cosmic rays** | masking via morphological detection | lacosmic algorithm | `astroscrappy`, `ccdproc.cosmicray_lacosmic`, `MaxiMask`, ` cosmic-conn`, |
| **blooming/bleeding**| masking        | masking (core only via SAT)        | `SEP` bleed trail detection, `astro-source-subtraction` |
| **linear streaks** | masking via line detection (hough, radon) | hough (skimage), dilation        | `MaxiMask`, `DOI: 10.1093/mnras/stab3563`, `acstools.satdet`  |
| **persistence** | modeling & subtraction(X), masking(X)   | -                                  | HST/WFC3 persistence pipeline, JWST pipeline, `tshirt.pipeline`  |
| **CTI trails** | correction models(X)                      | -                                  | HST/ACS CTI correction, JWST/MIRI tools, `ctisim`  |
| **IPC effect** | kernel deconvolution(X), PSF modeling(X)  | -                                  | JWST NIRCam IPC correction, `spaceKLIP` |
| **cross-talk** | modeling & subtraction(X), masking(X)     | -                                  | LSST/DECam cross-talk correction, `MzLS_pipeline` |
| **amplifier glow** | modeling & subtraction(X), masking(X)   | assumed pre-processed      | `ccdproc`/`astropy` (overscan utilities), `WIRCam Pipeline` |
| **ghosts/reflections**| masking (geometric/thresholding)(X)     | -                                  | `photutils` (bright source masking), `acstools.ghostcorr` |
| **scattered light** | extended source masking(X), modeling(X) | background model (partial)         | `photutils` (masking), `astropy.modeling`, `jwst.straylight` |
| **diffraction spikes**| radial/geometric masking(X)             | segmentation (implicit via object) | `MaxiMask`, `diffractionspikes` package |
| **fringing** | subtraction via template/model(X)         | assumed pre-processed          | `ZZCeti_pipeline`, `decam_nightmares`, `MIRI fringe tool` |
| **edge effects** | geometric masking(X)                    | -                                  | various instrument-specific pipelines, `WebbPSF` |
| **correlated read noise** | pattern modeling(X), filtering(X) | - | `CALWF3 pipeline`, `mirisim`, `mpdaf.MUSE` |
| **bad pixels/columns** | mapping via outlier detection | masking via BPM | `ccdproc.create_deviation`, `astropy.nddata.bitmask` |

**Note 1:** While general-purpose computer vision libraries provide tools like Hough transforms, most observatories and surveys implement custom streak detection tailored to their specific data characteristics.

---

## 2. Pipeline-Propagated Noise & Systematics

These are noise components or systematic errors introduced, amplified, or modified by the calibration and processing steps applied to the raw data.

* **Shot Noise (Poisson)**
    the fundamental quantum nature of light creates statistical fluctuations that follow Poisson statistics. the variance equals the mean signal in the original photon units. this noise is inherent to the signal itself and propagates through all calibration steps. after gain correction, the variance is signal/gain for an ideal detector.

* **Read Noise**
    electronic noise introduced during the analog-to-digital conversion of the pixel values. this is additive, approximately Gaussian, and independent of the signal level. read noise is typically characterized during detector qualification and measured from bias frames. it sets the fundamental detection floor for short exposures.

* **Bias Noise Propagation**
    the master bias frame, created by combining many individual bias frames, still contains read noise (reduced by √n for n frames). when this master bias is subtracted from the science frame, its variance adds in quadrature to the science frame variance: `Var(Sci - Bias) = Var(Sci) + Var(Bias)`.

* **Bias Structure Residuals**
    if the master bias frame does not perfectly represent the actual bias structure (due to temporal variations, temperature changes, or insufficient frames), subtracting it will leave residual 2D pattern noise. these can appear as subtle striping, fixed patterns, or amplifier boundary effects.

* **Dark Noise Propagation**
    master dark frames contain both read noise and shot noise from the dark current itself. subtracting the (scaled) master dark adds its variance to the science frame variance, modified by any scaling factor applied: `Var(Sci - k*Dark) = Var(Sci) + k²*Var(Dark)` where k is the scaling factor.

* **Dark Pattern Residuals**
    if the master dark frame doesn't perfectly represent the dark current pattern during science exposure (due to temperature changes, exposure time scaling issues, or hot pixel instability), pixel-by-pixel residuals will remain. the pattern and scale depend on detector temperature stability and scaling accuracy.

* **Flat-Field Noise Propagation**
    the master flat field frame contains photon noise from its own illumination, plus read noise and dark noise. this noise propagates non-linearly during division. the variance in the flat-fielded image is approximately `Var_Corr ≈ (Var_Raw / Flat²) + (Raw² * Var_Flat / Flat⁴)`. this becomes significant for faint signals and/or low SNR flat fields.

* **Flat-Fielding Residuals**
    if the master flat field doesn't perfectly match the pixel-to-pixel sensitivity variations during the science observation, residual errors remain. causes include temporal changes in dust patterns, filter wheel positioning errors, temperature changes, or spectral mismatch between flat source and science targets.

* **Defringing Noise/Residuals**
    when a fringe pattern is removed via scaled fringe frame subtraction, noise from that frame propagates into the result. additionally, if the fringe pattern changes between calibration and science (due to wavelength shifts or instrument flexure), residual fringe patterns remain.

* **Non-linearity Residuals**
    detector response often deviates from linearity, particularly near saturation. if these non-linearities are imperfectly corrected, photometric errors remain that depend on signal level. this creates systematic flux errors that correlate with source brightness.

* **Brighter-Fatter Residuals**
    in modern thick CCDs particularly, bright pixels appear larger than faint ones due to charge repulsion effects. this creates a signal-dependent PSF and photometric nonlinearity. imperfect correction leaves systematic errors in shape measurements and aperture photometry that depend on source brightness.

* **Background Subtraction Residuals**
    imperfect background modeling introduces systematic errors in photometry and detection thresholds. these arise from inadequate model flexibility, contamination from unmasked sources, or true background structures (cirrus, galactic emission, nebulosity) that shouldn't be removed.

* **Resampling Noise Correlation**
    geometrically transforming an image (e.g., during stacking) requires interpolation, where output pixels are computed from multiple input pixels. this introduces spatial correlations in the noise, where adjacent pixels are no longer independent, complicating statistical error analysis.

* **Resampling Artifacts**
    interpolation during resampling can introduce artifacts like ringing (oscillations near sharp edges) with sinc-based kernels, or excessive smoothing with simpler algorithms. the choice of kernel represents a trade-off between resolution preservation and artifact suppression.

* **PSF Matching Noise Correlation**
    convolving an image to homogenize PSFs introduces spatial correlations in the noise. the variance map must be updated by convolving with the kernel squared, and nearby pixels become correlated in proportion to the kernel width.

* **PSF Matching Artifacts**
    the convolution kernel used for PSF matching can introduce ringing, negative pixels, or other artifacts, especially if the target PSF differs significantly from the original or if the kernel is not properly regularized or apodized.

* **WCS Residuals**
    after applying an astrometric solution, small inaccuracies remain in the mapping between pixel and sky coordinates. these can arise from unmodeled optical distortions, atmospheric refraction effects, or fitting limitations, and cause slight misalignments when stacking images.

* **Stacking Artifacts**
    when combining multiple exposures, imperfect registration, outlier rejection, or weighting can create artifacts around stars, galaxy edges, or variable sources. these can include flux "ghosts," discontinuities, or systematic photometric errors dependent on the number of overlapping frames.


### Table 2: Pipeline-Propagated Noise & Systematics

This table focuses on noise or systematic errors introduced or amplified by the calibration and processing steps.

| effect                     | algorithms / process                           | implemented algorithm (`WeightMask`) | existing implementations               |
| :------------------------- | :--------------------------------------------- | :--------------------------------- | :---------------------------------------------- |
| **shot noise** | propagation via signal statistics | poisson term in variance map | `astropy.stats.poisson_conf_interval`, `photutils.segmentation` |
| **read noise** | propagation through pipeline | gaussian term in variance map | `astropy.nddata`, `ccdproc` |
| **bias noise propagation** | variance propagation (add var[bias])           | simplified prop. (via RN)          | `astropy.nddata.ccd_process`, `drizzlepac.astrodrizzle` |
| **bias structure residuals**| improved subtraction / modeling(X)             | - (relies on input quality)        | `ccdproc`, `pydis`, `msumastro` |
| **dark noise propagation** | variance propagation (add var[dark])           | simplified prop. (via RN)          | `astropy.nddata.ccd_process`, `caldb` |
| **dark pattern residuals** | improved subtraction / scaling(X)              | - (relies on input quality)        | `ccdproc`, `astropop`, `stsci.skypac` |
| **flat-field noise prop.** | variance propagation (full formula)          | simplified prop. ²                 | `astropy.nddata.ccd_process`, `proper` |
| **flat-fielding residuals** | illumination correction(X), sky flats(X)       | - (relies on input quality)        | `sep`/`photutils` (derive sky flats), `mcsAnaRed` |
| **defringing noise/residuals**| variance propagation (add var[fringe])(X)    | -                                  | `decam_nightmares`, `MUSE DRS`, `lpipe` |
| **non-linearity residuals** | improved correction(X), mask high counts(X)  | - (SAT mask helps)                 | `ccdproc`, `astro-conic`, `KMOS_pipeline` |
| **brighter-fatter residuals**| improved correction(X)                         | -                                  | LSST DM stack, Euclid-specific tools, `scarlet_lite` |
| **background subtraction res.**| improved modeling                              | SEP background                     | `sep`, `photutils.background`, `LAMBDAR` |
| **resampling noise correlation**| covariance tracking(X), effective variance(X)| N/A (stacking step)              | `SWarp`, `Montage`, `reproject`, `drizzlepac.astrodrizzle` |
| **resampling artifacts** | kernel choice(X), drizzling(X)                 | N/A (stacking step)              | `SWarp`, `astropy.reproject`, `STScI drizzlepac` |
| **PSF matching noise corr.** | variance convolution(X)                        | N/A (pre-stacking step)          | `psfmatch`, LSST DM stack, `HSTphot`, `mcsAnaRed` |
| **PSF matching artifacts** | kernel design(X), apodization(X)             | N/A (pre-stacking step)          | `psfmatch`, `photutils.psf`, `hotpants` |
| **WCS residuals** | better fitting(X), higher order WCS(X)         | N/A (astrometry step)            | `SCAMP`, `Gaia-oriented astrometry`, `tweakreg` |
| **stacking artifacts** | robust combining(X), consistent masking(X) | N/A (stacking step) | `SWarp`, `ccdproc.combine`, `astrodrizzle`, `IRAF imcombine` |

**Note 2:** Simplified Propagation (`simplified prop.`) means the variance formula used accounts for the *effect* of the flat value (`Flat^2` term) and basic noise sources (read noise, Poisson noise from final sky), but does not rigorously add the variance *from* the calibration frames (bias, dark, flat) themselves.


### Packages
* Maximask
* astroscrappy
* ccdproc
* cosmic-conn
* photutils
* drizzlepac
* swarp
* psfmatch
* hotpants
* tweakreg
* Montage
* casutools
* deepcr
* DeepGhostBusters
* maskfill
* Attention-Augmented-Cosmic-Ray-Detection-in-Astronomical-Images

## References
https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000502-SimulatingCosmicRays.pdf

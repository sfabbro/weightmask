"""MEF orchestration: per-HDU processing and the parallel fan-out over HDUs."""

from __future__ import annotations

import concurrent.futures
import os
import traceback
from typing import Optional, Tuple

import fitsio
import numpy as np

from .bad import compute_flat_bad_mask_cached, detect_dark_hot_pixels, global_median
from .config import resolve_config
from .errors import StageFailure
from .process import effective_tile_size, process_image
from .products import (
    ProductPublication,
    extract_individual_masks,
    flush_hdu_output,
    make_output_writers,
    rescale_confidence_to_global,
    store_output_maps,
    strip_compression_keywords,
)
from .replicas import (
    clear_chip_replicas,
    streak_catalog,
)


def process_hdu(
    hdu_sci,
    hdu_flat,
    config: dict,
    hdu_index: int,
    tile_size: int = 1024,
    flat_path: Optional[str] = None,
    hdu_badpix=None,
    extra_bad_mask: Optional[np.ndarray] = None,
    detector_prior: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Processes a single Science HDU to generate all mask and map products.

    ``extra_bad_mask`` (dark hot pixels, dead-CCD veto) is added to the flat
    and keep-map bad pixels; it never replaces the flat bad-pixel mask.
    Unreadable or inconsistent inputs raise (see ``process_image``).
    """
    config = resolve_config(config)
    hdu_name = getattr(hdu_sci, "name", f"HDU{hdu_index}")
    print(f"\n--- Processing HDU {hdu_index} ({hdu_name}) ---")

    sci_data_full = np.asarray(hdu_sci.read())
    sci_hdr = hdu_sci.read_header()

    flat_data_full = None
    if hdu_flat is not None:
        # Read the flat unconditionally: process_image uses it as the actual
        # flat field for the variance/weight math, not just for the bad mask.
        flat_data_full = np.ascontiguousarray(hdu_flat.read().astype(np.float32, copy=False))

    bad_mask = None
    if flat_data_full is not None:
        eff = effective_tile_size(tile_size, sci_data_full.shape)
        bad_mask = compute_flat_bad_mask_cached(
            flat_data_full,
            config["flat_masking"],
            eff,
            flat_path=flat_path,
            hdu_index=hdu_index,
        )

    badpix_mask = None
    if hdu_badpix is not None:
        ext = np.asarray(hdu_badpix.read())
        if ext.shape != sci_data_full.shape:
            raise ValueError(f"keep-map shape {ext.shape} != science shape {sci_data_full.shape}")
        if not np.all((ext == 0) | (ext == 1)):
            raise ValueError("keep-map must contain only 0 (bad) and 1 (good)")
        badpix_mask = ext == 0
        frac = float(np.mean(badpix_mask))
        if frac > config["flat_masking"]["dead_ccd_badpix_fraction"]:
            print(f"  NOTE: external mask flags {frac:.1%} of HDU {hdu_index} bad (dead CCD?) -- zero weight.")

    if extra_bad_mask is not None:
        badpix_mask = extra_bad_mask if badpix_mask is None else badpix_mask | extra_bad_mask

    return process_image(
        sci_data_full,
        sci_hdr,
        flat_data_full,
        config,
        tile_size,
        bad_mask=bad_mask,
        badpix_mask=badpix_mask,
        detector_prior=detector_prior,
    )


def _veto_dead_ccd_hdus(flat_bad_masks, flat_meds, flat_cfg, flat_shapes=None):
    """Zero-weight HDUs whose flat level is an outlier across the exposure.

    A whole CCD blanked by Elixir (or failed hardware) shows up as a flat
    median tens of sigma from its siblings. Such an HDU's weights would be
    garbage; mark it fully BAD explicitly instead of letting a partial mask
    through. No-op for small-HDU runs and when all medians agree. Mutates
    flat_bad_masks in place.
    """
    if not flat_cfg["dead_ccd_enable"]:
        return
    idxs = [i for i in flat_meds if np.isfinite(float(flat_meds[i]))]
    if len(idxs) < flat_cfg["dead_ccd_min_hdus"]:
        return
    meds = np.array([float(flat_meds[i]) for i in idxs])
    if not np.all(np.isfinite(meds)):
        return
    exp_med = float(np.median(meds))
    mad = float(np.median(np.abs(meds - exp_med)) * 1.4826)
    if not np.isfinite(exp_med) or exp_med <= 0:
        return
    sigma = flat_cfg["dead_ccd_mad_sigma"]
    floor = flat_cfg["dead_ccd_min_rel_dev"] * exp_med
    for i, m in zip(idxs, meds):
        if abs(m - exp_med) > max(sigma * mad, floor):
            if i in flat_bad_masks:
                flat_bad_masks[i] = np.ones_like(flat_bad_masks[i], dtype=bool)
            elif flat_shapes is not None and i in flat_shapes:
                flat_bad_masks[i] = np.ones(flat_shapes[i], dtype=bool)
            else:
                continue
            print(f"    HDU {i}: flat median {m:.3f} vs exposure {exp_med:.3f} -- flagging whole CCD BAD.")


def _resolve_max_workers(max_workers: int | None, n_hdus: int) -> int:
    """Worker count: ``None`` means up to eight; 1 or less is sequential (reproducible/CI)."""
    workers = min(8, os.cpu_count() or 4) if max_workers is None else max_workers
    return max(1, min(workers, n_hdus))


def _fits_path_of(hdul, explicit: Optional[str]) -> Optional[str]:
    if isinstance(explicit, str) and explicit:
        return explicit
    try:
        p = getattr(hdul, "_filename", None)
        if isinstance(p, bytes):
            p = p.decode()
        return p if isinstance(p, str) and p else None
    except Exception:
        return None


def _hdu_at(hdul, index: int):
    """Return hdul[index] or None if missing/unreadable."""
    if hdul is None:
        return None
    try:
        return hdul[index] if index < len(hdul) else None
    except Exception:
        return None


# Per-CCD identifier keys, most specific first. MegaCam frames carry the
# physical CCD id in CCDNAME (e.g. '8341-7-5'); CCDNAM is the older spelling
# of the same thing. 'CCD' is deliberately absent: on MegaCam it holds the
# detector model ('Marconi/EEV CCD42-90'), identical for all 36 HDUs, so
# naming from it would give every product in the file the same EXTNAME.
_CCD_IDENTIFIER_KEYS = ("CCDNAME", "CCDNAM")


def _hdu_identifier(hdu, header, index: int) -> str:
    """Output-name token for one HDU: its CCD id, else its own name, else the index.

    The input's ``EXTNAME`` is a tiling-compression artifact and fitsio's image
    HDUs expose no ``name``, so every product used to be named from its
    position alone (``MAP_HDU1``). Naming them after the CCD id instead
    (``MAP_8341-7-5``) makes each HDU self-describing.
    """
    for key in _CCD_IDENTIFIER_KEYS:
        value = None
        if header is not None:
            try:
                value = header.get(key)
            except Exception:
                value = None
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        if value is None or isinstance(value, str) and not value.strip():
            continue
        token = "-".join(str(value).split()).replace("/", "-")
        if token:
            return token
    name = getattr(hdu, "name", None)
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"HDU{index}"


def _open_hdu_handles(
    i: int,
    *,
    in_path: Optional[str],
    fl_path: Optional[str],
    bp_path: Optional[str],
    hdul_input,
    hdul_flat,
    hdul_badpix,
):
    """Yield (sci, flat, badpix, header, name) for one HDU, from path reopen or open handles.

    When ``in_path`` is set, opens per-worker FITS handles (thread-safe). Otherwise uses
    the already-open ``hdul_*`` objects. Caller must invoke the returned ``close()``.
    """
    opened: list = []

    def close():
        for handle in opened:
            try:
                handle.close()
            except Exception:
                pass

    try:
        if in_path is not None:
            fi = fitsio.FITS(in_path, "r")
            opened.append(fi)
            if i >= len(fi):
                close()
                return None, None, None, None, f"HDU{i}", close, "index out of range"
            hdu_sci = fi[i]
            if hdul_flat is not None and fl_path is not None:
                ff = fitsio.FITS(fl_path, "r")
                opened.append(ff)
                if i >= len(ff):
                    # A flat MEF shorter than the science MEF must not degrade
                    # to "no flat": process_image would substitute a unit flat
                    # and emit unflat-fielded weights that look successful.
                    # Build the reason before close(), which closes ff.
                    reason = f"flat {fl_path} has {len(ff)} HDU(s), need index {i}"
                    close()
                    return None, None, None, None, f"HDU{i}", close, reason
                hdu_flat_obj = ff[i]
            else:
                hdu_flat_obj = _hdu_at(hdul_flat, i)
            if hdul_badpix is not None and bp_path is not None:
                fb = fitsio.FITS(bp_path, "r")
                opened.append(fb)
                if i >= len(fb):
                    reason = f"keep-map {bp_path} has {len(fb)} HDU(s), need index {i}"
                    close()
                    return None, None, None, None, f"HDU{i}", close, reason
                hdu_badpix_obj = fb[i]
            else:
                hdu_badpix_obj = _hdu_at(hdul_badpix, i)
            try:
                hdu_header_raw = fi[i].read_header()
            except Exception:
                hdu_header_raw = None
            hdu_name = _hdu_identifier(hdu_sci, hdu_header_raw, i)
            return hdu_sci, hdu_flat_obj, hdu_badpix_obj, hdu_header_raw, hdu_name, close, None

        hdu_sci = hdul_input[i]
        hdu_flat_obj = _hdu_at(hdul_flat, i)
        hdu_badpix_obj = _hdu_at(hdul_badpix, i)
        try:
            hdu_header_raw = hdul_input[i].read_header()
        except Exception:
            hdu_header_raw = None
        hdu_name = _hdu_identifier(hdu_sci, hdu_header_raw, i)
        return hdu_sci, hdu_flat_obj, hdu_badpix_obj, hdu_header_raw, hdu_name, close, None
    except Exception as e:
        close()
        return None, None, None, None, f"HDU{i}", (lambda: None), str(e)


def _merge_dark_mask(i, hdu_sci, pre, *, dark_path, hdul_dark, dark_cfg):
    """Fold the dark frame's hot-pixel mask for HDU ``i`` into ``pre``.

    ``dark_path`` (a per-worker reopen) wins over the already-open ``hdul_dark``.
    Every failure declines to add a dark mask and returns ``pre`` unchanged: an
    absent, short or unreadable dark must not fail an HDU that is otherwise
    fine.
    """
    dark_hdu = None
    fd_local = None
    close_fd = False
    try:
        if dark_path is not None:
            try:
                fd_local = fitsio.FITS(dark_path, "r")
                close_fd = True
                if i < len(fd_local):
                    dark_hdu = fd_local[i]
                else:
                    print(
                        f"    WARNING: dark frame {dark_path} has {len(fd_local)} HDU(s), "
                        f"none for index {i}. No dark hot-pixel mask for this CCD."
                    )
                    dark_hdu = None
            except Exception as e:
                print(f"    WARNING: dark frame HDU {i} unavailable: {e}. Skipping dark mask.")
                dark_hdu = None
        elif hdul_dark is not None:
            try:
                if i < len(hdul_dark):
                    dark_hdu = hdul_dark[i]
                else:
                    print(
                        f"    WARNING: dark frame has {len(hdul_dark)} HDU(s), "
                        f"none for index {i}. No dark hot-pixel mask for this CCD."
                    )
                    dark_hdu = None
            except Exception as e:
                print(f"    WARNING: dark frame HDU {i} unavailable: {e}. Skipping dark mask.")
                dark_hdu = None
        if dark_hdu is None:
            return pre
        try:
            dark_data = np.ascontiguousarray(dark_hdu.read().astype(np.float32, copy=False))
            dark_hot = detect_dark_hot_pixels(dark_data, dark_cfg)
            sci_shape = tuple(hdu_sci.get_dims())
            if dark_hot.shape != sci_shape:
                print(f"    Skipping dark mask for HDU {i}: shape mismatch.")
            else:
                if pre is not None and pre.shape != sci_shape:
                    pre = None
                pre = dark_hot if pre is None else (pre | dark_hot)
                print(f"    Dark frame adds {int(np.count_nonzero(dark_hot))} hot pixels to HDU {i}.")
        except Exception as e:
            print(f"    WARNING: dark mask failed for HDU {i}: {e}")
        return pre
    finally:
        if close_fd and fd_local is not None:
            try:
                fd_local.close()
            except Exception:
                pass


def _process_all_hdus_to_paths(
    hdus_to_process: list,
    hdul_input,
    hdul_flat,
    config: dict,
    paths: dict,
    flat_path: Optional[str] = None,
    hdul_badpix=None,
    hdul_dark=None,
    max_workers: int | None = None,
    input_path: Optional[str] = None,
    badpix_path: Optional[str] = None,
    dark_path: Optional[str] = None,
    detector_priors: Optional[dict] = None,
) -> int:
    """Process every requested HDU, streaming each HDU's outputs to disk.
    Each HDU's maps are written to their output files as soon as they are
    produced instead of being accumulated in memory. A 36-CCD MegaPrime MEF
    therefore only ever holds a single CCD's products at a time (~a few
    hundred MB) rather than the whole MEF's (~7 GB), which previously blew
    through the container memory limit when several exposures were processed
    concurrently and OOM-killed the batch.
    Parallel by default: per-HDU ThreadPoolExecutor; single-HDU inputs run
    inline (per-file fast path). Compute runs concurrently, but only one
    worker-window of HDUs sits ahead of the writer, and writes are applied in
    HDU-index order for deterministic output.
    """
    eff_workers = _resolve_max_workers(max_workers, len(hdus_to_process))

    def _usable(p: Optional[str]) -> Optional[str]:
        return p if isinstance(p, str) and p and os.path.exists(p) else None

    in_path_u = _usable(_fits_path_of(hdul_input, input_path))
    fl_path_u = _usable(_fits_path_of(hdul_flat, flat_path)) if hdul_flat is not None else None
    bp_path_u = _usable(_fits_path_of(hdul_badpix, badpix_path)) if hdul_badpix is not None else None
    dk_path_u = _usable(_fits_path_of(hdul_dark, dark_path)) if hdul_dark is not None else None
    process_success_count = 0
    writers = make_output_writers(paths, hdul_input, config)
    flat_bad_masks = {}
    if hdul_flat is not None and flat_path:
        print(f"  Pre-computing flat bad masks for {len(hdus_to_process)} HDUs...")
        flat_cfg = config["flat_masking"]
        flat_meds: dict = {}
        flat_shapes: dict = {}
        for i in hdus_to_process:
            if i < len(hdul_flat):
                try:
                    flat_data = np.ascontiguousarray(hdul_flat[i].read().astype(np.float32, copy=False))
                    flat_meds[i] = global_median(flat_data)
                    flat_shapes[i] = flat_data.shape
                except Exception as e:
                    print(f"    WARNING: Failed to read flat for HDU {i}: {e}")
        # Tactic D: medians stay sequential (cheap reads for the dead-CCD
        # veto); the 15x15 flat-mask medians run in the ThreadPoolExecutor
        # workers (process_hdu).
        _veto_dead_ccd_hdus(flat_bad_masks, flat_meds, flat_cfg, flat_shapes=flat_shapes)
        print(f"  Flat medians ready for {len(flat_meds)} HDUs ({len(flat_bad_masks)} vetoed dead)")
    conf_scope = config["confidence_params"]["normalize_scope"]
    dark_cfg_global = config["dark_masking"]

    def _compute_one(i):
        pre = flat_bad_masks.get(i) if flat_bad_masks else None
        close = None

        try:
            hdu_sci, hdu_flat_obj, hdu_badpix_obj, hdu_header_raw, hdu_name, close, err = _open_hdu_handles(
                i,
                in_path=in_path_u,
                fl_path=fl_path_u,
                bp_path=bp_path_u,
                hdul_input=hdul_input,
                hdul_flat=hdul_flat,
                hdul_badpix=hdul_badpix,
            )
            if err is not None or hdu_sci is None:
                print(f"Skipping HDU {i}: {err or 'cannot access input HDU'}.")
                return (i, None, None, hdu_name)
            pre2 = _merge_dark_mask(i, hdu_sci, pre, dark_path=dk_path_u, hdul_dark=hdul_dark, dark_cfg=dark_cfg_global)
            prior = detector_priors.get(i) if isinstance(detector_priors, dict) else None
            result = process_hdu(
                hdu_sci,
                hdu_flat_obj,
                config,
                i,
                flat_path=flat_path,
                hdu_badpix=hdu_badpix_obj,
                extra_bad_mask=pre2,
                detector_prior=prior,
            )
            return (i, result, hdu_header_raw, hdu_name)
        except StageFailure as exc:
            if exc.hdu_index is None:
                exc.hdu_index = i
            raise
        except (OSError, ValueError, RuntimeError, fitsio.FITSFormatError) as e:
            print(f"Skipping HDU {i}: {e}")
            return (i, None, None, f"HDU{i}")
        except Exception as e:
            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
            return (i, None, None, f"HDU{i}")
        finally:
            if close is not None:
                close()

    streak_catalogs: list = []

    def _emit(i, result, hdu_header_raw, hdu_name_raw):
        """Store and flush one HDU's products (releases its arrays immediately)."""
        nonlocal process_success_count
        try:
            if result is None:
                print(f"Skipping HDU {i} due to processing errors.")
                return
            (mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info) = result
            bad_mask, sat_mask, cr_mask, obj_mask, streak_mask, nodata_mask = extract_individual_masks(
                header_info, mask_data
            )
            streak_catalogs.extend(streak_catalog(i, mask_data, hdu_header_raw))
            hdu_name = hdu_name_raw
            hdu_header = hdu_header_raw
            if hdu_header is None:
                hdu_header = fitsio.FITSHDR()
            hdu_header = strip_compression_keywords(hdu_header)
            sky_cards = (header_info or {}).get("sky_cards") or {}
            sky_header = {**(hdu_header or {}), **sky_cards} if sky_cards else hdu_header
            hdu_output: dict = {}
            store_output_maps(
                hdu_output,
                i,
                hdu_header,
                hdu_name,
                config,
                confidence_map,
                weight_map,
                mask_data,
                inv_var_data,
                sky_map,
                bad_mask,
                sat_mask,
                cr_mask,
                obj_mask,
                streak_mask,
                nodata_mask,
                paths,
                sky_header=sky_header,
            )
            flush_hdu_output(writers, hdu_output, i)
            process_success_count += 1
        except Exception as e:
            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")

    # Each HDU leaves memory as soon as it is produced, and at most
    # ``workers_n`` HDUs are ever computed ahead of the writer, so peak memory
    # stays at a few CCDs instead of the whole MEF's (~8 GB for 36 MegaPrime
    # CCDs). Writes still happen in HDU order, so output files are identical to
    # the previously fully-buffered path.
    workers_n = max(1, min(eff_workers, len(hdus_to_process)))
    if workers_n <= 1:
        for i in hdus_to_process:
            _emit(i, *_compute_one(i)[1:])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers_n) as ex:
            in_flight: dict = {}
            queue = iter(hdus_to_process)

            def _submit_next() -> bool:
                i = next(queue, None)
                if i is None:
                    return False
                in_flight[i] = ex.submit(_compute_one, i)
                return True

            for _ in range(workers_n):
                if not _submit_next():
                    break
            for i in hdus_to_process:
                fut = in_flight.pop(i, None)
                if fut is None:  # pragma: no cover - the window always holds the next index
                    continue
                try:
                    _i, _res, _hdr, _nm = fut.result()
                except StageFailure:
                    for pending in in_flight.values():
                        pending.cancel()
                    raise
                except Exception as e:
                    print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
                    _i, _res, _hdr, _nm = i, None, None, f"HDU{i}"
                _emit(_i, _res, _hdr, _nm)
                _submit_next()
    if not clear_chip_replicas(streak_catalogs, writers, config):
        return 0
    if conf_scope == "per_exposure" and not rescale_confidence_to_global(paths, writers, config):
        return 0
    return process_success_count


def process_all_hdus(
    hdus_to_process: list,
    hdul_input,
    hdul_flat,
    config: dict,
    paths: dict,
    flat_path: Optional[str] = None,
    hdul_badpix=None,
    hdul_dark=None,
    max_workers: int | None = None,
    input_path: Optional[str] = None,
    badpix_path: Optional[str] = None,
    dark_path: Optional[str] = None,
    detector_priors: Optional[dict] = None,
) -> int:
    """Process ``hdus_to_process`` and publish every product atomically; returns the HDUs published.

    ``config`` is merged over the packaged defaults. ``max_workers=None`` uses
    up to eight threads; 1 runs the HDUs sequentially.
    """
    config = resolve_config(config)
    publication = ProductPublication(paths)
    try:
        count = _process_all_hdus_to_paths(
            hdus_to_process,
            hdul_input,
            hdul_flat,
            config,
            publication.paths,
            flat_path=flat_path,
            hdul_badpix=hdul_badpix,
            hdul_dark=hdul_dark,
            max_workers=max_workers,
            input_path=input_path,
            badpix_path=badpix_path,
            dark_path=dark_path,
            detector_priors=detector_priors,
        )
        if count != len(hdus_to_process):
            return count
        publication.validate(hdus_to_process)
        publication.promote()
        return count
    except StageFailure:
        raise
    except (OSError, ValueError) as exc:
        print(f"  ERROR: product publication failed: {exc}")
        return 0
    finally:
        publication.cleanup()

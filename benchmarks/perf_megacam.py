#!/usr/bin/env python3
"""MegaCam 10-exposure per-step profile + speedup harness.

Resolves 10 long CFHT MegaPrime exposures, processes all science HDUs of
each exposure through the real ``mef.process_all_hdus`` entry path, and
writes per-stage timing breakdowns.

Dataset layout:
  benchmark_data/megacam/perf/<safe_id>.fits.fz   (raw, unmodified)
  test_outputs/perf/exposures.json                (pinned 10, reused verbatim)
  test_outputs/perf/megacam_perf.json / .md       (profile report)
  test_outputs/perf/cprofile_hdu.txt              (single-HDU cProfile)

Usage:
  pixi run python benchmarks/perf_megacam.py --resolve-only
  pixi run python benchmarks/perf_megacam.py --workers 1
  pixi run python benchmarks/perf_megacam.py --exposure-file /path/to/science.fits.fz --hdu-limit 2 --no-cprofile
  pixi run python benchmarks/perf_megacam.py --repeats 5 --seed 0 --no-cprofile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import shutil
import sys
import threading
import time
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import perf_protocol
PERF_DATA_DIR = ROOT / "benchmark_data" / "megacam" / "perf"
OUT_DIR = ROOT / "test_outputs" / "perf"
EXPOSURES_JSON = OUT_DIR / "exposures.json"
PERF_JSON = OUT_DIR / "megacam_perf.json"
PERF_MD = OUT_DIR / "megacam_perf.md"
CPROFILE_TXT = OUT_DIR / "cprofile_hdu.txt"
CONFIG_PATH = ROOT / "weightmask.yml"
FLAT_FILE = PERF_DATA_DIR / "flat_08Bm01_r.fits.fz"

BASE_WHERE = (
    "Observation.collection = 'CFHT' AND Observation.instrument_name = 'MegaPrime' AND Observation.type = 'OBJECT'"
)
BASE_SELECT = (
    "SELECT TOP {top} Plane.publisherID FROM caom2.Plane AS Plane "
    "JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID "
    "WHERE {where} ORDER BY Plane.publisherID"
)

# Plan-ordered stage keys (Step 2). Aggregator tolerates missing keys.
STAGE_KEYS = [
    "bad_flat",
    "saturation",
    "background_prelim",
    "bleed",
    "cosmics",
    "bgobj_loop",
    "background_final",
    "variance",
    "streaks",
    "weight_confidence",
    "sky_mesh",
    "hdu_total",
]


def _safe_id(publisher_id: str) -> str:
    """Map a publisherID (IVO URI or plain CADC id) to a filename-safe id."""
    s = str(publisher_id).strip()
    if "/" in s:
        s = s.split("/")[-1]
    if "?" in s:
        s = s.split("?")[-1]
    return s


def _quarantine_invalid(path: Path) -> None:
    from tests.benchmarks.download_data import _quarantine_invalid_file

    _quarantine_invalid_file(path)


def _download_http(url: str, out_path: Path) -> bool:
    from tests.benchmarks.download_data import download_http

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return download_http(url, out_path)


def _validate_megacam(path: Path) -> tuple[bool, str | None]:
    """Reuse the manifest INSTRUME/DETECTOR check via validate_case_file."""
    from tests.benchmarks.download_data import validate_case_file

    return validate_case_file({"expected_instrument": "MegaPrime", "expected_detector": "MegaCam"}, str(path))


def _exptime_of_file(path: Path) -> float | None:
    try:
        import fitsio
    except Exception:
        return None
    try:
        with fitsio.FITS(str(path)) as hdul:
            # Prefer first 2-D science HDU, fall back to primary.
            for hdu in hdul[1:]:
                try:
                    hdr = hdu.read_header()
                except Exception:
                    continue
                if "EXPTIME" in hdr:
                    try:
                        return float(hdr["EXPTIME"])
                    except (TypeError, ValueError):
                        continue
            try:
                hdr0 = hdul[0].read_header()
                if "EXPTIME" in hdr0:
                    return float(hdr0["EXPTIME"])
            except Exception:
                pass
    except Exception as e:
        print(f"  EXPTIME read failed for {path}: {e}")
        return None
    return None


def _query_candidates(top: int, with_time_filter: bool):
    """Run the manifest-proven CAOM2 query; returns (results_table, used_filter)."""
    from astroquery.cadc import Cadc

    cadc = Cadc()
    if with_time_filter:
        # Plan literal first attempt (camelCase). Known to fail with
        # "Column: [timeExposure] not found in TapSchema"; caller falls back.
        where = BASE_WHERE + " AND Plane.timeExposure > 500"
    else:
        where = BASE_WHERE
    query = BASE_SELECT.format(top=top, where=where)
    results = cadc.exec_sync(query)
    return cadc, results


def _resolve_with_server_prefilter(top: int):
    """Efficient candidate query using the real TAP column Plane.time_exposure.

    The plan's ``Plane.timeExposure`` predicate is invalid (TAP error
    "Column: [timeExposure] not found in TapSchema"); the schema column is
    ``time_exposure`` (verified via TAP_SCHEMA.columns). This keeps the
    plan's server-side prefilter intent while the client-side EXPTIME>=500
    header check remains the binding gate.
    """
    from astroquery.cadc import Cadc

    cadc = Cadc()
    where = BASE_WHERE + " AND Plane.time_exposure > 500"
    query = BASE_SELECT.format(top=top, where=where)
    results = cadc.exec_sync(query)
    return cadc, results


def resolve_exposures(force: bool = False) -> list[dict]:
    """Pin 10 long exposures; write and return the exposures.json records.

    CANFAR jobs pre-stage a job-local exposures.json (1-3 manifest records);
    any non-empty reuse with all files present is honored, not just the
    10-exposure baseline.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PERF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EXPOSURES_JSON.exists() and not force:
        try:
            records = json.loads(EXPOSURES_JSON.read_text())
            if isinstance(records, list) and len(records) >= 1:
                ok = True
                for rec in records:
                    fp = ROOT / rec.get("file", "")
                    if not fp.exists():
                        ok = False
                        break
                if ok:
                    print(f"Reusing pinned exposures from {EXPOSURES_JSON}")
                    return records
                print("Pinned exposures incomplete (missing files); re-resolving.")
        except Exception as e:
            print(f"Could not reuse {EXPOSURES_JSON}: {e}; re-resolving.")

    # Step 1 order: first attempt with the plan-literal predicate to confirm
    # the TAP error, then fall back per the contingency.
    cadc = None
    results = None
    try:
        cadc, results = _query_candidates(30, with_time_filter=True)
        print("  Plan-literal Plane.timeExposure query unexpectedly succeeded.")
    except Exception as e:
        print(f"  Plan-literal timeExposure predicate failed as expected: {e}")
        print("  Falling back: server prefilter on Plane.time_exposure > 500")
        print("  (schema-verified column; client EXPTIME>=500 remains the gate).")
        cadc, results = _resolve_with_server_prefilter(30)

    publisher_ids = [str(v) for v in list(results["publisherID"])]
    print(f"  Candidate publisherIDs: {len(publisher_ids)}")

    # Prefer calibrated 'p' products; dedupe by observation (numeric prefix)
    # so o/p pairs of one observation count once. Keep publisherID order.
    seen_obs: set[str] = set()
    ordered: list[str] = []
    deferred_o: list[str] = []
    for pid in publisher_ids:
        safe = _safe_id(pid)
        obs = safe[:-1] if safe and safe[-1] in ("o", "p") else safe
        if obs in seen_obs:
            continue
        if safe.endswith("p"):
            ordered.append(pid)
            seen_obs.add(obs)
        else:
            deferred_o.append(pid)
    for pid in deferred_o:
        safe = _safe_id(pid)
        obs = safe[:-1] if safe and safe[-1] in ("o", "p") else safe
        if obs not in seen_obs:
            ordered.append(pid)
            seen_obs.add(obs)

    # Resolve data URLs in candidate order (single get_data_urls call).
    url_by_pid: dict[str, str] = {}
    try:
        urls = cadc.get_data_urls(results)
        for pid, url in zip(publisher_ids, urls):
            url_by_pid[pid] = url
    except Exception as e:
        print(f"  get_data_urls failed: {e}")

    kept: list[dict] = []
    tried: set[str] = set(ordered)

    def _try_candidates(cands: list[str]) -> None:
        for pid in cands:
            if len(kept) >= 10:
                break
            safe = _safe_id(pid)
            out_path = PERF_DATA_DIR / f"{safe}.fits.fz"
            url = url_by_pid.get(pid)
            if out_path.exists():
                print(f"[{len(kept) + 1}/10] Exists: {safe}")
            else:
                if url is None:
                    # Single-row fallback: query this publisherID for its URL.
                    try:
                        from astroquery.cadc import Cadc as _Cadc

                        c2 = _Cadc()
                        q1 = (
                            "SELECT TOP 1 Plane.publisherID FROM caom2.Plane AS Plane "
                            f"WHERE Plane.publisherID = '{pid}'"
                        )
                        r1 = c2.exec_sync(q1)
                        u1 = c2.get_data_urls(r1)
                        url = u1[0] if u1 else None
                    except Exception as e:
                        print(f"  URL resolve failed for {pid}: {e}")
                        continue
                # Proven direct-pub path for calibrated 'p' products (download_cadc).
                candidates = []
                if safe.endswith("p"):
                    candidates.append(f"https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/{safe}.fits.fz")
                if url is not None:
                    candidates.append(url)
                ok = False
                for cand in candidates:
                    if _download_http(cand, out_path):
                        ok = True
                        break
                if not ok:
                    continue
            valid, reason = _validate_megacam(out_path)
            if not valid:
                print(f"  Validation failed for {safe}: {reason}")
                _quarantine_invalid(out_path)
                continue
            exptime = _exptime_of_file(out_path)
            print(f"  {safe}: EXPTIME={exptime}")
            if exptime is None or exptime < 500:
                print(f"  Skipping {safe}: EXPTIME {exptime} < 500 s.")
                continue
            try:
                size = out_path.stat().st_size
            except OSError:
                size = 0
            kept.append(
                {
                    "publisherID": pid,
                    "safe_id": safe,
                    "file": str(out_path.relative_to(ROOT)),
                    "exptime": float(exptime),
                    "size_bytes": int(size),
                }
            )
            print(f"  Kept [{len(kept)}/10]: {safe} ({exptime:.1f}s)")

    _try_candidates(ordered)
    if len(kept) < 10:
        print(f"Only {len(kept)}/10 after TOP-30; extending to TOP-100.")
        _, results100 = _resolve_with_server_prefilter(100)
        pids100 = [str(v) for v in list(results100["publisherID"])]
        try:
            urls100 = cadc.get_data_urls(results100)
            for pid, url in zip(pids100, urls100):
                url_by_pid.setdefault(pid, url)
        except Exception as e:
            print(f"  get_data_urls(TOP-100) failed: {e}")
        extra: list[str] = []
        seen2 = set(tried)
        # Rebuild order preference for the TOP-100 list.
        for pid in pids100:
            if pid in seen2:
                continue
            safe = _safe_id(pid)
            obs = safe[:-1] if safe and safe[-1] in ("o", "p") else safe
            if safe.endswith("p") and obs not in {k["safe_id"][:-1] for k in kept}:
                extra.append(pid)
                seen2.add(pid)
        for pid in pids100:
            if pid in seen2:
                continue
            safe = _safe_id(pid)
            if safe.endswith("p"):
                extra.append(pid)
                seen2.add(pid)
        for pid in pids100:
            if pid in seen2:
                continue
            extra.append(pid)
        _try_candidates(extra)

    if len(kept) < 10:
        raise SystemExit(f"Could only pin {len(kept)}/10 long exposures; aborting.")
    kept = kept[:10]
    EXPOSURES_JSON.write_text(json.dumps(kept, indent=2) + "\n")
    print(f"Wrote {EXPOSURES_JSON} with {len(kept)} exposures.")
    return kept


def _install_timing_collector():
    """Wrap process_image (both bindings) to record per-HDU timings."""
    import weightmask.mef as mef_mod
    import weightmask.process as proc_mod

    records: list[dict] = []
    lock = threading.Lock()
    orig = proc_mod.process_image

    def wrapper(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            timings = None
            if result is not None and len(result) == 6 and isinstance(result[5], dict):
                timings = dict(result[5].get("timings") or {})
            shape = None
            try:
                shape = tuple(args[0].shape) if args else None
            except Exception:
                shape = None
            with lock:
                records.append({"timings": timings or {}, "shape": shape})
        except Exception:
            pass
        return result

    proc_mod.process_image = wrapper
    mef_mod.process_image = wrapper
    return records, orig, proc_mod, mef_mod


def _uninstall_timing_collector(orig, proc_mod, mef_mod):
    proc_mod.process_image = orig
    mef_mod.process_image = orig


def _set_dotted(container: dict, dotted: str, value) -> None:
    """Set ``a.b.c=value`` creating intermediate dicts (CANFAR --config-set)."""
    node = container
    *parts, leaf = dotted.split(".")
    for part in parts:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[leaf] = value


def _parse_config_sets(pairs: list[str] | None) -> dict:
    """Parse repeatable ``dotted.key=value`` (values YAML-parsed, deep via _set_dotted)."""
    import yaml as _yaml

    overrides: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--config-set needs dotted.key=value, got {pair!r}")
        dotted, raw = pair.split("=", 1)
        dotted = dotted.strip()
        if not dotted:
            raise SystemExit(f"--config-set needs dotted.key=value, got {pair!r}")
        try:
            value = _yaml.safe_load(raw)
        except Exception:
            value = raw
        _set_dotted(overrides, dotted, value)
    return overrides


def _deep_merge(base: dict, over: dict) -> dict:
    """Deep-merge ``over`` into ``base`` (dicts recurse; scalars/lists replace)."""
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _process_one_exposure(
    rec,
    workers: int,
    out_suffix: str = "",
    *,
    output_dir: Path | None = None,
    include_worker_tag: bool = True,
    flat: str | None = None,
    write_mask: bool = False,
    file_tag: str = "",
    config_overrides: dict | None = None,
    hdu_limit: int = 0,
    all_products: bool = False,
) -> dict:
    import argparse as _ap

    import fitsio
    import yaml

    from weightmask import cli, mef

    in_path = ROOT / rec["file"]
    safe = rec["safe_id"]
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    if config_overrides:
        config = _deep_merge(config, config_overrides)
    hdul_input = fitsio.FITS(str(in_path), "r")
    try:
        hdus = cli.get_hdus_to_process(hdul_input, None)
    except Exception as e:
        hdul_input.close()
        raise SystemExit(f"get_hdus_to_process failed for {safe}: {e}")
    if hdu_limit > 0:
        hdus = hdus[:hdu_limit]
    nhdus_expected = len(hdus)
    shapes = []
    for i in hdus:
        try:
            info = hdul_input[i].get_info()
            dims = info.get("dims") or []
            if len(dims) >= 2:
                shapes.append((int(dims[-2]), int(dims[-1])))
        except Exception:
            continue
    try:
        file_bytes = in_path.stat().st_size
    except OSError:
        file_bytes = 0

    worker_tag = f".w{workers}" if include_worker_tag else ""
    tag = f"{safe}{out_suffix}{file_tag}{worker_tag}"
    if flat:
        tag += ".flat"
    output_dir = Path(output_dir or OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_all = bool(all_products)
    args = _ap.Namespace(
        output_map=str(output_dir / f"{tag}.weight.fits"),
        output_mask=str(output_dir / f"{tag}.mask.fits") if (write_mask or write_all) else None,
        output_invvar=str(output_dir / f"{tag}.ivar.fits") if write_all else None,
        output_sky=str(output_dir / f"{tag}.sky.fits") if write_all else None,
        output_weight_raw=str(output_dir / f"{tag}.weight_raw.fits") if write_all else None,
        individual_masks=write_all,
        max_workers=workers,
        tile_size=1024,
    )
    from weightmask.cli import determine_output_paths

    paths = determine_output_paths(args, str(in_path), config)
    if not write_all:
        for key in ("out_invvar_path", "out_sky_path", "out_weight_raw_path"):
            paths[key] = None
        if not write_mask:
            paths["out_mask_path"] = None
        paths["individual_mask_paths"] = {}

    hdul_flat = fitsio.FITS(str(flat), "r") if flat else None
    records, orig, proc_mod, mef_mod = _install_timing_collector()
    try:
        import resource as _resource

        cpu_start = _resource.getrusage(_resource.RUSAGE_SELF)
        cpu_start_s = float(cpu_start.ru_utime + cpu_start.ru_stime)
    except Exception:
        cpu_start_s = None
    t0 = time.perf_counter()
    try:
        n_ok = mef.process_all_hdus(
            hdus,
            hdul_input,
            hdul_flat,
            config,
            paths,
            args,
            flat_path=str(flat) if flat else None,
            max_workers=workers,
            input_path=str(in_path),
        )
    finally:
        wall = time.perf_counter() - t0
        try:
            import resource as _resource

            _ru = _resource.getrusage(_resource.RUSAGE_SELF)
            # Linux ru_maxrss is KiB; macOS reports bytes.
            _rss_kb = float(_ru.ru_maxrss) / 1024.0 if sys.platform == "darwin" else float(_ru.ru_maxrss)
            _cpu_now = float(_ru.ru_utime + _ru.ru_stime)
            _cpu_s = _cpu_now - cpu_start_s if cpu_start_s is not None else 0.0
        except Exception:
            _rss_kb, _cpu_s = 0.0, 0.0
        _uninstall_timing_collector(orig, proc_mod, mef_mod)
        try:
            hdul_input.close()
        except Exception:
            pass
        if hdul_flat is not None:
            try:
                hdul_flat.close()
            except Exception:
                pass
    stage_totals: dict[str, float] = {}
    mpix = 0.0
    for r in records:
        for k, v in (r["timings"] or {}).items():
            try:
                stage_totals[k] = stage_totals.get(k, 0.0) + float(v)
            except (TypeError, ValueError):
                continue
        sh = r.get("shape")
        if sh and len(sh) == 2:
            try:
                mpix += float(sh[0] * sh[1]) / 1e6
            except (TypeError, ValueError):
                pass
    if mpix == 0.0 and shapes:
        mpix = sum(h * w for h, w in shapes) / 1e6
    return {
        "publisherID": rec["publisherID"],
        "safe_id": safe,
        "file": rec["file"],
        "flat_file": str(flat) if flat else None,
        "exptime": rec.get("exptime"),
        "size_bytes": file_bytes,
        "nhdus_expected": nhdus_expected,
        "nhdus_processed": int(n_ok),
        "nhdus_timed": len(records),
        "wall_s": float(wall),
        "mpix": float(mpix),
        "stage_totals": stage_totals,
        "peak_rss_kb": float(_rss_kb),
        "rss_basis": "process-lifetime-high-water",
        "cpu_s": float(_cpu_s),
    }


def _aggregate_report(per_exp: list[dict], total_wall: float, *, extra_header: dict | None = None) -> dict:
    import os as _os

    totals: dict[str, float] = {}
    n_hdus = 0
    mpix_total = 0.0
    for e in per_exp:
        n_hdus += int(e.get("nhdus_processed", 0))
        mpix_total += float(e.get("mpix", 0.0))
        for k, v in (e.get("stage_totals") or {}).items():
            totals[k] = totals.get(k, 0.0) + float(v)
    hdu_wall = totals.get("hdu_total", sum(totals.values()))
    stages = {}
    for k, v in totals.items():
        share = (v / hdu_wall) if hdu_wall > 0 else 0.0
        stages[k] = {
            "total_s": float(v),
            "mean_per_hdu_s": float(v / n_hdus) if n_hdus else 0.0,
            "share": float(share),
        }

    # ``total_wall`` times only the work done in *this* process, but resumed
    # exposures contribute their pixels and per-HDU stage times from an earlier
    # run. Dividing one by the other yields a throughput that is off by orders
    # of magnitude (a fully-resumed run reported 6.4e7 Mpix/s against a true
    # 0.073), so refuse to report it rather than print a physical impossibility.
    n_resumed = sum(1 for e in per_exp if e.get("resumed"))
    warnings: list[str] = []
    if n_resumed:
        warnings.append(
            f"{n_resumed}/{len(per_exp)} exposures were resumed from checkpoint: "
            f"total_wall_s covers only this process, so throughput is not reported; "
            f"checkpoint source/config/input identity is unverified, so resumed stage rows are historical evidence"
        )
    if not n_resumed and total_wall > 0 and hdu_wall > 1.5 * total_wall:
        warnings.append(
            f"hdu_wall_s={hdu_wall:.1f} exceeds 1.5x total_wall_s={total_wall:.1f}; "
            f"the sequential pass and the per-HDU stage totals disagree"
        )
    throughput_comparable = not n_resumed
    mpix_per_s = float(mpix_total / total_wall) if (throughput_comparable and total_wall > 0) else None

    header = {
        "arch": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": _os.cpu_count(),
        "config": "weightmask.yml",
        "n_exposures": len(per_exp),
        "n_hdus": n_hdus,
    }
    if extra_header:
        header.update(extra_header)
    return {
        "header": header,
        "total_wall_s": float(total_wall),
        "hdu_wall_s": float(hdu_wall),
        "mpix_total": float(mpix_total),
        "mpix_per_s": mpix_per_s,
        "n_resumed_exposures": int(n_resumed),
        "warnings": warnings,
        "stages": stages,
        "per_exposure": per_exp,
        "cpu_basis": "delta-v3-timed-region",
    }


def _provenance(records: list[dict], config_overrides: dict, flat: str | None, thread_env: dict[str, str]) -> dict:
    inputs = []
    for record in records:
        path = ROOT / record["file"] if not Path(record["file"]).is_absolute() else Path(record["file"])
        if path.exists():
            inputs.append({"path": str(path), "sha256": perf_protocol.sha256_path(path)})
    configuration = [{"path": str(CONFIG_PATH), "sha256": perf_protocol.sha256_path(CONFIG_PATH)}]
    if flat and Path(flat).exists():
        configuration.append({"path": str(Path(flat)), "sha256": perf_protocol.sha256_path(flat)})
    return {
        "input_hashes": inputs,
        "configuration_hashes": configuration,
        "config_overrides": config_overrides,
        "thread_environment": thread_env,
    }


def _protocol_overrides(config_overrides: dict, cache_dir: Path) -> dict:
    import copy

    overrides = copy.deepcopy(config_overrides)
    _set_dotted(overrides, "flat_masking.bad_mask_cache_dir", str(cache_dir))
    return overrides


def _run_protocol(
    records: list[dict],
    workers: int,
    *,
    treatment_workers: int | None = None,
    repeats: int,
    seed: int,
    flat: str | None,
    write_mask: bool,
    config_overrides: dict,
    hdu_limit: int,
    all_products: bool,
) -> dict:
    repeats = perf_protocol.validate_repeats(repeats)
    treatment_workers = workers if treatment_workers is None else treatment_workers
    schedule = perf_protocol.interleaved_schedule(
        repeats,
        seed=seed,
        arms=("baseline", "treatment"),
        cold_repetitions=1,
    )
    protocol_root = OUT_DIR / "protocol" / f"workers-{workers}-vs-{treatment_workers}"
    runs: list[dict] = []
    samples = {"cold": {"baseline": [], "treatment": []}, "warm": {"baseline": [], "treatment": []}}
    for arm in ("baseline", "treatment"):
        arm_workers = workers if arm == "baseline" else treatment_workers
        warm_cache = perf_protocol.cache_directory(protocol_root / "cache", "warm", arm, 0)
        prewarm_dir = protocol_root / f"prewarm-{arm}"
        for record in records:
            _process_one_exposure(
                record,
                arm_workers,
                output_dir=prewarm_dir,
                include_worker_tag=False,
                flat=flat,
                write_mask=write_mask,
                config_overrides=_protocol_overrides(config_overrides, warm_cache),
                hdu_limit=hdu_limit,
                all_products=all_products,
            )
        shutil.rmtree(prewarm_dir, ignore_errors=True)
    for repetition, arm in schedule:
        cache = "cold" if not any(run["arm"] == arm for run in runs) else "warm"
        cache_dir = perf_protocol.cache_directory(protocol_root / "cache", cache, arm, repetition)
        run_dir = protocol_root / f"run-{repetition}-{arm}"
        started = time.perf_counter()
        try:
            arm_workers = workers if arm == "baseline" else treatment_workers
            per_exposure = [
                _process_one_exposure(
                    record,
                    arm_workers,
                    output_dir=run_dir,
                    include_worker_tag=False,
                    flat=flat,
                    write_mask=write_mask,
                    config_overrides=_protocol_overrides(config_overrides, cache_dir),
                    hdu_limit=hdu_limit,
                    all_products=all_products,
                )
                for record in records
            ]
        finally:
            if cache == "cold":
                perf_protocol.cleanup_cache_directory(cache_dir)
        elapsed = time.perf_counter() - started
        samples[cache][arm].append(elapsed)
        runs.append(
            {
                "arm": arm,
                "cache": cache,
                "repetition": repetition,
                "elapsed_s": float(elapsed),
                "output_dir": str(run_dir),
                "per_exposure": per_exposure,
            }
        )

    correctness = True
    for repetition in range(repeats):
        arms = [run for run in runs if run["repetition"] == repetition]
        if len(arms) != 2:
            correctness = False
            continue
        left, right = arms
        expected = _expected_product_names(
            records,
            workers=workers,
            flat=flat,
            write_mask=write_mask,
            all_products=all_products,
            include_worker_tag=False,
        )
        comparison = _compare_products(
            Path(left["output_dir"]),
            Path(right["output_dir"]),
            expected_files=expected,
        )
        correctness = correctness and bool(comparison["ok"])
    for arm in ("baseline", "treatment"):
        arm_runs = [run for run in runs if run["arm"] == arm]
        if arm_runs:
            reference = Path(arm_runs[0]["output_dir"])
            correctness = correctness and all(
                _compare_products(
                    Path(run["output_dir"]),
                    reference,
                    expected_files=_expected_product_names(
                        records,
                        workers=workers,
                        flat=flat,
                        write_mask=write_mask,
                        all_products=all_products,
                        include_worker_tag=False,
                    ),
                )["ok"]
                for run in arm_runs[1:]
            )
    perf_protocol.require_correctness_equivalence({"baseline": correctness, "treatment": correctness})
    warm_treatment = [run for run in runs if run["arm"] == "treatment" and run["cache"] == "warm"]
    target = statistics.median(run["elapsed_s"] for run in warm_treatment)
    selected = min(warm_treatment, key=lambda run: abs(run["elapsed_s"] - target))
    return {
        "protocol": {
            "repeats": repeats,
            "seed": seed,
            "schedule": [[int(repetition), arm] for repetition, arm in schedule],
            "arms": {
                "baseline": {"workers": workers},
                "treatment": {"workers": treatment_workers},
            },
            "distinct_arms": workers != treatment_workers,
            "descriptive": workers == treatment_workers,
            "prewarmed": True,
            "cold_repetitions": 1,
            "cache_states": ["cold", "warm"],
            "cache_definition": "cold runs use a private cache directory per sample and a fresh run; warm runs reuse one directory per arm",
            "results": {
                cache: {arm: perf_protocol.summarize(values) for arm, values in arms.items()}
                for cache, arms in samples.items()
            },
            "correctness_equivalent": True,
            "selected": {"arm": "treatment", "cache": "warm", "repetition": selected["repetition"]},
        },
        "representative": selected["per_exposure"],
        "total_wall_s": float(selected["elapsed_s"]),
    }


def _write_markdown(report: dict, sweep: dict, cprofile_note: str, *, out_md=None) -> None:
    hdr = report["header"]
    mpix_per_s = report.get("mpix_per_s")
    throughput = f"{mpix_per_s:.3f}" if mpix_per_s is not None else "n/a"
    lines = [
        "# MegaCam per-stage profile",
        "",
        f"arch={hdr.get('arch')} cpu={hdr.get('cpu_count')} platform={hdr.get('platform')} config={hdr.get('config')}",
        f"exposures={hdr.get('n_exposures')} hdus={hdr.get('n_hdus')} "
        f"total_wall={report['total_wall_s']:.1f}s hdu_wall={report['hdu_wall_s']:.1f}s "
        f"mpix={report['mpix_total']:.1f} mpix/s={throughput}",
        f"thread_environment={hdr.get('thread_environment')}",
        "",
    ]
    descriptive = report.get("sequential_descriptive")
    if descriptive:
        lines.append(
            f"sequential_descriptive.speedup_eligible={str(descriptive.get('speedup_eligible', False)).lower()} "
            f"wall_s={descriptive.get('wall_s', report['total_wall_s']):.1f}"
        )
    for warning in report.get("warnings") or []:
        lines += [f"> **WARNING:** {warning}", ""]
    lines += [
        "## Per-stage totals",
        "",
        "| stage | total_s | mean_per_hdu_s | share |",
        "| --- | --- | --- | --- |",
    ]
    for k in STAGE_KEYS + sorted(set(report["stages"]) - set(STAGE_KEYS)):
        s = report["stages"].get(k)
        if not s:
            continue
        lines.append(f"| {k} | {s['total_s']:.2f} | {s['mean_per_hdu_s']:.3f} | {s['share']:.3f} |")
    lines += ["", "## Per-exposure wall", "", "| exposure | hdus | wall_s | mpix |", "| --- | --- | --- | --- |"]
    for e in report["per_exposure"]:
        lines.append(
            f"| {e['safe_id']} | {e['nhdus_processed']}/{e['nhdus_expected']} | {e['wall_s']:.1f} | {e['mpix']:.1f} |"
        )
    lines += ["", "## Scaling sweep (first exposure only)", ""]
    if sweep:
        lines.append("| workers | wall_s |")
        lines.append("| --- | --- |")
        for w in sorted(sweep, key=lambda x: int(x)):
            lines.append(f"| {w} | {sweep[w]:.1f} |")
    else:
        lines.append("n/a")
    if report.get("protocol_reports"):
        lines += ["", "## Interleaved timing protocol", ""]
        for name, protocol in report["protocol_reports"].items():
            warm = protocol["results"]["warm"]
            lines.append(
                f"{name}: baseline median={warm['baseline']['median']:.3f}s "
                f"IQR={warm['baseline']['iqr']:.3f}s MAD={warm['baseline']['mad']:.3f}s; "
                f"treatment median={warm['treatment']['median']:.3f}s "
                f"IQR={warm['treatment']['iqr']:.3f}s MAD={warm['treatment']['mad']:.3f}s"
            )
    lines += [
        "",
        "## Release eligibility",
        "",
        f"release_evidence_eligible={str(report.get('timing_protocol', {}).get('release_evidence_eligible', False)).lower()}",
    ]
    lines += ["", "## cProfile", "", cprofile_note or "n/a", ""]
    (out_md or PERF_MD).write_text("\n".join(lines) + "\n")


def _run_cprofile_first_hdu(
    first_rec: dict,
    *,
    flat: str | None = None,
    out_txt=None,
    config_overrides: dict | None = None,
    hdu_limit: int = 0,
) -> str:
    import fitsio
    import yaml

    from weightmask import cli
    from weightmask.process import process_image as _orig

    in_path = ROOT / first_rec["file"]
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    if config_overrides:
        config = _deep_merge(config, config_overrides)
    hdul = fitsio.FITS(str(in_path), "r")
    hdul_flat = fitsio.FITS(str(flat), "r") if flat else None
    try:
        hdus = cli.get_hdus_to_process(hdul, None)
        if hdu_limit > 0:
            hdus = hdus[:hdu_limit]
        mid = hdus[len(hdus) // 2]
        import numpy as np

        sci = np.ascontiguousarray(hdul[mid].read())
        hdr = hdul[mid].read_header()
        flat_data = np.ascontiguousarray(hdul_flat[mid].read().astype(np.float32)) if hdul_flat is not None else None
    finally:
        hdul.close()
        if hdul_flat is not None:
            hdul_flat.close()
    pr = cProfile.Profile()
    pr.enable()
    _orig(sci, hdr, flat_data, config, 1024)
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)
    out_txt = out_txt or CPROFILE_TXT
    out_txt.write_text(buf.getvalue())
    return (
        f"single HDU (exposure {first_rec['safe_id']} hdu {mid} flat={bool(flat)}) top-40 cumulative -> {out_txt.name}"
    )


_PRODUCT_TOLERANCE_POLICY = {
    "inverse_variance": {"relative_floor": 2e-6, "absolute_floor": 1e-7, "ulp_factor": 8},
    "normalized_weight": {"relative_floor": 2e-6, "absolute_floor": 1e-6, "ulp_factor": 8},
    "weight": {"relative_floor": 2e-6, "absolute_floor": 1e-6, "ulp_factor": 8},
    "confidence": {"relative_floor": 2e-6, "absolute_floor": 1e-6, "ulp_factor": 8},
    "sky": {"relative_floor": 2e-6, "absolute_floor": 1e-4, "ulp_factor": 8},
}
_HEADER_IGNORED = {"CHECKSUM", "DATASUM", "DATE", "WMGENID", "EXTNAME", "COMMENT", "HISTORY", "EXTEND"}
_COMPRESSION_HEADER_KEYS = {
    "ZIMAGE",
    "ZSIMPLE",
    "ZTENSION",
    "ZBITPIX",
    "ZNAXIS",
    "ZPCOUNT",
    "ZGCOUNT",
    "ZCMPTYPE",
    "ZQUANTIZ",
    "ZDITHER0",
}
_COMPRESSION_HEADER_PREFIXES = ("ZNAXIS", "ZTILE", "ZNAME", "ZVAL")
_STRUCTURAL_HEADER_KEYS = {
    "SIMPLE",
    "XTENSION",
    "BITPIX",
    "NAXIS",
    "PCOUNT",
    "GCOUNT",
    "TFIELDS",
}


def _numeric_tolerances(kind: str, dtype) -> tuple[float, float]:
    """Return dtype-aware limits for reproducible floating-point products.

    Floors cover the documented product quantisation; the ULP term covers the
    rounding incurred when a product is written at the compared dtype.
    """
    import numpy as np

    policy = _PRODUCT_TOLERANCE_POLICY[kind]
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"numeric tolerance requires a floating dtype, got {dtype}")
    epsilon = np.finfo(dtype).eps
    return (
        max(float(policy["relative_floor"]), float(policy["ulp_factor"]) * epsilon),
        max(float(policy["absolute_floor"]), float(policy["ulp_factor"]) * epsilon),
    )


def _expected_product_names(
    records: list[dict],
    *,
    workers: int,
    flat: str | None,
    write_mask: bool,
    all_products: bool,
    out_suffix: str = "",
    file_tag: str = "",
    include_worker_tag: bool = True,
) -> set[str]:
    names = set()
    for record in records:
        worker_tag = f".w{workers}" if include_worker_tag else ""
        tag = f"{record['safe_id']}{out_suffix}{file_tag}{worker_tag}"
        if flat:
            tag += ".flat"
        suffixes = [".weight.fits"]
        if write_mask or all_products:
            suffixes.append(".mask.fits")
        if all_products:
            suffixes.extend(
                [
                    ".ivar.fits",
                    ".sky.fits",
                    ".weight_raw.fits",
                    ".weight.bad.fits",
                    ".weight.sat.fits",
                    ".weight.cr.fits",
                    ".weight.obj.fits",
                    ".weight.streak.fits",
                    ".weight.nodata.fits",
                ]
            )
        names.update(f"{tag}{suffix}" for suffix in suffixes)
    return names


def _product_files(directory: Path, since: float | None) -> dict[str, Path]:
    files = {}
    for path in sorted(Path(directory).glob("*")):
        if not path.is_file() or not path.name.endswith((".fits", ".fits.fz")):
            continue
        if since is not None:
            try:
                if path.stat().st_mtime < since - 1.0:
                    continue
            except OSError:
                continue
        files[path.name] = path
    return files


def _header_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return value.strip() if isinstance(value, str) else value


def _hdu_identity(header, index: int, *, ignore_extname: bool = False):
    extname = _header_value(header.get("EXTNAME"))
    ccd = None
    for key in ("CCDNAME", "CCDNAM", "CCDID", "CHIPID"):
        value = _header_value(header.get(key))
        if value not in (None, ""):
            ccd = str(value).upper()
            break
    if ignore_extname:
        extname = None
    if extname not in (None, ""):
        return ("EXTNAME", str(extname).upper())
    if ccd not in (None, ""):
        return ("CCD", ccd)
    return None


def _header_metadata(header):
    result = {}
    for key in header.keys():
        key = str(key).upper()
        if (
            key in _HEADER_IGNORED
            or key in _STRUCTURAL_HEADER_KEYS
            or key in _COMPRESSION_HEADER_KEYS
            or key.startswith("NAXIS")
            or key.startswith(_COMPRESSION_HEADER_PREFIXES)
        ):
            continue
        if key.startswith(("TTYPE", "TUNIT", "TFORM")):
            continue
        result[key] = _header_value(header[key])
    return result


def _product_kind(path: Path, header) -> str:
    filename = path.name.lower()
    artifact = str(_header_value(header.get("WMART")) or "").lower()
    if artifact == "quality_mask" or ".mask." in filename or any(
        token in filename
        for token in (
            ".bad.",
            ".sat.",
            ".cr.",
            ".obj.",
            ".streak.",
            ".nodata.",
            "_bad_",
            "_sat_",
            "_cr_",
            "_obj_",
            "_streak_",
            "_nodata_",
        )
    ):
        return "mask"
    if artifact == "inverse_variance" or ".ivar." in filename:
        return "inverse_variance"
    if artifact == "confidence" or "confidence" in filename:
        return "confidence"
    if artifact == "sky" or artifact == "sky_mesh" or ".sky." in filename:
        return "sky"
    if artifact == "weight" and ".weight_raw." in filename:
        return "weight"
    return "normalized_weight"


def _semantic_problem(path: Path, header, kind: str) -> str | None:
    artifact = _header_value(header.get("WMART"))
    semantics = _header_value(header.get("WMSEM"))
    if artifact is None or semantics is None:
        return f"{path.name} is missing WMART/WMSEM contract metadata for {kind}"
    artifact = str(artifact or "").lower()
    semantics = str(semantics or "")
    allowed = {
        "mask": {
            ("quality_mask", "named_quality_bits"),
            ("bad_mask", "boolean_mask"),
            ("saturation_mask", "boolean_mask"),
            ("cosmic_ray_mask", "boolean_mask"),
            ("object_mask", "boolean_mask"),
            ("streak_mask", "boolean_mask"),
            ("nodata_mask", "boolean_mask"),
        },
        "inverse_variance": {("inverse_variance", "inverse_variance_adu^-2")},
        "normalized_weight": {("weight", "masked_inverse_variance"), ("weight", "inverse_variance_adu^-2")},
        "weight": {("weight", "masked_inverse_variance"), ("weight", "inverse_variance_adu^-2")},
        "confidence": {
            ("confidence", "normalized_weight_0_to_1"),
            ("confidence", "normalized_weight_0_to_100"),
        },
        "sky": {("sky", "background_adu"), ("sky_mesh", "background_adu")},
    }
    if (artifact, semantics) not in allowed[kind]:
        return f"{path.name} declares incompatible {artifact}/{semantics} metadata for {kind}"
    return None


def _read_product_records(handle, *, ignore_extname: bool):
    records = []
    primary_metadata = {}
    for index, hdu in enumerate(handle):
        header = hdu.read_header()
        info = hdu.get_info()
        dims = tuple(info.get("dims") or ())
        if index == 0 and not dims and header.get("XTENSION") is None:
            primary_metadata = _header_metadata(header)
            continue
        identity = _hdu_identity(header, index, ignore_extname=ignore_extname)
        if identity is None:
            raise ValueError(f"HDU {index} has no authoritative EXTNAME or CCD identity")
        records.append(
            {
                "index": index,
                "header": header,
                "dims": dims,
                "identity": identity,
            }
        )
    if not records:
        raise ValueError("product file contains no nonempty authoritative HDUs")
    return records, primary_metadata


def _identity_map(records, side: str):
    result = {}
    for record in records:
        identity = record["identity"]
        if identity in result:
            raise ValueError(f"duplicate HDU identity {identity} in {side}")
        result[identity] = record
    return result


def _compare_product_file(path: Path, base: Path, *, ignore_extname: bool) -> list[str]:
    import fitsio
    import numpy as np

    with fitsio.FITS(str(path)) as f_new, fitsio.FITS(str(base)) as f_base:
        new_records, new_primary_metadata = _read_product_records(f_new, ignore_extname=ignore_extname)
        base_records, base_primary_metadata = _read_product_records(f_base, ignore_extname=ignore_extname)
        problems = []
        if new_primary_metadata != base_primary_metadata:
            changed = sorted(set(new_primary_metadata) | set(base_primary_metadata))
            changed = [key for key in changed if new_primary_metadata.get(key) != base_primary_metadata.get(key)]
            problems.append(f"primary metadata differs: {', '.join(changed)}")
        if len(new_records) != len(base_records):
            problems.append(f"hdu count {len(new_records)} != {len(base_records)}")
        new_order = [record["identity"] for record in new_records]
        base_order = [record["identity"] for record in base_records]
        if new_order != base_order:
            problems.append(f"HDU ordering differs: {new_order} vs {base_order}")
        new_by_identity = _identity_map(new_records, "current")
        base_by_identity = _identity_map(base_records, "baseline")
        for identity in sorted(set(new_by_identity) | set(base_by_identity), key=str):
            new = new_by_identity.get(identity)
            old = base_by_identity.get(identity)
            if new is None or old is None:
                problems.append(f"HDU identity missing: {identity}")
                continue
            if new["dims"] != old["dims"]:
                problems.append(f"HDU {identity} shape {new['dims']} vs {old['dims']}")
                continue
            kind = _product_kind(path, new["header"])
            semantic_problem = _semantic_problem(path, new["header"], kind)
            if semantic_problem:
                problems.append(semantic_problem)
            baseline_semantic_problem = _semantic_problem(base, old["header"], _product_kind(base, old["header"]))
            if baseline_semantic_problem:
                problems.append(baseline_semantic_problem)
            metadata_new = _header_metadata(new["header"])
            metadata_old = _header_metadata(old["header"])
            if metadata_new != metadata_old:
                changed = sorted(set(metadata_new) | set(metadata_old))
                changed = [key for key in changed if metadata_new.get(key) != metadata_old.get(key)]
                problems.append(f"HDU {identity} metadata differs: {', '.join(changed)}")
            if not new["dims"]:
                continue
            data_new = f_new[new["index"]].read()
            data_old = f_base[old["index"]].read()
            if kind == "mask":
                if data_new.dtype != data_old.dtype:
                    problems.append(f"HDU {identity} dtype {data_new.dtype} vs {data_old.dtype}")
                    continue
                equal = np.array_equal(data_new, data_old)
            else:
                if not np.issubdtype(data_new.dtype, np.floating) or not np.issubdtype(data_old.dtype, np.floating):
                    problems.append(f"HDU {identity} requires floating dtype, got {data_new.dtype} vs {data_old.dtype}")
                    continue
                if data_new.dtype != data_old.dtype:
                    problems.append(f"HDU {identity} dtype {data_new.dtype} vs {data_old.dtype}")
                    continue
                rtol, atol = _numeric_tolerances(kind, data_new.dtype)
                equal = np.allclose(data_new, data_old, rtol=rtol, atol=atol, equal_nan=True)
            if not equal:
                if kind == "mask":
                    differing = int(np.count_nonzero(data_new != data_old))
                else:
                    differing = int(
                        np.count_nonzero(~np.isclose(data_new, data_old, rtol=rtol, atol=atol, equal_nan=True))
                    )
                problems.append(f"HDU {identity} {kind} differs at {differing} pixel(s)")
    return problems


def _compare_products(
    this_dir: Path,
    baseline_dir: Path,
    *,
    expected_files: list[str] | set[str] | None = None,
    since: float | None = None,
    ignore_extname: bool = False,
) -> dict:
    current = _product_files(Path(this_dir), since)
    baseline = _product_files(Path(baseline_dir), None)
    current_names = set(current)
    baseline_names = set(baseline)
    expected = set(expected_files) if expected_files is not None else current_names | baseline_names
    missing_in_current = expected - current_names
    missing_in_baseline = expected - baseline_names
    unexpected_in_this = current_names - expected
    unexpected_in_baseline = baseline_names - expected
    report: dict = {
        "expected_file_count": len(expected),
        "file_count": len(current_names & expected),
        "identical": 0,
        "different": 0,
        "missing": sorted(missing_in_baseline),
        "missing_in_baseline": sorted(missing_in_baseline),
        "missing_in_this": sorted(missing_in_current),
        "unexpected_in_this": sorted(unexpected_in_this),
        "unexpected_in_baseline": sorted(unexpected_in_baseline),
        "details": [],
    }
    for name in sorted(expected & current_names & baseline_names):
        try:
            problems = _compare_product_file(current[name], baseline[name], ignore_extname=ignore_extname)
        except Exception as exc:
            problems = [f"read failed: {exc}"]
        if problems:
            report["different"] += 1
            report["details"].append({"file": name, "problems": problems})
        else:
            report["identical"] += 1
    report["ok"] = not (
        report["different"]
        or report["missing_in_baseline"]
        or report["missing_in_this"]
        or report["unexpected_in_this"]
        or report["unexpected_in_baseline"]
        or not report["expected_file_count"]
    )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MegaCam 10-exposure perf harness.")
    ap.add_argument("--resolve-only", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1, help="protocol repetitions; use at least 5 for release timing")
    ap.add_argument("--seed", type=int, default=0, help="seed for the deterministic A/B protocol schedule")
    ap.add_argument("--force-resolve", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Redirect run products and reports here (default test_outputs/perf).",
    )
    ap.add_argument(
        "--hdu-limit",
        type=int,
        default=0,
        help="Process at most this many HDUs per exposure (0 = all); for quick runs.",
    )
    ap.add_argument(
        "--no-cprofile",
        action="store_true",
        help="Skip the single-HDU cProfile pass (it is one full HDU, often the longest step of a short run).",
    )
    ap.add_argument(
        "--compare-baseline",
        type=str,
        default="",
        help="Directory of a previous run; compare this run's products HDU-by-HDU against it.",
    )
    ap.add_argument(
        "--compare-ignore-extname",
        action="store_true",
        help="With --compare-baseline, ignore EXTNAME differences (use after an intended renaming).",
    )
    ap.add_argument("--flat", type=str, default=None, help="Flat-field MEF for the with-flats variant.")
    ap.add_argument("--tag", type=str, default="", help="Report/output tag (e.g. 'flat'); default untagged.")
    ap.add_argument("--masks", action="store_true", help="Also write integer mask maps.")
    ap.add_argument(
        "--all-products",
        action="store_true",
        help="Write every product (map, mask, ivar, sky, raw weight, per-contaminant masks).",
    )
    ap.add_argument("--resume", action="store_true", help="Skip exposures already in checkpoint_<tag>.json.")
    ap.add_argument(
        "--exposure-ids",
        type=str,
        default="",
        help="Comma-separated safe_id allowlist applied after resolve; empty means all resolved.",
    )
    ap.add_argument(
        "--exposure-file",
        action="append",
        default=[],
        help="Profile this local MegaCam FITS file; repeatable, bypasses resolved exposure cache.",
    )
    ap.add_argument(
        "--config-set",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="Repeatable config override (values YAML-parsed, deep-merged over weightmask.yml).",
    )
    args = ap.parse_args(argv)
    thread_env = perf_protocol.thread_environment(os.environ)
    os.environ.update(thread_env)
    if 1 < args.repeats < perf_protocol.MIN_REPEATS:
        ap.error(f"--repeats must be 1 or at least {perf_protocol.MIN_REPEATS}")
    if args.workers is not None and args.workers < 1:
        ap.error("--workers must be positive")
    if args.hdu_limit < 0:
        ap.error("--hdu-limit must be nonnegative")
    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        if not OUT_DIR.is_absolute():
            OUT_DIR = ROOT / OUT_DIR
        print(f"Redirecting run outputs to {OUT_DIR}")
    compare_dir = Path(args.compare_baseline) if args.compare_baseline else None
    if compare_dir is not None and not compare_dir.is_dir():
        raise SystemExit(f"--compare-baseline directory not found: {compare_dir}")
    if compare_dir is not None and compare_dir.resolve() == OUT_DIR.resolve():
        raise SystemExit("--compare-baseline and --out-dir cannot be the same directory")
    if args.exposure_file:
        from weightmask.utils import paths_alias

        if args.force_resolve:
            ap.error("--exposure-file and --force-resolve cannot be combined")
        records = []
        for value in args.exposure_file:
            path = Path(value).resolve()
            if any(paths_alias(path, rec["file"]) for rec in records):
                ap.error(f"duplicate --exposure-file: {path}")
            valid, reason = _validate_megacam(path)
            if not valid:
                ap.error(f"--exposure-file validation failed for {path}: {reason}")
            records.append(
                {
                    "publisherID": None,
                    "safe_id": path.name.removesuffix(".fz").removesuffix(".fits"),
                    "file": str(path),
                    "exptime": _exptime_of_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    else:
        records = resolve_exposures(force=args.force_resolve)
    config_overrides = _parse_config_sets(args.config_set)
    if config_overrides:
        print(f"Config overrides: {json.dumps(config_overrides, sort_keys=True)}")
    if args.exposure_ids.strip():
        allow = {s.strip() for s in args.exposure_ids.split(",") if s.strip()}
        before = [rec["safe_id"] for rec in records]
        records = [rec for rec in records if rec["safe_id"] in allow]
        missing = sorted(allow - set(before))
        if missing:
            raise SystemExit(f"--exposure-ids has no resolved record for: {', '.join(missing)}")
        if not records:
            raise SystemExit("--exposure-ids filtered out every exposure; aborting.")
        print(f"Exposure allowlist: {sorted(allow)} -> {len(records)} exposure(s).")
    if args.resolve_only:
        if not records:
            print("ERROR: no exposures resolved.")
            return 1
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    perf_json = OUT_DIR / f"megacam_perf{suffix}.json"
    perf_md = OUT_DIR / f"megacam_perf{suffix}.md"
    cprofile_txt = OUT_DIR / f"cprofile_hdu{suffix}.txt"

    _flat_arg = Path(args.flat) if args.flat else None
    flat = None if _flat_arg is None else str(_flat_arg if _flat_arg.is_absolute() else (ROOT / _flat_arg))
    if flat:
        if not Path(flat).exists():
            raise SystemExit(f"--flat file not found: {flat}")
        from tests.benchmarks.download_data import validate_case_file

        valid, reason = validate_case_file({"expected_instrument": "MegaPrime", "expected_detector": "MegaCam"}, flat)
        if not valid:
            raise SystemExit(f"--flat validation failed: {reason}")
        (OUT_DIR / "flats.json").write_text(json.dumps({"file": os.path.relpath(flat, ROOT)}, indent=2) + "\n")
    # (a) sequential pass over all resolved exposures x all HDUs.
    variant = f"with-flats ({flat})" if flat else "no-flat"
    print(f"Running sequential {variant} pass (workers=1) over {len(records)} exposures x all HDUs...")
    checkpoint = OUT_DIR / f"checkpoint{suffix}.json"
    done: dict[str, dict] = {}
    if args.resume and checkpoint.exists():
        try:
            for e in json.loads(checkpoint.read_text()):
                if isinstance(e, dict) and e.get("safe_id"):
                    done[e["safe_id"]] = e
            print(f"Resuming: {len(done)} exposures already checkpointed.")
        except Exception as exc:
            print(f"WARNING: ignoring unreadable checkpoint {checkpoint}: {exc}")
    per_exp: list[dict] = []
    run_started = time.time()
    t_all = time.perf_counter()
    for rec in records:
        if rec["safe_id"] in done:
            print(f"--- exposure {rec['safe_id']} (checkpointed, skipping) ---")
            cached = dict(done[rec["safe_id"]])
            # Tag it: a resumed record contributes pixels and stage times to the
            # report, but none of this process's wall clock.
            cached["resumed"] = True
            per_exp.append(cached)
            continue
        print(f"--- exposure {rec['safe_id']} ---")
        per_exp.append(
            _process_one_exposure(
                rec,
                1,
                flat=flat,
                write_mask=args.masks,
                file_tag=(f".{args.tag}" if args.tag else ""),
                config_overrides=config_overrides,
                hdu_limit=args.hdu_limit,
                all_products=args.all_products,
            )
        )
        checkpoint.write_text(json.dumps(per_exp, indent=2) + "\n")
    total_wall = time.perf_counter() - t_all

    # Sanity: every exposure processed all its 2-D HDUs.
    for e in per_exp:
        if e["nhdus_processed"] != e["nhdus_expected"]:
            print(f"WARNING: {e['safe_id']}: processed {e['nhdus_processed']}/{e['nhdus_expected']}")

    # (b) scaling sweep on first exposure only (untagged runs; tagged variants reuse it).
    from weightmask.mef import _resolve_max_workers

    sweep: dict[str, float] = {}
    first = records[0]
    n_first = per_exp[0]["nhdus_expected"]
    if not args.tag:
        for w in (1, 2, 4, 8):
            eff = _resolve_max_workers(w, n_first)
            print(f"--- sweep workers={w} (eff={eff}) on {first['safe_id']} ---")
            if w == 1:
                sweep["1"] = float(per_exp[0]["wall_s"])
                continue
            r = _process_one_exposure(
                first,
                eff,
                out_suffix=f".sweep{w}",
                flat=flat,
                write_mask=args.masks,
                config_overrides=config_overrides,
                hdu_limit=args.hdu_limit,
                all_products=args.all_products,
            )
            sweep[str(w)] = float(r["wall_s"])
    else:
        sweep["1"] = float(per_exp[0]["wall_s"])

    # (c) single-HDU cProfile.
    if args.no_cprofile:
        note = "skipped (--no-cprofile)"
    else:
        note = _run_cprofile_first_hdu(
            first, flat=flat, out_txt=cprofile_txt, config_overrides=config_overrides, hdu_limit=args.hdu_limit
        )

    exit_code = int(any(e["nhdus_processed"] != e["nhdus_expected"] for e in per_exp))
    if args.workers is not None and int(args.workers) != 1:
        print(f"Running extra full pass at workers={args.workers}...")
        t1 = time.perf_counter()
        wpass: list[dict] = []
        for rec in records:
            wpass.append(
                _process_one_exposure(
                    rec,
                    int(args.workers),
                    out_suffix=f".w{args.workers}",
                    flat=flat,
                    write_mask=args.masks,
                    file_tag=(f".{args.tag}" if args.tag else ""),
                    config_overrides=config_overrides,
                    hdu_limit=args.hdu_limit,
                    all_products=args.all_products,
                )
            )
        wwall = time.perf_counter() - t1
        if any(e["nhdus_processed"] != e["nhdus_expected"] for e in wpass):
            exit_code = 1
        print(f"Extra pass wall: {wwall:.1f}s")
        wstages: dict[str, float] = {}
        for e in wpass:
            for k, v in (e.get("stage_totals") or {}).items():
                wstages[k] = wstages.get(k, 0.0) + float(v)
        wnhdus = sum(int(e.get("nhdus_processed", 0)) for e in wpass)
        sidecar = {
            "tag": args.tag or "baseline",
            "cpu_basis": "delta-v3-timed-region",
            "rss_basis": "process-lifetime-high-water",
            "extra_pass_wall_s": float(wwall),
            "mpix_total": float(sum(float(e.get("mpix", 0.0)) for e in wpass)),
            "nhdus": int(wnhdus),
            "peak_rss_kb": float(max([float(e.get("peak_rss_kb", 0.0)) for e in wpass] or [0.0])),
            "cpu_s": float(sum(float(e.get("cpu_s", 0.0)) for e in wpass)),
            "stages": {
                k: {
                    "total_s": float(v),
                    "mean_per_hdu_s": float(v / wnhdus) if wnhdus else 0.0,
                    "share": float(v / max(wstages.get("hdu_total", sum(wstages.values())), 1e-9)),
                }
                for k, v in wstages.items()
            },
        }
        sidecar_path = OUT_DIR / f"megacam_pass{suffix}.w{args.workers}.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"Wrote {sidecar_path}")

    protocol_reports: dict[str, dict] = {}
    if args.repeats >= perf_protocol.MIN_REPEATS:
        if not args.tag:
            for w in (2, 4, 8):
                eff = _resolve_max_workers(w, n_first)
                protocol = _run_protocol(
                    [first],
                    1,
                    treatment_workers=eff,
                    repeats=args.repeats,
                    seed=args.seed + w,
                    flat=flat,
                    write_mask=args.masks,
                    config_overrides=config_overrides,
                    hdu_limit=args.hdu_limit,
                    all_products=args.all_products,
                )
                protocol_reports[f"workers_{w}"] = protocol
                sweep[str(w)] = protocol["protocol"]["results"]["warm"]["treatment"]["median"]

    extra = {
        "variant": ("with-flats" if flat else "no-flat"),
        "tag": args.tag or "baseline",
        "config_overrides": config_overrides,
    }
    extra.update(_provenance(records, config_overrides, flat, thread_env))
    if flat:
        extra["flat_file"] = str(Path(flat).relative_to(ROOT)) if str(flat).startswith(str(ROOT)) else str(flat)
    report = _aggregate_report(per_exp, total_wall, extra_header=extra)
    report["sequential_descriptive"] = {
        "wall_s": float(total_wall),
        "workers": 1,
        "speedup_eligible": False,
        "reason": "single treatment measurement; no baseline/treatment protocol",
    }
    report["sweep_workers"] = {k: float(v) for k, v in sweep.items()}
    report["timing_protocol"] = {
        "repeats": args.repeats,
        "seed": args.seed,
        "correctness_equivalent": bool(protocol_reports) and all(
            protocol["protocol"]["correctness_equivalent"] for protocol in protocol_reports.values()
        ),
        "distinct_arms": bool(protocol_reports) and any(
            protocol["protocol"]["distinct_arms"] for protocol in protocol_reports.values()
        ),
        "full_products": bool(args.all_products),
        "release_evidence_eligible": False,
        "reason": "at least five interleaved repetitions and product equivalence are required",
    }
    report["timing_protocol"]["eligibility"] = perf_protocol.protocols_release_eligibility(
        [protocol["protocol"] for protocol in protocol_reports.values()],
        args.repeats,
        False,
        full_products=report["timing_protocol"]["full_products"],
    )
    report["protocol_reports"] = {}
    for name, protocol in protocol_reports.items():
        protocol["protocol"]["eligibility"] = perf_protocol.release_eligibility(
            args.repeats,
            protocol["protocol"]["correctness_equivalent"],
            False,
            full_products=args.all_products,
            distinct_arms=protocol["protocol"]["distinct_arms"],
        )
        report["protocol_reports"][name] = protocol["protocol"]
    if compare_dir is not None:
        print(f"--- comparing products against baseline {compare_dir} ---")
        expected_files = _expected_product_names(
            records,
            workers=1,
            flat=flat,
            write_mask=args.masks,
            all_products=args.all_products,
            file_tag=(f".{args.tag}" if args.tag else ""),
        )
        if not args.tag:
            for w in (2, 4, 8):
                eff = _resolve_max_workers(w, n_first)
                expected_files.update(
                    _expected_product_names(
                        [first],
                        workers=eff,
                        flat=flat,
                        write_mask=args.masks,
                        all_products=args.all_products,
                        out_suffix=f".sweep{w}",
                    )
                )
        if args.workers is not None and int(args.workers) != 1:
            expected_files.update(
                _expected_product_names(
                    records,
                    workers=int(args.workers),
                    flat=flat,
                    write_mask=args.masks,
                    all_products=args.all_products,
                    out_suffix=f".w{args.workers}",
                    file_tag=(f".{args.tag}" if args.tag else ""),
                )
            )
        comparison = _compare_products(
            OUT_DIR,
            compare_dir,
            expected_files=expected_files,
            since=run_started,
            ignore_extname=args.compare_ignore_extname,
        )
        report["baseline_comparison"] = comparison
        report["timing_protocol"] = {
            "repeats": args.repeats,
            "seed": args.seed,
            "correctness_equivalent": bool(report["timing_protocol"]["correctness_equivalent"]),
            "distinct_arms": bool(report["timing_protocol"]["distinct_arms"]),
            "full_products": bool(args.all_products),
            "release_evidence_eligible": False,
            "reason": "product comparison completed",
        }
        report["timing_protocol"]["eligibility"] = perf_protocol.protocols_release_eligibility(
            [protocol for protocol in report["protocol_reports"].values()],
            args.repeats,
            bool(comparison["ok"]),
            full_products=args.all_products,
        )
        report["timing_protocol"]["release_evidence_eligible"] = report["timing_protocol"]["eligibility"]["eligible"]
        for protocol in report["protocol_reports"].values():
            protocol["eligibility"] = perf_protocol.release_eligibility(
                args.repeats,
                protocol["correctness_equivalent"],
                bool(comparison["ok"]),
                full_products=args.all_products,
                distinct_arms=protocol["distinct_arms"],
            )
        for detail in comparison["details"][:10]:
            print(f"  {detail['file']}: {'; '.join(detail['problems'][:3])}")
        if comparison["missing"]:
            print(f"  missing in baseline: {comparison['missing'][:5]}")
        if comparison["missing_in_this"]:
            print(f"  missing in current run: {comparison['missing_in_this'][:5]}")
        if comparison["unexpected_in_this"]:
            print(f"  unexpected in current run: {comparison['unexpected_in_this'][:5]}")
        if comparison["unexpected_in_baseline"]:
            print(f"  unexpected in baseline: {comparison['unexpected_in_baseline'][:5]}")
        if not comparison["ok"]:
            print(f"BASELINE MISMATCH: {comparison['different']} differing / {comparison['file_count']} product files")
            exit_code = 1
        else:
            print(f"Baseline OK: {comparison['identical']}/{comparison['file_count']} product files identical.")
    perf_json.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, sweep, note, out_md=perf_md)
    print(f"Wrote {perf_json} and {perf_md}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

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
  pixi run python benchmarks/perf_megacam.py --all-hdus --workers 1
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PERF_DATA_DIR = ROOT / "benchmark_data" / "megacam" / "perf"
OUT_DIR = ROOT / "test_outputs" / "perf"
EXPOSURES_JSON = OUT_DIR / "exposures.json"
PERF_JSON = OUT_DIR / "megacam_perf.json"
PERF_MD = OUT_DIR / "megacam_perf.md"
CPROFILE_TXT = OUT_DIR / "cprofile_hdu.txt"
CONFIG_PATH = ROOT / "weightmask.yml"
FLAT_PID = "ivo://cadc.nrc.ca/CFHT?08Bm01.flat.r.36.02/08Bm01.flat.r.36.02"
FLAT_FILE = PERF_DATA_DIR / "flat_08Bm01_r.fits.fz"
FLATS_JSON = OUT_DIR / "flats.json"

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
# RUSAGE_SELF counters are process-cumulative; successive _process_one_exposure
# calls in one process must report deltas, not running totals.
_LAST_CPU_S: float | None = None


def _safe_id(publisher_id: str) -> str:
    """Map a publisherID (IVO URI or plain CADC id) to a filename-safe id."""
    s = str(publisher_id).strip()
    if "/" in s:
        s = s.split("/")[-1]
    if "?" in s:
        s = s.split("?")[-1]
    return s


def _quarantine_invalid(path: Path) -> None:
    invalid = Path(str(path) + ".invalid")
    try:
        if invalid.exists():
            invalid.unlink()
        shutil.move(str(path), str(invalid))
        print(f"  Moved invalid file to {invalid}")
    except Exception as e:  # pragma: no cover - filesystem race
        print(f"  Warning: failed to quarantine invalid file {path}: {e}")


def _download_http(url: str, out_path: Path) -> bool:
    if out_path.exists():
        print(f"  Already exists: {out_path}")
        return True
    print(f"  Downloading from {url} ...")
    try:
        import ssl
        from urllib.request import urlopen

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ctx = ssl._create_unverified_context()
        with urlopen(url, context=ctx) as resp, open(out_path, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 1024)
        print(f"  Saved to {out_path}")
        return True
    except Exception as e:
        print(f"  HTTP download failed: {e}")
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass
        return False


def _validate_megacam(path: Path) -> tuple[bool, str | None]:
    """Reuse the manifest INSTRUME/DETECTOR check via validate_case_file."""
    try:
        from tests.benchmarks.download_data import validate_case_file

        case = {"expected_instrument": "MegaPrime", "expected_detector": "MegaCam"}
        return validate_case_file(case, str(path))
    except ImportError:
        pass
    except Exception as e:
        return False, str(e)
    try:
        import fitsio

        found_inst = found_det = False
        with fitsio.FITS(str(path)) as hdul:
            for hdu in hdul:
                try:
                    hdr = hdu.read_header()
                except Exception:
                    continue
                if "INSTRUME" in hdr and "megaprime" in str(hdr["INSTRUME"]).lower():
                    found_inst = True
                if "DETECTOR" in hdr and "megacam" in str(hdr["DETECTOR"]).lower():
                    found_det = True
        if found_inst and found_det:
            return True, None
        return False, f"inline check failed inst={found_inst} det={found_det}"
    except Exception as e:
        return False, str(e)


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
    flat: str | None = None,
    write_mask: bool = False,
    file_tag: str = "",
    config_overrides: dict | None = None,
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

    tag = f"{safe}{out_suffix}{file_tag}.w{workers}"
    if flat:
        tag += ".flat"
    args = _ap.Namespace(
        output_map=str(OUT_DIR / f"{tag}.weight.fits"),
        output_mask=str(OUT_DIR / f"{tag}.mask.fits") if write_mask else None,
        output_invvar=None,
        output_sky=None,
        output_weight_raw=None,
        individual_masks=False,
        max_workers=workers,
        tile_size=1024,
    )
    from weightmask.cli import determine_output_paths

    paths = determine_output_paths(args, str(in_path), config)
    for key in ("out_invvar_path", "out_sky_path", "out_weight_raw_path"):
        paths[key] = None
    if not write_mask:
        paths["out_mask_path"] = None
    paths["individual_mask_paths"] = {}

    hdul_flat = fitsio.FITS(str(flat), "r") if flat else None
    records, orig, proc_mod, mef_mod = _install_timing_collector()
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
            global _LAST_CPU_S
            _cpu_s = _cpu_now - _LAST_CPU_S if _LAST_CPU_S is not None else _cpu_now
            _LAST_CPU_S = _cpu_now
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
    header = {
        "arch": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": _os.cpu_count(),
        "config": "weightmask.yml (unmodified)",
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
        "mpix_per_s": float(mpix_total / total_wall) if total_wall > 0 else 0.0,
        "stages": stages,
        "per_exposure": per_exp,
        "cpu_basis": "delta-v2",
    }


def _write_markdown(report: dict, sweep: dict, cprofile_note: str, *, out_md=None) -> None:
    hdr = report["header"]
    lines = [
        "# MegaCam per-stage profile",
        "",
        f"arch={hdr.get('arch')} cpu={hdr.get('cpu_count')} platform={hdr.get('platform')} config={hdr.get('config')}",
        f"exposures={hdr.get('n_exposures')} hdus={hdr.get('n_hdus')} "
        f"total_wall={report['total_wall_s']:.1f}s hdu_wall={report['hdu_wall_s']:.1f}s "
        f"mpix={report['mpix_total']:.1f} mpix/s={report['mpix_per_s']:.3f}",
        "",
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
    lines += ["", "## cProfile", "", cprofile_note or "n/a", ""]
    (out_md or PERF_MD).write_text("\n".join(lines) + "\n")


def _run_cprofile_first_hdu(
    first_rec: dict, *, flat: str | None = None, out_txt=None, config_overrides: dict | None = None
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
        mid = hdus[len(hdus) // 2]
        import numpy as np

        sci = np.ascontiguousarray(hdul[mid].read().astype(np.float32))
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MegaCam 10-exposure perf harness.")
    ap.add_argument("--resolve-only", action="store_true")
    ap.add_argument("--all-hdus", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force-resolve", action="store_true")
    ap.add_argument("--flat", type=str, default=None, help="Flat-field MEF for the with-flats variant.")
    ap.add_argument("--tag", type=str, default="", help="Report/output tag (e.g. 'flat'); default untagged.")
    ap.add_argument("--masks", action="store_true", help="Also write integer mask maps.")
    ap.add_argument("--resume", action="store_true", help="Skip exposures already in checkpoint_<tag>.json.")
    ap.add_argument(
        "--exposure-ids",
        type=str,
        default="",
        help="Comma-separated safe_id allowlist applied after resolve; empty means all resolved.",
    )
    ap.add_argument(
        "--config-set",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="Repeatable config override (values YAML-parsed, deep-merged over weightmask.yml).",
    )
    args = ap.parse_args(argv)
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
        FLATS_JSON.write_text(
            json.dumps({"flat_publisherID": FLAT_PID, "file": str(Path(flat).relative_to(ROOT))}, indent=2) + "\n"
        )
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
    t_all = time.perf_counter()
    for rec in records:
        if rec["safe_id"] in done:
            print(f"--- exposure {rec['safe_id']} (checkpointed, skipping) ---")
            per_exp.append(done[rec["safe_id"]])
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
            r = _process_one_exposure(first, eff, out_suffix=f".sweep{w}", config_overrides=config_overrides)
            sweep[str(w)] = float(r["wall_s"])
    else:
        sweep["1"] = float(per_exp[0]["wall_s"])

    # (c) single-HDU cProfile.
    note = _run_cprofile_first_hdu(first, flat=flat, out_txt=cprofile_txt, config_overrides=config_overrides)

    extra = {"variant": ("with-flats" if flat else "no-flat"), "tag": args.tag or "baseline"}
    if flat:
        extra["flat_publisherID"] = FLAT_PID
        extra["flat_file"] = str(Path(flat).relative_to(ROOT)) if str(flat).startswith(str(ROOT)) else str(flat)
    report = _aggregate_report(per_exp, total_wall, extra_header=extra)
    report["sweep_workers"] = {k: float(v) for k, v in sweep.items()}
    perf_json.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, sweep, note, out_md=perf_md)
    print(f"Wrote {perf_json} and {perf_md}")

    # Optional extra pass at a requested worker count (Step 5 comparison).
    # CANFAR jobs consume this sidecar (wall/peak/CPU/stages of the parallel
    # pass); per-exposure rusage is exact because the backend is threaded.
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
                )
            )
        wwall = time.perf_counter() - t1
        print(f"Extra pass wall: {wwall:.1f}s")
        wstages: dict[str, float] = {}
        for e in wpass:
            for k, v in (e.get("stage_totals") or {}).items():
                wstages[k] = wstages.get(k, 0.0) + float(v)
        wnhdus = sum(int(e.get("nhdus_processed", 0)) for e in wpass)
        sidecar = {
            "tag": args.tag or "baseline",
            "cpu_basis": "delta-v2",
            "extra_pass_wall_s": float(wwall),
            "mpix_total": float(sum(float(e.get("mpix", 0.0)) for e in wpass)),
            "nhdus": int(wnhdus),
            "peak_rss_kb": float(max([float(e.get("peak_rss_kb", 0.0)) for e in wpass] or [0.0])),
            "cpu_s": float(sum(float(e.get("cpu_s", 0.0)) for e in wpass)),
            "stages": {
                k: {
                    "total_s": float(v),
                    "mean_per_hdu_s": float(v / wnhdus) if wnhdus else 0.0,
                    "share": float(v / max(sum(wstages.values()), 1e-9)),
                }
                for k, v in wstages.items()
            },
        }
        sidecar_path = OUT_DIR / f"megacam_pass{suffix}.w{args.workers}.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
        print(f"Wrote {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

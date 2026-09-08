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

BASE_WHERE = (
    "Observation.collection = 'CFHT' AND "
    "Observation.instrument_name = 'MegaPrime' AND "
    "Observation.type = 'OBJECT'"
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
    """Pin 10 long exposures; write and return the exposures.json records."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PERF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EXPOSURES_JSON.exists() and not force:
        try:
            records = json.loads(EXPOSURES_JSON.read_text())
            if isinstance(records, list) and len(records) == 10:
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


def _process_one_exposure(rec, workers: int, out_suffix: str = "") -> dict:
    import argparse as _ap

    import fitsio
    import yaml

    from weightmask import cli, mef

    in_path = ROOT / rec["file"]
    safe = rec["safe_id"]
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
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

    tag = f"{safe}{out_suffix}.w{workers}"
    args = _ap.Namespace(
        output_map=str(OUT_DIR / f"{tag}.weight.fits"),
        output_mask=None,
        output_invvar=None,
        output_sky=None,
        output_weight_raw=None,
        individual_masks=False,
        max_workers=workers,
        tile_size=1024,
    )
    from weightmask.cli import determine_output_paths

    paths = determine_output_paths(args, str(in_path), config)
    for key in ("out_mask_path", "out_invvar_path", "out_sky_path", "out_weight_raw_path"):
        paths[key] = None
    paths["individual_mask_paths"] = {}

    records, orig, proc_mod, mef_mod = _install_timing_collector()
    t0 = time.perf_counter()
    try:
        n_ok = mef.process_all_hdus(
            hdus,
            hdul_input,
            None,
            config,
            paths,
            args,
            flat_path=None,
            max_workers=workers,
            input_path=str(in_path),
        )
    finally:
        wall = time.perf_counter() - t0
        _uninstall_timing_collector(orig, proc_mod, mef_mod)
        try:
            hdul_input.close()
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
        "exptime": rec.get("exptime"),
        "size_bytes": file_bytes,
        "nhdus_expected": nhdus_expected,
        "nhdus_processed": int(n_ok),
        "nhdus_timed": len(records),
        "wall_s": float(wall),
        "mpix": float(mpix),
        "stage_totals": stage_totals,
    }


def _aggregate_report(per_exp: list[dict], total_wall: float) -> dict:
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
    return {
        "header": {
            "arch": platform.machine(),
            "platform": platform.platform(),
            "cpu_count": _os.cpu_count(),
            "config": "weightmask.yml (unmodified)",
            "n_exposures": len(per_exp),
            "n_hdus": n_hdus,
        },
        "total_wall_s": float(total_wall),
        "hdu_wall_s": float(hdu_wall),
        "mpix_total": float(mpix_total),
        "mpix_per_s": float(mpix_total / total_wall) if total_wall > 0 else 0.0,
        "stages": stages,
        "per_exposure": per_exp,
    }


def _write_markdown(report: dict, sweep: dict, cprofile_note: str) -> None:
    hdr = report["header"]
    lines = [
        "# MegaCam per-stage profile",
        "",
        f"arch={hdr.get('arch')} cpu={hdr.get('cpu_count')} "
        f"platform={hdr.get('platform')} config={hdr.get('config')}",
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
        lines.append(f"| {e['safe_id']} | {e['nhdus_processed']}/{e['nhdus_expected']} | {e['wall_s']:.1f} | {e['mpix']:.1f} |")
    lines += ["", "## Scaling sweep (first exposure only)", ""]
    if sweep:
        lines.append("| workers | wall_s |")
        lines.append("| --- | --- |")
        for w in sorted(sweep, key=lambda x: int(x)):
            lines.append(f"| {w} | {sweep[w]:.1f} |")
    else:
        lines.append("n/a")
    lines += ["", "## cProfile", "", cprofile_note, ""]
    PERF_MD.write_text("\n".join(lines) + "\n")


def _run_cprofile_first_hdu(first_rec: dict) -> str:
    import fitsio
    import yaml

    from weightmask import cli
    from weightmask.process import process_image as _orig

    in_path = ROOT / first_rec["file"]
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)
    hdul = fitsio.FITS(str(in_path), "r")
    try:
        hdus = cli.get_hdus_to_process(hdul, None)
        mid = hdus[len(hdus) // 2]
        import numpy as np

        sci = np.ascontiguousarray(hdul[mid].read().astype(np.float32))
        hdr = hdul[mid].read_header()
    finally:
        hdul.close()
    pr = cProfile.Profile()
    pr.enable()
    _orig(sci, hdr, None, config, 1024)
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)
    CPROFILE_TXT.write_text(buf.getvalue())
    return f"single HDU (exposure {first_rec['safe_id']} hdu {mid}) top-40 cumulative -> {CPROFILE_TXT.name}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MegaCam 10-exposure perf harness.")
    ap.add_argument("--resolve-only", action="store_true")
    ap.add_argument("--all-hdus", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force-resolve", action="store_true")
    args = ap.parse_args(argv)

    records = resolve_exposures(force=args.force_resolve)
    if args.resolve_only:
        if len(records) != 10:
            print(f"ERROR: expected 10 exposures, have {len(records)}")
            return 1
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # (a) sequential baseline over all 10 x all HDUs.
    print("Running sequential baseline (workers=1) over 10 exposures x all HDUs...")
    per_exp: list[dict] = []
    t_all = time.perf_counter()
    for rec in records:
        print(f"--- exposure {rec['safe_id']} ---")
        per_exp.append(_process_one_exposure(rec, 1))
    total_wall = time.perf_counter() - t_all

    # Sanity: every exposure processed all its 2-D HDUs.
    for e in per_exp:
        if e["nhdus_processed"] != e["nhdus_expected"]:
            print(f"WARNING: {e['safe_id']}: processed {e['nhdus_processed']}/{e['nhdus_expected']}")

    # (b) scaling sweep on first exposure only.
    from weightmask.mef import _resolve_max_workers

    sweep: dict[str, float] = {}
    first = records[0]
    n_first = per_exp[0]["nhdus_expected"]
    for w in (1, 2, 4, 8):
        eff = _resolve_max_workers(w, n_first)
        print(f"--- sweep workers={w} (eff={eff}) on {first['safe_id']} ---")
        if w == 1:
            sweep["1"] = float(per_exp[0]["wall_s"])
            continue
        r = _process_one_exposure(first, eff, out_suffix=f".sweep{w}")
        sweep[str(w)] = float(r["wall_s"])

    # (c) single-HDU cProfile.
    note = _run_cprofile_first_hdu(first)

    report = _aggregate_report(per_exp, total_wall)
    report["sweep_workers"] = {k: float(v) for k, v in sweep.items()}
    PERF_JSON.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, sweep, note)
    print(f"Wrote {PERF_JSON} and {PERF_MD}")

    # Optional extra pass at a requested worker count (Step 5 comparison).
    if args.workers is not None and int(args.workers) != 1:
        print(f"Running extra full pass at workers={args.workers}...")
        t1 = time.perf_counter()
        for rec in records:
            _process_one_exposure(rec, int(args.workers), out_suffix=f".w{args.workers}")
        print(f"Extra pass wall: {time.perf_counter() - t1:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

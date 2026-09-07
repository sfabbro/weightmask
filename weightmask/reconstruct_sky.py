"""Rebuild full-resolution sky maps from SKYMESH mesh FITS products."""

from __future__ import annotations

import argparse
import os
import sys

import fitsio

from . import __version__
from .background import parse_sky_mesh_header, reconstruct_sky_from_header
from .utils import extract_hdu_spec


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="weightmask-reconstruct-sky",
        description="Rebuild a full-resolution sky map from a WeightMask SKYMESH FITS product.",
        epilog=(
            "Examples:\n"
            "  weightmask-reconstruct-sky sky_mesh.fits -o sky_full.fits\n"
            "  weightmask-reconstruct-sky sky_mesh.fits -o sky_full.fits --hdu 1\n"
            "  weightmask reconstruct-sky sky_mesh.fits -o sky_full.fits\n"
            "\n"
            "See docs/usage.md and docs/algorithms.md. This is a separate program, "
            "not a weightmask flag."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inputs = parser.add_argument_group("Inputs")
    outputs = parser.add_argument_group("Outputs")
    run = parser.add_argument_group("Run")
    inputs.add_argument(
        "input_file",
        type=str,
        help="Sky mesh FITS (SKYMESH / MESHBW / MESHBH / SKYH / SKYW cards required).",
    )
    outputs.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output full-resolution sky FITS.",
    )
    run.add_argument(
        "--hdu",
        type=int,
        default=None,
        help="HDU index. Default: every SKYMESH image HDU.",
    )
    run.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args(argv)


def _image_hdus(hdul, hdu: int | None) -> list[int]:
    if hdu is not None:
        if not (0 <= hdu < len(hdul)):
            print(f"ERROR: Specified HDU {hdu} not found.")
            return []
        info = hdul[hdu].get_info()
        if info.get("hdutype") != 0 or info.get("ndims") != 2:
            print(f"ERROR: HDU {hdu} is not a 2D image.")
            return []
        return [hdu]
    out = []
    for idx, hdu_i in enumerate(hdul):
        try:
            info = hdu_i.get_info()
            if info.get("hdutype") == 0 and info.get("ndims") == 2:
                out.append(idx)
        except Exception:
            continue
    if not out:
        print("ERROR: No suitable Image HDUs found.")
    return out


def reconstruct_sky_fits(input_path: str, output_path: str, hdu: int | None = None) -> int:
    """Rebuild full-res sky FITS from a SKYMESH mesh product. Returns exit code."""
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print("  weightmask-reconstruct-sky <mesh.fits> -o <full.fits>")
        return 1
    try:
        hdul = fitsio.FITS(input_path, "r")
    except OSError as e:
        print(f"ERROR: Could not open {input_path}: {e}")
        return 1

    try:
        candidates = _image_hdus(hdul, hdu)
        if not candidates:
            return 1
        jobs = []
        for i in candidates:
            hdr = hdul[i].read_header()
            try:
                parse_sky_mesh_header(hdr)
            except ValueError as e:
                if hdu is not None:
                    print(f"ERROR: HDU {i}: {e}")
                    return 1
                continue
            full = reconstruct_sky_from_header(hdul[i].read(), hdr)
            try:
                name = hdul[i].get_extname() or f"SKY_{i}"
            except Exception:
                name = f"SKY_{i}"
            out_hdr = {
                k: v
                for k, v in dict(hdr).items()
                if str(k).upper() not in {"SKYMESH", "MESHBW", "MESHBH", "SKYH", "SKYW"}
            }
            jobs.append((full, out_hdr, name))
        if not jobs:
            print("ERROR: No SKYMESH image HDUs found to rebuild.")
            print("  weightmask-reconstruct-sky <mesh.fits> -o <full.fits>")
            return 1

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        if len(jobs) == 1:
            fitsio.write(output_path, jobs[0][0], header=jobs[0][1], clobber=True)
        else:
            fitsio.write(output_path, None, header=None, clobber=True)
            with fitsio.FITS(output_path, "rw") as fout:
                for data, hdr, name in jobs:
                    fout.write(data, header=hdr, extname=name)
        for data, _hdr, name in jobs:
            print(f"  Rebuilt {name}: {data.shape[0]}x{data.shape[1]}")
        print(f"Wrote full sky map: {output_path}")
        return 0
    finally:
        try:
            hdul.close()
        except Exception:
            pass


def main(argv=None) -> int:
    args = parse_args(argv)
    input_path, input_hdu = extract_hdu_spec(args.input_file)
    hdu = args.hdu if args.hdu is not None else input_hdu
    print("Reconstructing full sky from SKYMESH product...")
    return reconstruct_sky_fits(input_path, args.output, hdu=hdu)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Parity between weightmask flat-derived BAD masks and Elixir keep-maps.

Joins flat_map.txt (pid -> flat) with mask_map.txt (pid -> Elixir mask) on
pid, computes our mask per HDU, and reports per-HDU and total agreement:
|ours|, |elixir|, IoU, our extras (ours\\elixir), our misses (elixir\\ours).

Usage:
    pixi run python benchmarks/elixir_parity.py <flat_map> <mask_map> [max_pids]
"""

import sys

import fitsio
import numpy as np

from weightmask.bad import compute_flat_bad_mask


def load_map(path):
    out = {}
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out[parts[0]] = parts[2]
    return out


def main():
    import yaml

    flat_map, mask_map = load_map(sys.argv[1]), load_map(sys.argv[2])
    max_pids = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    cfg = yaml.safe_load(open("weightmask.yml"))["flat_masking"]
    pids = [p for p in flat_map if p in mask_map][:max_pids]
    t_o = t_e = t_i = 0
    for pid in pids:
        with fitsio.FITS(flat_map[pid], "r") as ff:
            idxs = [i for i in range(len(ff)) if ff[i].get_info().get("ndims") == 2]
            flats = {i: ff[i].read().astype(np.float32) for i in idxs}
        with fitsio.FITS(mask_map[pid], "r") as mf:
            for i in idxs:
                if i >= len(mf):
                    continue
                elixir = mf[i].read()
                if elixir.shape != flats[i].shape:
                    continue
                ours = compute_flat_bad_mask(flats[i], cfg)
                el = elixir == 0
                o, e, inter = int(ours.sum()), int(el.sum()), int((ours & el).sum())
                union = o + e - inter
                print(f"{pid} hdu{i}: ours={o} elixir={e} iou={inter/max(1,union):.3f} "
                      f"extras={o-inter} misses={e-inter}")
                t_o += o
                t_e += e
                t_i += inter
    union = t_o + t_e - t_i
    print(f"TOTAL: ours={t_o} elixir={t_e} iou={t_i/max(1,union):.3f}")


if __name__ == "__main__":
    main()

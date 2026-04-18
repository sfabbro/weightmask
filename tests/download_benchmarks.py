import os
from urllib.request import urlretrieve


def download_file(url, out_path):
    if os.path.exists(out_path):
        print(f"Already downloaded: {out_path}")
        return
    print(f"Downloading {url} ...")
    urlretrieve(url, out_path)
    print(f"Saved to {out_path}")


def get_benchmarks():
    base_dir = "benchmark_data"
    os.makedirs(base_dir, exist_ok=True)

    # 1. CFHT MegaCam public science exposure (MEF)
    # 1043132p is a known deep field observation
    megacam_url = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/1043132p.fits.fz"
    download_file(megacam_url, os.path.join(base_dir, "cfht_megacam_1043132p.fits.fz"))

    print("\nBenchmark downloads complete.")


if __name__ == "__main__":
    get_benchmarks()

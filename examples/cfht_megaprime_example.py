#!/usr/bin/env python3
"""
Example: Download and process CFHT MegaPrime image 2079618p from CADC
"""

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from http.client import HTTPException
from pathlib import Path

from astropy.io import fits


def download_cfht_image():
    """Download CFHT MegaPrime image 2079618p from CADC"""

    # CADC archive URL for CFHT MegaPrime
    base_url = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT"
    filename = "2079618p.fits.fz"
    url = f"{base_url}/{filename}"

    print(f"Downloading {filename} from CADC...")

    temporary = None
    try:
        # ponytail: stdlib HTTP errors and atomic replacement preserve an existing image.
        with urllib.request.urlopen(url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=".", prefix=f".{filename}.", delete=False) as download:
                temporary = Path(download.name)
                shutil.copyfileobj(response, download)
        with fits.open(temporary) as hdul:
            hdul.verify("exception")
            size = temporary.stat().st_size
            for hdu in hdul:
                info = hdu.fileinfo()
                if info["datLoc"] + info["datSpan"] > size:
                    raise OSError("Downloaded FITS data are truncated")
        os.replace(temporary, filename)
        print(f"Downloaded {filename}")
        return filename
    except (OSError, HTTPException, ValueError, fits.verify.VerifyError) as e:
        print(f"ERROR: Failed to download {filename}: {e}")
        return None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_weightmask(input_file):
    """Run weightmask on the downloaded image"""

    if not os.path.exists(input_file):
        print(f"ERROR: Input file {input_file} not found")
        return False

    # Find weightmask config
    config_path = Path(__file__).parent.parent / "weightmask.yml"

    # Run weightmask command
    cmd = [
        sys.executable,
        "-m",
        "weightmask.cli",
        input_file,
        "--config",
        str(config_path),
        "--output_mask",
        f"{input_file.replace('.fits.fz', '.mask.fits')}",
        "--output_invvar",
        f"{input_file.replace('.fits.fz', '.ivar.fits')}",
        "--output_sky",
        f"{input_file.replace('.fits.fz', '.sky.fits')}",
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("WeightMask processing completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: WeightMask processing failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main function"""
    print("CFHT MegaPrime WeightMask Example")
    print("=================================")

    # Download the image
    filename = download_cfht_image()
    if not filename:
        return 1

    # Process with weightmask
    success = run_weightmask(filename)

    if success:
        print("\nOutput files generated:")
        base = filename.replace(".fits.fz", "")
        for suffix in [".weight.fits", ".mask.fits", ".ivar.fits", ".sky.fits"]:
            output_file = f"{base}{suffix}"
            if os.path.exists(output_file):
                print(f"  - {output_file}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

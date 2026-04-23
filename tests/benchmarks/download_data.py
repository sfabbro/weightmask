import argparse
import json
import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import fitsio

try:
    from astroquery.cadc import Cadc
    from astroquery.mast import Observations
except ImportError:
    Cadc = None
    Observations = None

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


def load_manifest(suite_name):
    manifest_path = MANIFEST_DIR / f"{suite_name}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for suite '{suite_name}'")
    with open(manifest_path, "r") as handle:
        return json.load(handle)


def _collect_header_values(path, key):
    values = []
    try:
        with fitsio.FITS(path) as hdul:
            for hdu in hdul:
                try:
                    hdr = hdu.read_header()
                except Exception:
                    continue
                if key in hdr:
                    values.append(str(hdr[key]))
    except Exception:
        return []
    return values


def validate_case_file(case, path):
    expected_instrument = case.get("expected_instrument")
    expected_detector = case.get("expected_detector")
    if not expected_instrument and not expected_detector:
        return True, None

    inst_values = _collect_header_values(path, "INSTRUME")
    det_values = _collect_header_values(path, "DETECTOR")
    if expected_instrument and not any(expected_instrument.lower() in value.lower() for value in inst_values):
        return False, f"expected instrument '{expected_instrument}', got {sorted(set(inst_values)) or 'none'}"
    if expected_detector and not any(expected_detector.lower() in value.lower() for value in det_values):
        return False, f"expected detector '{expected_detector}', got {sorted(set(det_values)) or 'none'}"
    return True, None


def _quarantine_invalid_file(path):
    invalid_path = f"{path}.invalid"
    try:
        if os.path.exists(invalid_path):
            os.remove(invalid_path)
        shutil.move(path, invalid_path)
        print(f"  Moved invalid file to {invalid_path}")
    except Exception as e:
        print(f"  Warning: failed to quarantine invalid file {path}: {e}")


def download_http(url, out_path):
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    print(f"  Downloading from {url} ...")
    try:
        urlretrieve(url, out_path)
        print(f"  Saved to {out_path}")
        return True
    except Exception as e:
        print(f"  HTTP Download failed: {e}")
        return False


def download_cadc(cadc_id, out_path):
    if not Cadc:
        print("  Error: astroquery not installed. Cannot download from CADC.")
        return False

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    print(f"  Querying CADC for ID: {cadc_id} ...")
    try:
        # We assume the ID is a CFHT publisherID such as '2079618p'.
        # Keep this path strict and let manifest-level FITS header validation
        # reject wrong instruments like WIRCam.
        if cadc_id.endswith("p"):
            url = f"https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/{cadc_id}.fits.fz"
            return download_http(url, out_path)

        print(f"  Unsupported CADC ID format: {cadc_id}")
        return False
    except Exception as e:
        print(f"  CADC Resolution failed: {e}")
        return False


def download_cadc_query(query, out_path):
    if not Cadc:
        print("  Error: astroquery not installed.")
        return False

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    print("  Executing CADC query ...")
    cadc = Cadc()
    try:
        results = cadc.exec_sync(query)
        if len(results) > 0:
            urls = cadc.get_data_urls(results)
            if urls:
                return download_http(urls[0], out_path)
        print("  Query returned no results.")
        return False
    except Exception as e:
        print(f"  CADC Query failed: {e}")
        return False


def download_mast(mast_id, product_name, out_path, proposal_id=None):
    if not Observations:
        print("  Error: astroquery not installed. Cannot download from MAST.")
        return False

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return True

    print(f"  Querying MAST for observation: {mast_id} (Proposal: {proposal_id}) ...")
    try:
        # Search by proposal_id if available to narrow down
        if proposal_id:
            obs_table = Observations.query_criteria(proposal_id=proposal_id, instrument_name="ACS/WFC")
            # Filter for the specific rootname if possible
            if len(obs_table) > 0:
                mask = [mast_id.upper() in str(oid).upper() for oid in obs_table["obs_id"]]
                if any(mask):
                    obs_table = obs_table[mask]
                else:
                    # Try searching for everything in the proposal if the rootname isn't in obs_id
                    print(f"  Warning: Rootname {mast_id} not found in proposal {proposal_id} obs_id column.")
        else:
            # Search by obs_id or rootname directly
            obs_table = Observations.query_criteria(obs_id=mast_id)
            if len(obs_table) == 0:
                obs_table = Observations.query_criteria(obs_id=mast_id.upper())

        if len(obs_table) == 0:
            print(f"  No MAST observations found for {mast_id}")
            return False

        products = Observations.get_product_list(obs_table)
        # Search for product in the product table
        mask = [product_name.lower() in str(pf).lower() for pf in products["productFilename"]]
        filtered = products[mask]

        if len(filtered) == 0:
            # Try exact match or contains
            filtered = products[[product_name in f for f in products["productFilename"]]]

        if len(filtered) == 0:
            print(f"  Product {product_name} not found in MAST observation {mast_id}")
            return False

        manifest = Observations.download_products(filtered[0], productType="SCIENCE")
        downloaded_path = manifest["Local Path"][0]

        # Move to desired location
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.move(downloaded_path, out_path)
        print(f"  Saved to {out_path}")
        return True
    except Exception as e:
        print(f"  MAST Download failed: {e}")
        return False


def process_suite(suite_name):
    print(f"\nProcessing suite: {suite_name}")
    manifest = load_manifest(suite_name)

    for case in manifest["cases"]:
        case_id = case["case_id"]
        source = case.get("download_source")
        out_path = ROOT / case["local_path"]

        os.makedirs(out_path.parent, exist_ok=True)

        print(f"Case: {case_id}")

        if os.path.exists(out_path):
            valid, reason = validate_case_file(case, out_path)
            if valid:
                print(f"  Already exists and validated: {out_path}")
                continue
            print(f"  Existing file failed validation: {reason}")
            _quarantine_invalid_file(out_path)

        success = False
        if source == "http":
            success = download_http(case["download_url"], out_path)
        elif source == "cadc":
            success = download_cadc(case["cadc_id"], out_path)
        elif source == "cadc_query":
            success = download_cadc_query(case["cadc_query"], out_path)
        elif source == "mast":
            success = download_mast(
                case["mast_id"], case["mast_product"], out_path, proposal_id=case.get("mast_proposal_id")
            )
        else:
            print(f"  No automated download source for {case_id}")
            continue

        if not success:
            print(f"  FAILED to download data for {case_id}")
            continue

        valid, reason = validate_case_file(case, out_path)
        if not valid:
            print(f"  Downloaded file failed validation: {reason}")
            _quarantine_invalid_file(out_path)
            print(f"  FAILED to download valid data for {case_id}")


def main():
    parser = argparse.ArgumentParser(description="Download real benchmark data for WeightMask.")
    parser.add_argument("--suite", choices=["megacam_real", "acs_compare", "all"], default="all")
    args = parser.parse_args()

    if args.suite == "all":
        for s in ["megacam_real", "acs_compare"]:
            process_suite(s)
    else:
        process_suite(args.suite)


if __name__ == "__main__":
    main()

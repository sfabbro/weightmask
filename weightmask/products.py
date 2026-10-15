"""Output products: per-HDU map assembly, streaming MEF writers, atomic publication."""

from __future__ import annotations

import os
import threading
import uuid

import fitsio
import numpy as np

from . import __version__
from .contract import (
    CONFIDENCE_SEMANTICS,
    CONFIDENCE_SEMANTICS_SCALED,
    CONTRACT_VERSION,
    INVERSE_VARIANCE_SEMANTICS,
    MASK_POLARITY,
    ArtifactMetadata,
    BoundedPrioritySampler,
    ProducerMetadata,
)


def extract_individual_masks(header_info: dict, mask_data):
    keys = ("bad", "sat", "cr", "obj", "streak", "nodata")
    source = header_info.get("individual_masks", {}) if header_info else {}
    if mask_data is None:
        return tuple(np.array([]) for _ in keys)
    shape = mask_data.shape
    out = []
    for k in keys:
        if k in source:
            out.append(source[k])
        else:
            print(f"  WARNING: individual mask '{k}' missing from header_info; emitting all-False.")
            out.append(np.zeros(shape, dtype=bool))
    return tuple(out)


def _store_individual_masks(
    output_data,
    i,
    hdu_header,
    hdu_name,
    bad_mask,
    sat_mask,
    cr_mask,
    obj_mask,
    streak_mask,
    nodata_mask,
):
    if i not in output_data:
        output_data[i] = {}
    output_data[i]["individual_masks"] = {
        "bad": {
            "data": bad_mask.astype(np.uint8),
            "header": header_with_contract_metadata(hdu_header, "bad_mask", mask=True, semantics="boolean_mask"),
            "name": f"BAD_{hdu_name}",
        },
        "sat": {
            "data": sat_mask.astype(np.uint8),
            "header": header_with_contract_metadata(hdu_header, "saturation_mask", mask=True, semantics="boolean_mask"),
            "name": f"SAT_{hdu_name}",
        },
        "cr": {
            "data": cr_mask.astype(np.uint8),
            "header": header_with_contract_metadata(hdu_header, "cosmic_ray_mask", mask=True, semantics="boolean_mask"),
            "name": f"CR_{hdu_name}",
        },
        "obj": {
            "data": obj_mask.astype(np.uint8),
            "header": header_with_contract_metadata(hdu_header, "object_mask", mask=True, semantics="boolean_mask"),
            "name": f"OBJ_{hdu_name}",
        },
        "streak": {
            "data": streak_mask.astype(np.uint8),
            "header": header_with_contract_metadata(hdu_header, "streak_mask", mask=True, semantics="boolean_mask"),
            "name": f"STREAK_{hdu_name}",
        },
        "nodata": {
            "data": nodata_mask.astype(np.uint8),
            "header": header_with_contract_metadata(hdu_header, "nodata_mask", mask=True, semantics="boolean_mask"),
            "name": f"NODATA_{hdu_name}",
        },
    }


def _assign_map_if_valid(output_data, i, key, data, header, hdu_name):
    if data is not None:
        if i not in output_data:
            output_data[i] = {}
        output_data[i][key] = {
            "data": data,
            "header": header,
            "name": f"{key.upper()}_{hdu_name}",
        }


# FITS reserved keywords from the tiled-image compression (fpack) convention.
# They describe the *on-disk* encoding of the input image, not the science
# pixels; copying them onto an uncompressed output makes readers apply the
# compression scaling a second time and corrupts every output value.
_COMPRESSION_HEADER_KEYS = (
    "BZERO",
    "BSCALE",
    "BITPIX",
    "ZQUANTIZ",
    "ZBLANK",
    "ZSIMPLE",
    "ZCMPTYPE",
    "ZNAXIS",
    "ZBITPIX",
    "ZDITHER0",
)


def strip_compression_keywords(header):
    """Remove fpack/quantization keywords from an output header.

    The input HDU header is reused for the output products, but the outputs are
    written as ordinary (uncompressed) FITS. The input's ``BZERO``/``BSCALE``/
    ``BITPIX`` (and the ``Z*`` tile-compression cards) must not be copied,
    otherwise fitsio applies the compression scaling to the already-scaled
    output values -- e.g. a [0,1] weight map came back offset by ``BZERO=32668``
    when the input was fpacked.
    """
    if header is None:
        return header
    out = dict(header)
    for key in list(out.keys()):
        if (
            key in _COMPRESSION_HEADER_KEYS
            or key.startswith("ZTILE")
            or key.startswith("ZNAME")
            or key.startswith("ZVAL")
        ):
            out.pop(key, None)
    return out


def header_with_contract_metadata(header, artifact_type: str, *, mask: bool = False, semantics: str | None = None):
    """Copy a FITS-style header and add portable Wave 5 contract metadata."""
    try:
        output_header = dict(header or {})
    except (TypeError, ValueError):
        output_header = {}
    metadata = ArtifactMetadata(
        artifact_type,
        ProducerMetadata(version=__version__),
        provenance={"producer_stage": "weightmask.cli"},
        mask_polarity=MASK_POLARITY if mask else None,
        semantics=semantics,
    )
    output_header.update(metadata.to_header())
    return output_header


def store_output_maps(
    output_data,
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
    sky_header=None,
):
    if paths["out_map_path"]:
        output_format = config["output_params"]["output_map_format"]
        map_data = confidence_map if output_format == "confidence" else weight_map
        if output_format == "confidence" and config["confidence_params"]["normalize_scope"] == "per_exposure":
            # Normalize once after streaming; per-HDU clipping loses high weights.
            map_data = weight_map
        if output_format != "confidence":
            semantics = "masked_inverse_variance"
        elif config["confidence_params"]["scale_to_100"]:
            semantics = CONFIDENCE_SEMANTICS_SCALED
        else:
            semantics = CONFIDENCE_SEMANTICS
        _assign_map_if_valid(
            output_data,
            i,
            "map",
            map_data,
            header_with_contract_metadata(hdu_header, output_format, semantics=semantics),
            hdu_name,
        )
    if paths["out_mask_path"]:
        _assign_map_if_valid(
            output_data,
            i,
            "mask",
            mask_data,
            header_with_contract_metadata(hdu_header, "quality_mask", mask=True, semantics="named_quality_bits"),
            hdu_name,
        )
    if paths["out_invvar_path"]:
        _assign_map_if_valid(
            output_data,
            i,
            "invvar",
            inv_var_data,
            header_with_contract_metadata(hdu_header, "inverse_variance", semantics=INVERSE_VARIANCE_SEMANTICS),
            hdu_name,
        )
    if paths["out_sky_path"]:
        sky_output_header = sky_header if sky_header is not None else hdu_header
        sky_artifact = "sky_mesh" if (sky_header or {}).get("SKYMESH", False) else "sky"
        _assign_map_if_valid(
            output_data,
            i,
            "sky",
            sky_map,
            header_with_contract_metadata(sky_output_header, sky_artifact, semantics="background_adu"),
            hdu_name,
        )
    if paths["out_weight_raw_path"] and weight_map is not None:
        if i not in output_data:
            output_data[i] = {}
        output_data[i]["weight_raw"] = {
            "data": weight_map,
            "header": header_with_contract_metadata(hdu_header, "weight", semantics="masked_inverse_variance"),
            "name": f"WEIGHT_{hdu_name}",
        }
    if paths["individual_mask_paths"] and mask_data is not None:
        _store_individual_masks(
            output_data,
            i,
            hdu_header,
            hdu_name,
            bad_mask,
            sat_mask,
            cr_mask,
            obj_mask,
            streak_mask,
            nodata_mask,
        )


def rescale_confidence_to_global(paths, writers, config):
    """Normalize streamed raw weights once to exposure-global confidence."""
    if config["output_params"]["output_map_format"] != "confidence":
        print("  Confidence global norm skipped (map product holds weight, not confidence).")
        return True
    map_path = paths.get("out_map_path")
    writer = writers.get("map")
    if not map_path or writer is None:
        return True
    percentile = config["confidence_params"]["normalize_percentile"]
    upper = 100.0 if config["confidence_params"]["scale_to_100"] else 1.0
    entries = list(writer.positions.items())
    writer_identities = getattr(writer, "identities", {})
    identities = [str(writer_identities.get(hdu_index, f"HDU{hdu_index}")) for hdu_index, _pos in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate HDU identity in exposure confidence normalization")
    try:
        with fitsio.FITS(map_path, "rw") as f:
            sampler = BoundedPrioritySampler()
            for identity, (_hdu_index, pos) in zip(identities, entries):
                data = f[pos].read()
                sampler.update(identity, data)
            sample = sampler.sample()
            if not sample.size:
                return True
            normalization = float(np.percentile(sample, percentile))
            if not np.isfinite(normalization) or normalization <= 0:
                print(f"  ERROR: global confidence normalization is invalid ({normalization}).")
                return False
            for _hdu_index, pos in entries:
                data = f[pos].read()
                confidence = np.clip(data / normalization, 0.0, 1.0) * upper
                f[pos].write(confidence.astype(writer.dtype or np.float32, copy=False))
    except OSError as e:
        print(f"  ERROR: confidence global normalization failed: {e}")
        return False
    print(f"  Confidence normalized to exposure-global p{percentile:g} {normalization:.3g}.")
    return True


def ensure_hdu_extname(fits_obj, pos, extname):
    current = fits_obj[pos].read_header().get("EXTNAME")
    if str(current or "").strip() != extname:
        fits_obj[pos].write_key("EXTNAME", extname)


_PRODUCT_PATH_KEYS = (
    "out_map_path",
    "out_mask_path",
    "out_invvar_path",
    "out_sky_path",
    "out_weight_raw_path",
)


_INDIVIDUAL_MASK_KEYS = ("bad", "sat", "cr", "obj", "streak", "nodata")


def _ordered_product_paths(paths):
    products = [(key, paths.get(key)) for key in _PRODUCT_PATH_KEYS]
    individual = paths.get("individual_mask_paths") or {}
    products.extend((f"individual_mask_paths.{key}", individual.get(key)) for key in _INDIVIDUAL_MASK_KEYS)
    return [(key, os.fspath(path)) for key, path in products if path]


class ProductPublication:
    def __init__(self, paths):
        self.generation_id = uuid.uuid4().hex
        self._entries = []
        self._committed = False
        self._rollback_incomplete = False
        temporary_paths = dict(paths)
        temporary_paths["individual_mask_paths"] = dict(paths.get("individual_mask_paths") or {})
        temporary_paths["_generation_id"] = self.generation_id
        for key, final_path in _ordered_product_paths(paths):
            directory = os.path.dirname(os.path.abspath(final_path))
            basename = os.path.basename(final_path)
            temporary = os.path.join(directory, f".{basename}.wm-tmp-{self.generation_id}")
            backup = os.path.join(directory, f".{basename}.wm-bak-{self.generation_id}")
            self._entries.append((final_path, temporary, backup))
            if key.startswith("individual_mask_paths."):
                temporary_paths["individual_mask_paths"][key.rsplit(".", 1)[1]] = temporary
            else:
                temporary_paths[key] = temporary
        self.paths = temporary_paths

    @property
    def temporary_path(self):
        if len(self._entries) != 1:
            raise ValueError("temporary_path is only defined for a single-product publication")
        return self._entries[0][1]

    def validate(self, expected_hdus):
        expected_count = len(expected_hdus)
        reference_hdu_count = None
        reference_shapes = None
        reference_names = None
        for _final, temporary, _backup in self._entries:
            with fitsio.FITS(temporary, "r") as handle:
                if reference_hdu_count is None:
                    reference_hdu_count = len(handle)
                elif len(handle) != reference_hdu_count:
                    raise ValueError("output products have different HDU counts")
                images = [
                    item for item in handle if item.get_info().get("hdutype") == 0 and item.get_info().get("ndims") == 2
                ]
                if len(images) != expected_count:
                    raise ValueError(f"output contains {len(images)} image HDUs; expected {expected_count}")
                shapes = []
                names = []
                for item in images:
                    header = item.read_header()
                    generation = str(header.get("WMGENID", ""))
                    if generation != self.generation_id:
                        raise ValueError("output product generation ID is missing or inconsistent")
                    if str(header.get("WMVERS", "")) != CONTRACT_VERSION or not header.get("WMART"):
                        raise ValueError("output product contract metadata is missing or inconsistent")
                    shape = tuple(item.get_dims())
                    name = str(header.get("EXTNAME", "")).strip()
                    if not name:
                        raise ValueError("output product EXTNAME is missing")
                    shapes.append(shape)
                    names.append(name.split("_", 1)[-1])
                if len(set(names)) != len(names):
                    raise ValueError("output product EXTNAME values are not unique")
                if reference_shapes is None:
                    reference_shapes = shapes
                    reference_names = names
                elif shapes != reference_shapes or names != reference_names:
                    raise ValueError("output product HDU shapes or EXTNAME values are not synchronized")

    def promote(self):
        moved = []
        try:
            for final, temporary, backup in self._entries:
                had_original = os.path.exists(final)
                if had_original:
                    os.replace(final, backup)
                moved.append((final, temporary, backup, had_original))
                os.replace(temporary, final)
        except OSError as promotion_error:
            rollback_errors = []
            for final, temporary, backup, had_original in reversed(moved):
                try:
                    if had_original:
                        if not os.path.exists(backup):
                            raise OSError(f"rollback backup is missing: {backup}")
                        os.replace(backup, final)
                    elif os.path.exists(final):
                        try:
                            os.replace(final, temporary)
                        except OSError as quarantine_error:
                            try:
                                os.unlink(final)
                            except OSError as unlink_error:
                                raise OSError(
                                    f"could not quarantine ({quarantine_error}) or remove new output: {unlink_error}"
                                ) from quarantine_error
                except OSError as exc:
                    rollback_errors.append(exc)
            if rollback_errors:
                self._rollback_incomplete = True
                raise OSError(
                    f"publication failed ({promotion_error}) and rollback was incomplete: {rollback_errors[0]}"
                ) from promotion_error
            raise
        self._committed = True

    def cleanup(self):
        pending = []
        for _final, temporary, backup in self._entries:
            paths = (temporary,) if self._rollback_incomplete else (temporary, backup)
            for path in paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    pending.append((path, backup, exc))
        for path, backup, first_error in pending:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                kind = "post-commit backup" if self._committed and path == backup else "publication artifact"
                print(f"  WARNING: Could not remove {kind} '{path}': {exc}; first attempt failed: {first_error}")


class _StreamingMapWriter:
    """Stream one map product (map/mask/invvar/sky/...) into a MEF file.

    HDUs are appended as they are produced, so a large multi-extension input
    never has to be held in memory at once. The on-disk layout matches the
    previous buffered writer exactly:

    * a single-extension input with the image at HDU 0 -> a one-HDU file;
    * a MEF whose primary is itself an image -> that image becomes the primary
      and the remaining images are appended as extensions;
    * a MEF whose primary is not an image -> an empty primary header is written
      first, then every image is appended as an extension.
    """

    def __init__(
        self,
        out_path: str,
        hdul_input,
        primary_header,
        compress: bool = False,
        dtype=None,
        generation_id: str | None = None,
    ):
        self.out_path = out_path
        self.hdul_input = hdul_input
        self.primary_header = dict(primary_header or {})
        self._opened = False
        self.positions: dict = {}
        self.identities: dict = {}
        self._lock = threading.Lock()
        self.compress = bool(compress)
        self.dtype = np.dtype(dtype) if dtype is not None else None
        self.generation_id = generation_id
        if generation_id:
            self.primary_header["WMGENID"] = generation_id

    def _prep(self, data):
        if data is None or self.dtype is None:
            return data
        try:
            return np.ascontiguousarray(data, dtype=self.dtype)
        except Exception:
            return np.asarray(data, dtype=self.dtype)

    def write(self, hdu_index: int, data, header, extname: str) -> None:
        data = self._prep(data)
        header = dict(header or {})
        header["EXTNAME"] = extname
        if self.generation_id:
            header["WMGENID"] = self.generation_id
        # Only pass ``compress`` when compressing: fitsio warns about (and
        # ignores) a placeholder value such as "NOT_SET".
        kwargs = {"compress": "RICE_1"} if self.compress else {}
        with self._lock:
            self.identities[hdu_index] = extname
            if not self._opened:
                if hdu_index == 0:
                    primary_data = data
                    primary_header = header
                else:
                    primary_data = None
                    primary_header = self.primary_header
                fitsio.write(self.out_path, primary_data, header=primary_header, clobber=True, **kwargs)
                self._opened = True
                if hdu_index == 0:
                    with fitsio.FITS(self.out_path, "rw") as f_out:
                        # Compressed primary images live at HDU 1, not HDU 0.
                        position = len(f_out) - 1
                        ensure_hdu_extname(f_out, position, extname)
                        self.positions[hdu_index] = position
                    return
            with fitsio.FITS(self.out_path, "rw") as f_out:
                f_out.write(data, header=header, extname=extname, **kwargs)
                position = len(f_out) - 1
                ensure_hdu_extname(f_out, position, extname)
                self.positions[hdu_index] = position


def _wire_dtype(bitpix, *, is_mask: bool):
    """Map configured bitpix to a numpy wire dtype (mask uint, float otherwise)."""
    if is_mask:
        return {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}[bitpix]
    return {32: np.float32, 64: np.float64, -32: np.float32, -64: np.float64}[bitpix]


def make_output_writers(paths: dict, hdul_input, config: dict) -> dict:
    """Build a streaming writer per requested output product."""
    primary_header = None
    if len(hdul_input) > 0:
        try:
            primary_header = strip_compression_keywords(hdul_input[0].read_header())
        except Exception:
            primary_header = None
    op = config["output_params"]
    compress = op["compress"]
    generation_id = paths.get("_generation_id")
    mask_dt = _wire_dtype(op["mask_bitpix"], is_mask=True)
    float_dt = _wire_dtype(op["ivar_bitpix"], is_mask=False)
    writers: dict = {}
    for key, path_key in (
        ("map", "out_map_path"),
        ("mask", "out_mask_path"),
        ("invvar", "out_invvar_path"),
        ("sky", "out_sky_path"),
        ("weight_raw", "out_weight_raw_path"),
    ):
        out_path = paths.get(path_key)
        if out_path:
            dt = mask_dt if key == "mask" else float_dt
            writers[key] = _StreamingMapWriter(
                out_path,
                hdul_input,
                primary_header,
                compress=compress,
                dtype=dt,
                generation_id=generation_id,
            )
    for mask_type in ("bad", "sat", "cr", "obj", "streak", "nodata"):
        out_path = (paths.get("individual_mask_paths") or {}).get(mask_type)
        if out_path:
            writers[f"ind_{mask_type}"] = _StreamingMapWriter(
                out_path,
                hdul_input,
                primary_header,
                compress=compress,
                dtype=np.uint8,
                generation_id=generation_id,
            )
    return writers


def flush_hdu_output(writers: dict, hdu_output: dict, hdu_index: int) -> None:
    """Write a single HDU's buffered maps to their output files and release them."""
    entry_map = hdu_output.get(hdu_index) or {}
    for key in ("map", "mask", "invvar", "sky", "weight_raw"):
        entry = entry_map.get(key)
        if entry is not None and key in writers:
            writers[key].write(hdu_index, entry["data"], entry["header"], entry["name"])

    individual = entry_map.get("individual_masks") or {}
    for mask_type in ("bad", "sat", "cr", "obj", "streak", "nodata"):
        entry = individual.get(mask_type)
        writer_key = f"ind_{mask_type}"
        if entry is not None and writer_key in writers:
            writers[writer_key].write(hdu_index, entry["data"], entry["header"], entry["name"])

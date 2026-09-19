"""Guard the curated real-MegaCam label set and the geometry it rests on.

The science exposures are not committed, so these tests cannot re-run the
detectors.  What they *can* do -- and what matters for a fixture other people
score against -- is pin the labelling rule to the evidence stored beside every
label: a `trail` row must still satisfy every gate the curator applied, an
`artefact` row must still name the veto that produced it, and the tallies must
match the rows.  Relabelling without evidence then fails here.

The geometry helpers get their own tests because two separate sign/convention
mistakes in them produced convincing-looking false labels during curation (see
`test_mirrored_chips_do_not_share_a_mosaic_line`).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmarks.curate_trail_truth import (
    SCHEMA,
    _angle_between,
    _connected_groups,
    _normalise_normal,
    find_chip_replicas,
    line_to_mosaic,
    longest_support_run,
    write_fixture,
)

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks" / "trail_truth" / "megacam_real_labels.json"
MEGACAM_SHAPE = (4644, 2112)


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE) as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def tolerances(fixture):
    return fixture["tolerances"]


# --------------------------------------------------------------------------- #
# fixture invariants
# --------------------------------------------------------------------------- #
def test_schema_and_provenance(fixture):
    assert fixture["schema"] == SCHEMA
    assert fixture["instrument"] == "MegaPrime/MegaCam"
    assert fixture["data_requirements"]["exposures"], "no exposures recorded"
    for name, info in fixture["data_requirements"]["exposures"].items():
        assert info["path"].endswith((".fits", ".fits.fz")), name
        assert info["pointing_group"], name
    assert fixture["provenance"]["runs"], "the fixture must record how it was produced"


def test_fixture_carries_only_labelled_entries(fixture):
    labels = {entry["label"] for entry in fixture["entries"]}
    assert labels <= {"trail", "artefact"}, "undecided rows belong in the review sheet, not the fixture"
    assert fixture["entries"], "the fixture is empty"


def test_summary_matches_the_entries(fixture):
    entries = fixture["entries"]
    summary = fixture["summary"]
    assert summary["n_entries"] == len(entries)
    for label in ("trail", "artefact", "uncertain"):
        assert summary["by_label"][label] == sum(1 for entry in entries if entry["label"] == label), label

    by_veto = {}
    by_proposer = {}
    by_exposure = {}
    for entry in entries:
        by_exposure.setdefault(entry["exposure"], {"trail": 0, "artefact": 0, "uncertain": 0})[entry["label"]] += 1
        key = "+".join(entry["proposed_by"]) or "unknown"
        by_proposer[key] = by_proposer.get(key, 0) + 1
        if entry["label"] == "artefact":
            veto = _veto_of(entry)
            by_veto[veto] = by_veto.get(veto, 0) + 1
    assert summary["by_veto"] == by_veto
    assert summary["by_proposer"] == by_proposer
    for exposure, counts in summary["by_exposure"].items():
        assert counts == by_exposure[exposure]


def _veto_of(entry):
    if entry["evidence"]["static_matches"]:
        return "static_across_epochs"
    if entry["evidence"]["axis_veto"]:
        return entry["evidence"]["axis_veto"]["kind"]
    if entry["mosaic"] and entry["mosaic"]["group"]["chip_replicated"]:
        return "chip_replica"
    return "unattributed"


def test_every_entry_names_a_known_exposure(fixture):
    known = set(fixture["data_requirements"]["exposures"])
    for entry in fixture["entries"]:
        assert entry["exposure"] in known, entry["exposure"]


def test_trail_rows_still_satisfy_every_gate(fixture, tolerances):
    """The curation rule must be reproducible from the stored evidence alone."""
    trails = [entry for entry in fixture["entries"] if entry["label"] == "trail"]
    for entry in trails:
        where = f"{entry['exposure']}:{entry['hdu']}:{entry['geometry']['offset_px']}"
        mosaic = entry["mosaic"]
        assert mosaic is not None, where
        assert mosaic["group"]["n_ccds"] >= tolerances["min_group_ccds"], where
        assert not mosaic["group"]["chip_replicated"], where
        assert not entry["evidence"]["static_matches"], where
        assert entry["evidence"]["axis_veto"] is None, where
        assert entry["evidence"]["axis_alignment_deg"] >= tolerances["min_axis_deg"], where
        independent = entry["independent"]
        assert independent["support_px"] >= tolerances["min_support_px"], where
        assert independent["z_max"] >= tolerances["min_z_max"], where


def test_artefact_rows_name_their_veto(fixture):
    for entry in fixture["entries"]:
        if entry["label"] != "artefact":
            continue
        assert _veto_of(entry) != "unattributed", (
            f"{entry['exposure']}:{entry['hdu']} is labelled artefact without a recorded veto"
        )


def test_no_artefact_is_also_multi_ccd_clean(fixture, tolerances):
    """Nothing labelled artefact may satisfy the full trail rule."""
    for entry in fixture["entries"]:
        if entry["label"] != "artefact":
            continue
        mosaic = entry["mosaic"]
        clean = (
            mosaic is not None
            and mosaic["group"]["n_ccds"] >= tolerances["min_group_ccds"]
            and not mosaic["group"]["chip_replicated"]
            and not entry["evidence"]["static_matches"]
            and entry["evidence"]["axis_veto"] is None
            and entry["evidence"]["axis_alignment_deg"] >= tolerances["min_axis_deg"]
            and entry["independent"]["support_px"] >= tolerances["min_support_px"]
            and entry["independent"]["z_max"] >= tolerances["min_z_max"]
        )
        assert not clean, f"{entry['exposure']}:{entry['hdu']} is labelled artefact but passes every trail gate"


def test_finding_is_consistent_with_the_tallies(fixture):
    finding = fixture["finding"]
    assert finding["real_trails_found"] == fixture["summary"]["by_label"]["trail"]
    assert finding["measured"], "an empty positive set must come with the measurement that justifies it"
    if finding["real_trails_found"] == 0:
        assert "consequence" in finding and finding["consequence"], (
            "no real trails means the fixture can only score false positives; say so"
        )


def test_geometry_is_self_consistent(fixture):
    for entry in fixture["entries"]:
        geometry = entry["geometry"]
        normal = np.array(geometry["normal"], dtype=float)
        direction = np.array(geometry["direction"], dtype=float)
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-5, entry["geometry"]["point"]
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-5, entry["geometry"]["point"]
        assert abs(float(normal @ direction)) < 1e-5, "normal must be perpendicular to direction"
        assert geometry["span_along_px"] > 0.0
        # The line must actually cross a MegaCam CCD, or the scored band is empty.
        from benchmarks.curate_trail_truth import _clip_line_to_shape

        assert _clip_line_to_shape(geometry["point"], geometry["direction"], MEGACAM_SHAPE) is not None


def test_write_fixture_round_trips(tmp_path, fixture):
    target = tmp_path / "labels.json"
    write_fixture(fixture, str(target))
    assert target.read_text() == FIXTURE.read_text()


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _chip(cd1_1, crpix1, cd2_2, crpix2, crval=(0.0, 0.0)):
    return {
        "crval": list(crval),
        "crpix": [crpix1, crpix2],
        "cd": [[cd1_1, 0.0], [0.0, cd2_2]],
        "ctypes": ["RA---TAN", "DEC--TAN"],
        "shape": [2112, 4644],
    }


def test_lines_on_opposite_sides_do_not_look_identical():
    """Canonicalising the normal's sign without the offset makes two distinct
    parallel lines share a representation.

    MegaCam chips are mirrored in ``CD1_1``, so a chip-vertical line's mosaic
    direction -- and hence the sign of its offset -- flips from chip to chip.
    Canonicalising only the normal therefore lets a line at ``+xi`` and one at
    ``-xi`` compare as equal, which is one of the ways this tool produced bogus
    cross-CCD groups before the pair was flipped jointly.
    """
    scale = 5.194e-05
    positive = _chip(+scale, 0.0, -scale, 0.0)
    negative = _chip(-scale, 0.0, +scale, 0.0)
    left = line_to_mosaic([1000.0, 500.0], [0.0, 1.0], positive)
    right = line_to_mosaic([1000.0, 500.0], [0.0, 1.0], negative)
    assert left is not None and right is not None
    # Parallel, but on opposite sides of the mosaic centre.
    assert _angle_between(left["normal"], right["normal"]) == pytest.approx(0.0, abs=1e-9)
    assert left["offset_deg"] > 0.0 > right["offset_deg"]
    assert abs(left["offset_deg"]) == pytest.approx(abs(right["offset_deg"]), rel=1e-9)
    # Flipping both members jointly keeps them distinct: the pair is 2000 px
    # apart, far outside any grouping tolerance, so they cannot be merged.
    separation_px = (left["offset_deg"] - right["offset_deg"]) / scale
    assert separation_px > 1000.0
    # Neither may be reported with the other's sign, which is how the mirrored
    # chip's line would otherwise be indistinguishable from its counterpart.
    assert math.copysign(1.0, left["offset_deg"]) != math.copysign(1.0, right["offset_deg"])


def test_mirrored_chip_vertical_columns_share_a_mosaic_column():
    """The real pair behind the +-xi episode: two adjacent chips, one mirrored.

    Both bad columns land at the same positive ``xi`` because the mirrored chip's
    negative CD cancels its negative offset from the reference pixel.  They are
    genuinely collinear, which is why the along-line connectivity and axis tests
    -- not the sign convention -- are what reject them.
    """
    scale = 5.194e-05
    upper = _chip(+scale, -1049.3, -scale, 4638.0)
    lower = _chip(-scale, 3179.9, +scale, 4625.3)
    a = line_to_mosaic([1654.5, 2373.914], [0.0, 1.0], upper)
    b = line_to_mosaic([468.5, 2528.0], [0.0, 1.0], lower)
    assert a is not None and b is not None
    assert a["offset_deg"] > 0.0 and b["offset_deg"] > 0.0
    assert abs(a["offset_deg"] - b["offset_deg"]) / scale < 15.0, (
        "this pair is meant to coincide within the grouping tolerance of 15 px"
    )


def test_line_to_mosaic_normal_and_offset_are_consistent():
    """The reported normal must reproduce the reported offset."""
    chip = _chip(+5.194e-05, -7300.0, -5.194e-05, 9635.0)
    for point, direction in (
        ([100.0, 200.0], [1.0, 0.0]),
        ([1000.0, 3000.0], [0.0, 1.0]),
        ([500.0, 4000.0], [math.cos(0.7), math.sin(0.7)]),
    ):
        mosaic = line_to_mosaic(point, direction, chip)
        assert mosaic is not None
        normal = np.array(mosaic["normal"])
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-9
        from benchmarks.curate_trail_truth import ccd_to_mosaic

        xy = ccd_to_mosaic([point], chip)[0]
        assert float(normal @ xy) == pytest.approx(mosaic["offset_deg"], abs=1e-9)


def test_normalise_normal_canonicalises_sign():
    assert _normalise_normal(-1.0, 0.0) == (1.0, 0.0)
    assert _normalise_normal(0.0, -2.0) == (0.0, 1.0)
    assert _normalise_normal(0.0, 0.0) == (0.0, 0.0)


def test_angle_between_is_sign_insensitive():
    assert _angle_between((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(0.0)
    assert _angle_between((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)
    assert _angle_between((1.0, 0.0), (math.cos(0.1), math.sin(0.1))) == pytest.approx(math.degrees(0.1))


def test_longest_support_run_finds_the_ridge():
    profile = np.zeros(100)
    profile[10:40] = 5.0
    span, start, peak = longest_support_run(profile, threshold=4.0, max_gap_samples=0)
    assert (span, start) == (30, 10)
    assert peak == pytest.approx(5.0)


def test_longest_support_run_bridges_small_gaps_only():
    profile = np.zeros(60)
    profile[5:20] = 6.0
    profile[21:35] = 7.0  # exactly one missing sample at index 20
    span, start, _ = longest_support_run(profile, threshold=4.0, max_gap_samples=1)
    assert start == 5 and span == 30
    profile_gap2 = np.zeros(60)
    profile_gap2[5:20] = 6.0
    profile_gap2[22:35] = 7.0  # two missing samples: not bridged
    span_gap2, start_gap2, _ = longest_support_run(profile_gap2, threshold=4.0, max_gap_samples=1)
    assert (start_gap2, span_gap2) == (5, 15)
    # A wider hole splits the run; the longer side wins.
    profile2 = np.zeros(60)
    profile2[5:20] = 6.0
    profile2[30:35] = 7.0
    span2, start2, _ = longest_support_run(profile2, threshold=4.0, max_gap_samples=1)
    assert start2 == 5 and span2 == 15


def test_longest_support_run_on_empty_profile():
    assert longest_support_run(np.zeros(10), threshold=4.0) == (0, 0, 0.0)


def _member(offset, normal=(1.0, 0.0), support=500.0):
    return {
        "measured_offset": offset,
        "normal": list(normal),
        "independent": {"support_px": support},
    }


def test_find_chip_replicas_flags_equal_ccd_local_lines():
    members = [_member(1000.0), _member(1002.0), _member(-4000.0)]
    replicas = find_chip_replicas(members, tol_px=8.0, tol_deg=2.0)
    assert set(replicas) == {0, 1}


def test_find_chip_replicas_ignores_different_angles():
    members = [_member(1000.0), _member(1000.0, normal=(math.cos(0.2), math.sin(0.2)))]
    assert find_chip_replicas(members, tol_px=8.0, tol_deg=2.0) == {}


def test_connected_groups_splits_on_along_line_gaps():
    near = {"mosaic": {"t_range_px": [0.0, 900.0]}}
    adjacent = {"mosaic": {"t_range_px": [1000.0, 2000.0]}}
    far = {"mosaic": {"t_range_px": [9000.0, 12000.0]}}
    chunks = _connected_groups([near, adjacent, far], max_gap_px=800.0)
    assert [len(chunk) for chunk in chunks] == [2, 1]
    merged = _connected_groups([near, adjacent, far], max_gap_px=20000.0)
    assert [len(chunk) for chunk in merged] == [3]


def test_connected_groups_keeps_members_without_a_line_together():
    without = {}
    positioned = {"mosaic": {"t_range_px": [0.0, 10.0]}}
    chunks = _connected_groups([positioned, without], max_gap_px=10.0)
    assert sorted(len(chunk) for chunk in chunks) == [1, 1]

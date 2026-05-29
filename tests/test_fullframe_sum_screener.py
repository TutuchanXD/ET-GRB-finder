import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "GRB_from_fullframe_uint16_sum_template_match_noplot.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("fullframe_screener", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_run(tmp_path: Path, n_frames: int = 4, shape=(8, 9)) -> Path:
    run = tmp_path / "run"
    frames = run / "frames"
    frames.mkdir(parents=True)
    for index in range(n_frames):
        arr = np.full(shape, index, dtype=np.uint16)
        np.save(frames / f"frame_{index:06d}.npy", arr)
    return run


def make_residual_run_pair(tmp_path: Path, n_frames: int = 2, shape=(12, 12)) -> tuple[Path, Path]:
    run = tmp_path / "run"
    template = tmp_path / "template"
    run_frames = run / "frames"
    template_frames = template / "frames"
    run_frames.mkdir(parents=True)
    template_frames.mkdir(parents=True)
    for index in range(n_frames):
        tmpl = np.full(shape, 100, dtype=np.uint16)
        cur = tmpl.copy()
        cur[5, 5] += 50
        cur[5, 6] += 30
        np.save(template_frames / f"frame_{index:06d}.npy", tmpl)
        np.save(run_frames / f"frame_{index:06d}.npy", cur)
    return run, template


def test_frame_paths_from_current_run_layout(tmp_path):
    mod = load_module()
    run = make_run(tmp_path, n_frames=3)

    paths = mod.frame_paths_from_run(run)

    assert [path.name for path in paths] == [
        "frame_000000.npy",
        "frame_000001.npy",
        "frame_000002.npy",
    ]


def test_sum_frame_tile_uses_half_open_frame_window(tmp_path):
    mod = load_module()
    run = make_run(tmp_path, n_frames=4, shape=(8, 9))
    paths = mod.frame_paths_from_run(run)

    summed = mod.sum_frame_tile(paths, 1, 4, slice(2, 5), slice(3, 7))

    assert summed.shape == (3, 4)
    assert summed.dtype == np.uint32
    assert np.all(summed == 1 + 2 + 3)


def test_template_source_brightening_is_kept_not_dropped():
    mod = load_module()
    candidates = [{"x": 5, "y": 6, "source_peak_value": 100}]
    template_xy = np.array([[5.2, 6.1]], dtype=float)

    rows = mod.annotate_template_matches(candidates, template_xy, match_radius_px=0.75)

    assert len(rows) == 1
    assert rows[0]["template_match_flag"] == 1
    assert rows[0]["candidate_channel"] == "template_source_brightening"


def test_detect_sources_collapses_connected_saturated_plateau():
    mod = load_module()
    cfg = mod.ScreenerConfig(max_filter_size=3)
    tile = np.zeros((9, 9), dtype=np.uint32)
    tile[3:5, 4:6] = 100

    rows, threshold, _bkg_median, _bkg_sigma = mod.detect_sources_on_tile(
        tile,
        row_origin=10,
        col_origin=20,
        core_bounds=(10, 19, 20, 29),
        cfg=cfg,
    )

    assert threshold == 0
    assert len(rows) == 1
    assert rows[0]["source_peak_value"] == 100
    assert 24 <= rows[0]["x"] <= 25
    assert 13 <= rows[0]["y"] <= 14


def test_residual_detection_ignores_static_stars_and_keeps_brightening():
    mod = load_module()
    cfg = mod.ScreenerConfig(max_filter_size=3, residual_threshold_sigma=3.0)
    template = np.full((15, 15), 1000, dtype=np.uint32)
    summed = template.copy()

    template[3, 3] = 5000
    summed[3, 3] = 5000
    template[10, 10] = 6000
    summed[10, 10] = 6000
    summed[7, 7] += 100
    summed[7, 8] += 50
    summed[8, 7] += 40

    rows, threshold, bkg_median, bkg_sigma = mod.detect_residual_sources_on_tile(
        summed,
        template,
        row_origin=0,
        col_origin=0,
        core_bounds=(0, 15, 0, 15),
        cfg=cfg,
    )

    assert bkg_median == 0
    assert bkg_sigma == 0
    assert threshold == 0
    assert len(rows) == 1
    assert rows[0]["x"] == 7
    assert rows[0]["y"] == 7
    assert rows[0]["residual_npix"] == 3
    assert rows[0]["residual_flux"] == 190


def test_residual_detection_rejects_single_pixel_cosmic_shape_by_default():
    mod = load_module()
    cfg = mod.ScreenerConfig(max_filter_size=3)
    template = np.zeros((11, 11), dtype=np.uint32)
    summed = template.copy()
    summed[5, 5] = 1000

    rows, _threshold, _bkg_median, _bkg_sigma = mod.detect_residual_sources_on_tile(
        summed,
        template,
        row_origin=0,
        col_origin=0,
        core_bounds=(0, 11, 0, 11),
        cfg=cfg,
    )

    assert rows == []


def test_candidate_local_xy_uses_column_origin_for_x():
    mod = load_module()

    local_x, local_y = mod.candidate_local_xy(
        {"x": 14, "y": 4},
        row_slice=slice(0, 12),
        col_slice=slice(8, 20),
    )

    assert (local_x, local_y) == (6, 4)


def test_measure_candidate_local_excess_keeps_sparse_zero_sigma_diff():
    mod = load_module()
    cfg = mod.ScreenerConfig(cut_half=4, annulus_r_in=3.0, annulus_r_out=4.0)
    template = np.full((15, 15), 100, dtype=np.uint32)
    summed = template.copy()
    summed[7, 7] += 50
    summed[7, 8] += 25
    summed[8, 7] += 25

    flux, npix, peak, bkg_med, bkg_sigma = mod.measure_candidate_local_excess(
        summed,
        template,
        (7, 7),
        cfg,
    )

    assert flux == 100
    assert npix == 3
    assert peak == 50
    assert bkg_med == 0
    assert bkg_sigma == 0


def test_truth_matching_is_optional_and_uses_events_csv(tmp_path):
    mod = load_module()
    truth = tmp_path / "events.csv"
    with truth.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "requested_detector_xpix",
                "requested_detector_ypix",
                "first_visible_frame",
                "last_visible_frame",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "event_id": 7,
                "requested_detector_xpix": 10.2,
                "requested_detector_ypix": 20.4,
                "first_visible_frame": 12,
                "last_visible_frame": 30,
            }
        )

    rows = mod.load_truth_events(truth)
    matched = mod.annotate_truth_matches(
        [{"x": 11, "y": 20, "frame_start": 12, "frame_end": 23}],
        rows,
        radius_px=2.0,
    )

    assert matched[0]["truth_event_id"] == "7"
    assert matched[0]["truth_match_flag"] == 1


def test_main_does_not_write_template_source_catalog_by_default(tmp_path):
    mod = load_module()
    run = make_run(tmp_path, n_frames=2)
    out = tmp_path / "out"

    rc = mod.main(
        [
            "--input-run",
            str(run),
            "--output-dir",
            str(out),
            "--window-size",
            "2",
            "--stride",
            "2",
            "--max-windows",
            "1",
        ]
    )

    assert rc == 0
    assert not (out / "template_sources.csv").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["template_match_sources"] is False


def test_post_residual_checks_can_be_disabled_from_cli(tmp_path):
    mod = load_module()
    run, template = make_residual_run_pair(tmp_path)
    out = tmp_path / "out"

    rc = mod.main(
        [
            "--input-run",
            str(run),
            "--template-run",
            str(template),
            "--output-dir",
            str(out),
            "--window-size",
            "2",
            "--stride",
            "2",
            "--max-windows",
            "1",
            "--no-local-shape-check",
            "--no-temporal-check",
            "--keep-all-residual-candidates",
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["local_shape_check"] is False
    assert manifest["temporal_check"] is False
    assert manifest["keep_all_residual_candidates"] is True
    assert manifest["candidates_after_measurement"] == 1
    assert manifest["final_candidates"] == 1

    rows = list(csv.DictReader((out / "streaming_sum_transient_candidates.csv").open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["pass_single_stack"] == "1"
    assert row["local_shape_check_enabled"] == "0"
    assert row["temporal_check_enabled"] == "0"
    assert int(row["local_excess_flux"]) == int(row["residual_flux"])
    assert int(row["peak_npix"]) == int(row["residual_npix"])
    assert row["temporal_flux_series"] == "[]"

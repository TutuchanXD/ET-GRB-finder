import csv
import importlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCRIPT_DEFAULT_KEYS = {
    "input_run",
    "template_run",
    "output_dir",
    "truth_events_csv",
    "truth_match_radius_px",
    "spatial_bin",
    "window_size",
    "stride",
    "max_windows",
    "template_strategy",
    "tile_size",
    "halo",
    "input_bit_depth",
    "max_filter_size",
    "source_threshold_sigma",
    "residual_threshold_sigma",
    "residual_min_npix",
    "residual_max_npix",
    "min_residual_peak_value",
    "min_residual_flux",
    "min_flux_peak_ratio",
    "max_final_candidates_per_window",
    "match_radius_px",
    "cut_half",
    "annulus_r_in",
    "annulus_r_out",
    "local_threshold_sigma",
    "seed_radius",
    "effective_npix_threshold",
    "peak_pixel_snr_check",
    "peak_pixel_snr_threshold",
    "temporal_cut_half",
    "temporal_aperture_radius",
    "temporal_annulus_r_in",
    "temporal_annulus_r_out",
    "temporal_sigma",
    "temporal_min_active_frames",
    "cosmic_single_frame_fraction",
    "cosmic_max_active_frames",
    "previous_match_radius_px",
    "template_match_sources",
    "local_shape_check",
    "temporal_check",
    "keep_all_residual_candidates",
    "overwrite",
}


def load_module():
    return importlib.import_module("grbfinder.api")


def load_script_module(script: str):
    path = REPO_ROOT / script
    module_name = "test_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
    assert manifest["template_strategy"] == "paired-template"
    assert manifest["candidates_after_measurement"] == 1
    assert manifest["final_candidates"] == 1

    summary_rows = list(csv.DictReader((out / "streaming_sum_summary.csv").open()))
    assert [
        (row["frame_start"], row["frame_end"], row["template_frame_start"], row["template_frame_end"])
        for row in summary_rows
    ] == [("0", "1", "0", "1")]

    rows = list(csv.DictReader((out / "streaming_sum_transient_candidates.csv").open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["pass_single_stack"] == "1"
    assert row["local_shape_check_enabled"] == "0"
    assert row["temporal_check_enabled"] == "0"
    assert int(row["local_excess_flux"]) == int(row["residual_flux"])
    assert int(row["peak_npix"]) == int(row["residual_npix"])
    assert row["temporal_flux_series"] == "[]"


def test_rectangular_spatial_bin_writes_manifest_fields(tmp_path):
    mod = load_module()
    run = make_run(tmp_path, n_frames=4, shape=(6, 8))
    out = tmp_path / "out"

    rc = mod.main(
        [
            "--input-run",
            str(run),
            "--output-dir",
            str(out),
            "--spatial-bin",
            "3x4",
            "--window-size",
            "2",
            "--stride",
            "2",
            "--no-local-shape-check",
            "--no-temporal-check",
            "--keep-all-residual-candidates",
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["spatial_bin_size"] is None
    assert manifest["spatial_bin_rows"] == 3
    assert manifest["spatial_bin_cols"] == 4
    assert manifest["detection_shape"] == [2, 2]
    assert manifest["cropped_input_shape"] == [6, 8]


def test_onboard_first_window_is_template_only_without_external_template(tmp_path):
    mod = load_module()
    run = tmp_path / "run"
    frames = run / "frames"
    frames.mkdir(parents=True)
    for index in range(4):
        arr = np.full((12, 12), 100, dtype=np.uint16)
        if index >= 2:
            arr[5, 5] += 80
            arr[5, 6] += 60
        np.save(frames / f"frame_{index:06d}.npy", arr)
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
            "--no-local-shape-check",
            "--no-temporal-check",
            "--keep-all-residual-candidates",
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["template_window"] == [0, 1]
    assert manifest["detection_windows_processed"] == 1
    assert manifest["windows_processed"] == 1

    summary_rows = list(csv.DictReader((out / "streaming_sum_summary.csv").open()))
    assert [(row["frame_start"], row["frame_end"]) for row in summary_rows] == [("2", "3")]


def test_rolling_previous_template_strategy_uses_previous_complete_window(tmp_path):
    mod = load_module()
    run = tmp_path / "run"
    frames = run / "frames"
    frames.mkdir(parents=True)
    for index in range(6):
        arr = np.full((12, 12), 100, dtype=np.uint16)
        if index >= 2:
            arr[5, 5] += 80
            arr[5, 6] += 60
        np.save(frames / f"frame_{index:06d}.npy", arr)
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
            "--template-strategy",
            "rolling-previous",
            "--no-local-shape-check",
            "--no-temporal-check",
            "--keep-all-residual-candidates",
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["template_strategy"] == "rolling-previous"
    assert manifest["template_window"] == [0, 1]

    summary_rows = list(csv.DictReader((out / "streaming_sum_summary.csv").open()))
    assert [
        (
            row["frame_start"],
            row["frame_end"],
            row["template_frame_start"],
            row["template_frame_end"],
            row["final_candidates"],
        )
        for row in summary_rows
    ] == [
        ("2", "3", "0", "1", "1"),
        ("4", "5", "2", "3", "0"),
    ]


def test_temporal_support_uses_first_template_window_as_low_cache_baseline(tmp_path):
    mod = load_module()
    run = tmp_path / "run"
    frames = run / "frames"
    frames.mkdir(parents=True)
    for index in range(6):
        arr = np.zeros((15, 15), dtype=np.uint16)
        arr[7, 7] = 1000
        if index == 2:
            arr[7, 7] += 500
            arr[7, 8] += 300
            arr[8, 7] += 200
        np.save(frames / f"frame_{index:06d}.npy", arr)
    paths = mod.frame_paths_from_run(run)
    cfg = mod.ScreenerConfig(
        window_size=2,
        temporal_cut_half=2,
        temporal_aperture_radius=1.5,
        temporal_annulus_r_in=2.0,
        temporal_annulus_r_out=2.8,
        temporal_sigma=3.0,
    )

    temporal = mod.measure_temporal_support(
        paths,
        paths,
        0,
        2,
        2,
        6,
        7,
        7,
        cfg,
    )

    assert json.loads(temporal["temporal_flux_series"]) == [1000, 0, 0, 0]
    assert temporal["temporal_active_frames"] == 1
    assert temporal["likely_cosmic_ray"] == 1


def test_temporal_support_converts_detector_xy_for_binned_frames(tmp_path):
    mod = load_module()
    run = tmp_path / "run"
    frames = run / "frames"
    frames.mkdir(parents=True)
    for index in range(6):
        arr = np.zeros((12, 12), dtype=np.uint16)
        if index == 2:
            arr[4:6, 4:6] = 10
        np.save(frames / f"frame_{index:06d}.npy", arr)
    paths = mod.frame_paths_from_run(run)
    cfg = mod.ScreenerConfig(
        spatial_bin=mod.SpatialBin(2, 2),
        window_size=2,
        temporal_cut_half=2,
        temporal_aperture_radius=0.1,
        temporal_annulus_r_in=1.0,
        temporal_annulus_r_out=2.0,
        temporal_sigma=3.0,
    )
    x, y = mod.detector_center_from_bin(2, 2, cfg.spatial_bin)

    temporal = mod.measure_temporal_support(
        paths,
        paths,
        0,
        2,
        2,
        6,
        x,
        y,
        cfg,
    )

    assert json.loads(temporal["temporal_flux_series"]) == [40, 0, 0, 0]
    assert temporal["temporal_active_frames"] == 1
    assert temporal["likely_cosmic_ray"] == 1


def test_previous_block_state_marks_match_without_being_a_hard_requirement():
    mod = load_module()
    previous = [
        {
            "x": 10.0,
            "y": 20.0,
            "frame_start": 12,
            "frame_end": 23,
            "track_length": 2,
        }
    ]
    current = [
        {"x": 11.0, "y": 21.0, "frame_start": 24, "frame_end": 35},
        {"x": 90.0, "y": 90.0, "frame_start": 24, "frame_end": 35},
    ]

    annotated = mod.annotate_previous_block_matches(current, previous, radius_px=2.0)

    assert annotated[0]["previous_block_match_flag"] == 1
    assert annotated[0]["track_length"] == 3
    assert annotated[1]["previous_block_match_flag"] == 0
    assert annotated[1]["track_length"] == 1


def test_temporal_peak_maps_flag_single_frame_positive_residual():
    mod = load_module()
    cfg = mod.ScreenerConfig(cosmic_single_frame_fraction=0.8, cosmic_max_active_frames=1)
    active = np.zeros((5, 5), dtype=np.uint8)
    total = np.zeros((5, 5), dtype=np.int64)
    peak = np.zeros((5, 5), dtype=np.int64)
    active[2, 3] = 1
    total[2, 3] = 1000
    peak[2, 3] = 1000

    temporal = mod.measure_temporal_support_from_peak_maps(active, total, peak, 3, 2, cfg)

    assert temporal["temporal_active_frames"] == 1
    assert temporal["temporal_max_single_frame_fraction"] == 1.0
    assert temporal["likely_cosmic_ray"] == 1
    assert temporal["temporal_flux_series"] == "[]"


def test_residual_prefilter_rejects_weak_or_too_spiky_candidates():
    mod = load_module()
    cfg = mod.ScreenerConfig(min_residual_peak_value=10000, min_flux_peak_ratio=3.0)

    weak = {"residual_peak_value": 9000, "residual_flux": 90000}
    spiky = {"residual_peak_value": 20000, "residual_flux": 40000}
    psf_like = {"residual_peak_value": 20000, "residual_flux": 80000}

    assert mod.passes_residual_prefilter(weak, cfg) is False
    assert mod.passes_residual_prefilter(spiky, cfg) is False
    assert mod.passes_residual_prefilter(psf_like, cfg) is True


def test_final_gate_passes_prefiltered_candidate_when_peak_snr_check_is_disabled():
    mod = load_module()
    cfg = mod.ScreenerConfig(
        effective_npix_threshold=99,
        peak_pixel_snr_check=False,
        temporal_check=False,
    )

    assert mod.passes_final_candidate_gate(
        {"peak_npix": 1, "peak_pixel_snr": 0.0, "temporal_active_frames": 0},
        cfg,
    ) is True


def test_peak_snr_final_gate_requires_local_area_and_snr_when_enabled():
    mod = load_module()
    cfg = mod.ScreenerConfig(
        effective_npix_threshold=4,
        peak_pixel_snr_check=True,
        peak_pixel_snr_threshold=5.0,
        temporal_check=False,
    )

    assert mod.passes_final_candidate_gate({"peak_npix": 3, "peak_pixel_snr": 10.0}, cfg) is False
    assert mod.passes_final_candidate_gate({"peak_npix": 4, "peak_pixel_snr": 4.9}, cfg) is False
    assert mod.passes_final_candidate_gate({"peak_npix": 4, "peak_pixel_snr": 5.0}, cfg) is True


def test_window_budget_keeps_highest_flux_final_candidates():
    mod = load_module()
    cfg = mod.ScreenerConfig(max_final_candidates_per_window=2)
    rows = [
        {"pass_single_stack": 1, "residual_flux": 10},
        {"pass_single_stack": 1, "residual_flux": 30},
        {"pass_single_stack": 0, "residual_flux": 100},
        {"pass_single_stack": 1, "residual_flux": 20},
    ]

    selected = mod.select_final_rows_for_window(rows, cfg)

    assert [row["residual_flux"] for row in selected] == [30, 20]


def test_spatial_bin_coordinate_mapping_for_square_and_rectangular_bins():
    mod = load_module()
    for bin_size in [2, 3]:
        spatial_bin = mod.SpatialBin(bin_size, bin_size)
        assert mod.binned_detection_shape((9120, 8900), spatial_bin) == (
            9120 // bin_size,
            8900 // bin_size,
        )
        x, y = mod.detector_center_from_bin(4, 5, spatial_bin)
        assert x == 4 * bin_size + (bin_size - 1) / 2.0
        assert y == 5 * bin_size + (bin_size - 1) / 2.0

    rectangular = mod.SpatialBin(3, 4)
    assert mod.binned_detection_shape((9120, 8900), rectangular) == (3040, 2225)
    assert mod.detector_center_from_bin(4, 5, rectangular) == (17.5, 16.0)


def test_short_script_wrappers_expose_spatial_bin_help():
    for script in [
        "scripts/grbfind.py",
        "scripts/grbfind-bin2.py",
        "scripts/grbfind-bin3.py",
    ]:
        proc = subprocess.run(
            ["python", script, "--help"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "--spatial-bin" in proc.stdout


def test_short_script_wrappers_list_complete_adjustable_defaults():
    scripts = [
        ("scripts/grbfind.py", "1x1", ""),
        ("scripts/grbfind-bin2.py", "2x2", "_bin2"),
        ("scripts/grbfind-bin3.py", "3x3", "_bin3"),
    ]

    for script, spatial_bin, output_suffix in scripts:
        module = load_script_module(script)
        defaults = module.SCRIPT_DEFAULTS

        assert set(defaults) == EXPECTED_SCRIPT_DEFAULT_KEYS
        assert defaults["spatial_bin"] == spatial_bin
        assert str(defaults["output_dir"]).endswith(output_suffix)
        assert defaults["max_windows"] is None
        assert defaults["template_strategy"] == "rolling-previous"
        assert defaults["local_shape_check"] is False
        assert defaults["peak_pixel_snr_check"] is False
        assert defaults["peak_pixel_snr_threshold"] == 5.0


def test_short_script_wrappers_default_to_rolling_previous_template_strategy(tmp_path):
    run = make_run(tmp_path, n_frames=6, shape=(12, 12))
    scripts = [
        ("scripts/grbfind.py", 1, 1),
        ("scripts/grbfind-bin2.py", 2, 2),
        ("scripts/grbfind-bin3.py", 3, 3),
    ]

    for script, bin_rows, bin_cols in scripts:
        out = tmp_path / script.replace("/", "_").replace(".py", "")
        subprocess.run(
            [
                "python",
                script,
                "--input-run",
                str(run),
                "--output-dir",
                str(out),
                "--window-size",
                "2",
                "--stride",
                "2",
                "--no-local-shape-check",
                "--no-temporal-check",
                "--keep-all-residual-candidates",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["template_strategy"] == "rolling-previous"
        assert manifest["spatial_bin_rows"] == bin_rows
        assert manifest["spatial_bin_cols"] == bin_cols
        assert manifest["all_windows_including_template"] == 3
        assert manifest["windows_processed"] == 2

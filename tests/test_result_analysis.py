import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.grbfinder_analysis.core import AnalysisRequest, run_analysis


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_analysis_fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_run = tmp_path / "input_run"
    frames = input_run / "frames"
    frames.mkdir(parents=True)
    for index in range(4):
        frame = np.zeros((40, 40), dtype=np.uint16)
        if index in (0, 1):
            frame[3, 4] = 500
        np.save(frames / f"frame_{index:06d}.npy", frame)

    events_csv = input_run / "events.csv"
    write_csv(
        events_csv,
        [
            "event_id",
            "first_visible_frame",
            "last_visible_frame",
            "requested_detector_xpix",
            "requested_detector_ypix",
            "stamp_center_col",
            "stamp_center_row",
            "peak_global_frame",
            "edge_clipped",
        ],
        [
            {
                "event_id": "1",
                "first_visible_frame": "2",
                "last_visible_frame": "3",
                "requested_detector_xpix": "8.0",
                "requested_detector_ypix": "8.0",
                "stamp_center_col": "8",
                "stamp_center_row": "8",
                "peak_global_frame": "3",
                "edge_clipped": "False",
            },
            {
                "event_id": "2",
                "first_visible_frame": "2",
                "last_visible_frame": "3",
                "requested_detector_xpix": "39.0",
                "requested_detector_ypix": "39.0",
                "stamp_center_col": "39",
                "stamp_center_row": "39",
                "peak_global_frame": "3",
                "edge_clipped": "True",
            },
        ],
    )

    cosmic_dir = input_run / "copied_source_aux" / "cosmic_events"
    cosmic_dir.mkdir(parents=True)
    dtype = np.dtype(
        [
            ("frame_index", "<i8"),
            ("local_frame_index", "<i8"),
            ("stamp_index", "<i8"),
            ("library_label", "<i8"),
            ("y0", "<i8"),
            ("x0", "<i8"),
            ("y1", "<i8"),
            ("x1", "<i8"),
            ("stamp_y0", "<i8"),
            ("stamp_x0", "<i8"),
            ("stamp_height", "<i8"),
            ("stamp_width", "<i8"),
            ("clipped_by_frame", "?"),
            ("total_adu", "<f8"),
            ("peak_adu", "<f8"),
            ("library_total_adu", "<f8"),
            ("library_peak_adu", "<f8"),
        ]
    )
    np.save(cosmic_dir / "frame_000002_events.npy", np.array([], dtype=dtype))
    cosmic_event = np.array(
        [(3, 1, 10, 99, 19, 19, 22, 22, 0, 0, 3, 3, False, 1000.0, 300.0, 1000.0, 300.0)],
        dtype=dtype,
    )
    np.save(cosmic_dir / "frame_000003_events.npy", cosmic_event)

    result_dir = tmp_path / "result"
    result_dir.mkdir()
    manifest = {
        "input_run": str(input_run),
        "truth_events_csv": str(events_csv),
        "template_strategy": "rolling-previous",
        "spatial_bin_rows": 1,
        "spatial_bin_cols": 1,
        "windows_processed": 1,
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest))
    write_csv(
        result_dir / "streaming_sum_summary.csv",
        [
            "frame_start",
            "frame_end",
            "window_frame_count",
            "template_frame_start",
            "template_frame_end",
            "template_source_count",
            "initial_sources",
            "after_residual_prefilter",
            "template_matched_sources_kept",
            "previous_block_matches",
            "after_measurement",
            "final_candidates",
        ],
        [
            {
                "frame_start": "2",
                "frame_end": "3",
                "window_frame_count": "2",
                "template_frame_start": "0",
                "template_frame_end": "1",
                "template_source_count": "0",
                "initial_sources": "4",
                "after_residual_prefilter": "4",
                "template_matched_sources_kept": "0",
                "previous_block_matches": "0",
                "after_measurement": "4",
                "final_candidates": "4",
            }
        ],
    )
    candidate_fields = [
        "frame_start",
        "frame_end",
        "window_frame_count",
        "x",
        "y",
        "bin_x",
        "bin_y",
        "source_peak_value",
        "nearest_template_dist_px",
        "nearest_template_x",
        "nearest_template_y",
        "template_match_flag",
        "candidate_channel",
        "residual_peak_value",
        "residual_flux",
        "peak_pixel_snr",
        "temporal_active_frames",
        "temporal_max_single_frame_fraction",
        "likely_cosmic_ray",
        "truth_match_flag",
        "truth_event_id",
        "truth_dist_px",
        "temporal_flux_series",
    ]
    rows = [
        {
            "frame_start": "2",
            "frame_end": "3",
            "window_frame_count": "2",
            "x": "20",
            "y": "20",
            "bin_x": "20",
            "bin_y": "20",
            "source_peak_value": "120",
            "nearest_template_dist_px": "nan",
            "nearest_template_x": "nan",
            "nearest_template_y": "nan",
            "template_match_flag": "0",
            "candidate_channel": "new_source",
            "residual_peak_value": "120",
            "residual_flux": "200",
            "peak_pixel_snr": "70",
            "temporal_active_frames": "1",
            "temporal_max_single_frame_fraction": "0.95",
            "likely_cosmic_ray": "1",
            "truth_match_flag": "0",
            "truth_event_id": "",
            "truth_dist_px": "nan",
            "temporal_flux_series": "[10, 200]",
        },
        {
            "frame_start": "2",
            "frame_end": "3",
            "window_frame_count": "2",
            "x": "8",
            "y": "8",
            "bin_x": "8",
            "bin_y": "8",
            "source_peak_value": "100",
            "nearest_template_dist_px": "nan",
            "nearest_template_x": "nan",
            "nearest_template_y": "nan",
            "template_match_flag": "0",
            "candidate_channel": "new_source",
            "residual_peak_value": "100",
            "residual_flux": "500",
            "peak_pixel_snr": "50",
            "temporal_active_frames": "2",
            "temporal_max_single_frame_fraction": "0.55",
            "likely_cosmic_ray": "0",
            "truth_match_flag": "1",
            "truth_event_id": "1",
            "truth_dist_px": "0.0",
            "temporal_flux_series": "[120, 130]",
        },
        {
            "frame_start": "2",
            "frame_end": "3",
            "window_frame_count": "2",
            "x": "4",
            "y": "3",
            "bin_x": "4",
            "bin_y": "3",
            "source_peak_value": "70",
            "nearest_template_dist_px": "nan",
            "nearest_template_x": "nan",
            "nearest_template_y": "nan",
            "template_match_flag": "0",
            "candidate_channel": "new_source",
            "residual_peak_value": "70",
            "residual_flux": "300",
            "peak_pixel_snr": "20",
            "temporal_active_frames": "2",
            "temporal_max_single_frame_fraction": "0.60",
            "likely_cosmic_ray": "0",
            "truth_match_flag": "0",
            "truth_event_id": "",
            "truth_dist_px": "nan",
            "temporal_flux_series": "[40, 45]",
        },
        {
            "frame_start": "2",
            "frame_end": "3",
            "window_frame_count": "2",
            "x": "30",
            "y": "30",
            "bin_x": "30",
            "bin_y": "30",
            "source_peak_value": "10",
            "nearest_template_dist_px": "nan",
            "nearest_template_x": "nan",
            "nearest_template_y": "nan",
            "template_match_flag": "0",
            "candidate_channel": "new_source",
            "residual_peak_value": "10",
            "residual_flux": "10",
            "peak_pixel_snr": "5",
            "temporal_active_frames": "2",
            "temporal_max_single_frame_fraction": "0.50",
            "likely_cosmic_ray": "0",
            "truth_match_flag": "0",
            "truth_event_id": "",
            "truth_dist_px": "nan",
            "temporal_flux_series": "[6, 7]",
        },
    ]
    write_csv(result_dir / "streaming_sum_transient_candidates.csv", candidate_fields, rows)
    write_csv(result_dir / "streaming_sum_candidates_after_measurement.csv", candidate_fields, rows)
    return result_dir, input_run


def make_star_cache(input_run: Path, x: float = 4.0, y: float = 3.0) -> None:
    cache = input_run / "copied_source_aux" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache / "stars_test.npz",
        detector_xpix=np.asarray([x]),
        detector_ypix=np.asarray([y]),
        total_flux=np.asarray([1234.0]),
        source_id=np.asarray([42]),
    )


def test_run_analysis_writes_recall_ranks_and_false_positive_diagnostics(tmp_path):
    result_dir, _input_run = make_analysis_fixture(tmp_path)
    output_dir = result_dir / "analysis"

    result = run_analysis(AnalysisRequest(result_dir=result_dir, output_dir=output_dir, top_n=10))

    assert result.summary["truth_events_in_windows"] == 2
    assert result.summary["truth_events_detected"] == 1
    assert result.summary["truth_recall"] == 0.5
    assert result.summary["false_candidates"] == 3
    assert result.summary["false_cosmic_truth"] == 1
    assert result.summary["false_template_star_like"] == 1

    truth_rows = read_csv(output_dir / "truth_event_ranks.csv")
    hit = next(row for row in truth_rows if row["event_id"] == "1")
    assert hit["detected"] == "1"
    assert hit["snr_rank"] == "2"
    assert hit["residual_flux_rank"] == "1"
    assert hit["residual_peak_rank"] == "2"
    miss = next(row for row in truth_rows if row["event_id"] == "2")
    assert miss["detected"] == "0"
    assert miss["miss_reason"] == "edge_or_stamp_clipped"

    false_rows = read_csv(output_dir / "false_positive_diagnostics.csv")
    by_xy = {(row["x"], row["y"]): row for row in false_rows}
    assert by_xy[("20", "20")]["false_positive_class"] == "false_likely_cosmic"
    assert by_xy[("20", "20")]["cosmic_truth_match_flag"] == "1"
    assert by_xy[("20", "20")]["cosmic_truth_same_frame_flag"] == "1"
    assert by_xy[("4", "3")]["false_positive_class"] == "false_template_star_like"
    assert by_xy[("30", "30")]["false_positive_class"] == "false_unknown_residual"

    assert (output_dir / "analysis_summary.json").exists()
    assert (output_dir / "analysis_report.md").exists()
    assert (output_dir / "window_grb_recall.csv").exists()
    assert (output_dir / "false_positive_groups.csv").exists()
    assert (output_dir / "top_candidates_by_snr.csv").exists()
    assert (output_dir / "template_sources.csv").exists()
    report = (output_dir / "analysis_report.md").read_text()
    assert "snr_rank" in report
    assert "residual_flux_rank" in report
    assert "cosmic_truth_same_frame_flag" in report


def test_analysis_prefers_source_cache_for_template_catalog(tmp_path):
    result_dir, input_run = make_analysis_fixture(tmp_path)
    for path in (input_run / "frames").glob("frame_*.npy"):
        frame = np.zeros((40, 40), dtype=np.uint16)
        np.save(path, frame)
    make_star_cache(input_run)

    output_dir = result_dir / "analysis_cache"
    result = run_analysis(AnalysisRequest(result_dir=result_dir, output_dir=output_dir, top_n=10))

    assert result.summary["false_template_star_like"] == 1
    template_rows = read_csv(output_dir / "template_sources.csv")
    assert template_rows == [
        {
            "x": "4",
            "y": "3",
            "source_peak_value": "1234",
            "template_source_threshold": "",
            "template_bkg_median": "",
            "template_bkg_sigma": "",
            "source_id": "42",
            "catalog_source": "copied_source_aux_cache",
        }
    ]


def test_missed_truth_visible_in_template_window_gets_specific_reason(tmp_path):
    result_dir, input_run = make_analysis_fixture(tmp_path)
    events_csv = input_run / "events.csv"
    rows = read_csv(events_csv)
    rows.append(
        {
            "event_id": "3",
            "first_visible_frame": "1",
            "last_visible_frame": "3",
            "requested_detector_xpix": "38.0",
            "requested_detector_ypix": "1.0",
            "stamp_center_col": "38",
            "stamp_center_row": "1",
            "peak_global_frame": "2",
            "edge_clipped": "False",
        }
    )
    write_csv(events_csv, list(rows[0].keys()), rows)

    output_dir = result_dir / "analysis_template_overlap"
    run_analysis(AnalysisRequest(result_dir=result_dir, output_dir=output_dir, top_n=10))

    truth_rows = read_csv(output_dir / "truth_event_ranks.csv")
    template_overlap = next(row for row in truth_rows if row["event_id"] == "3")
    assert template_overlap["detected"] == "0"
    assert template_overlap["miss_reason"] == "present_in_template_window"


def test_cli_script_runs_from_repo_root(tmp_path):
    result_dir, _input_run = make_analysis_fixture(tmp_path)
    output_dir = result_dir / "analysis_cli"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/analyze_grbfinder_results.py",
            "--result-dir",
            str(result_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "analysis_summary.json").exists()

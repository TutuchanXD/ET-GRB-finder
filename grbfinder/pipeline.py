from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ScreenerConfig
from .detection import (
    annotate_template_matches,
    classify_candidate_priority,
    detect_residual_sources_on_tile,
    detect_sources_on_tile,
    passes_residual_prefilter,
    residual_flux_peak_ratio,
    select_final_rows_for_window,
)
from .geometry import (
    binned_detection_shape,
    candidate_local_xy,
    expand_tile,
    iter_detection_tiles,
)
from .io import (
    default_truth_path,
    frame_paths_from_run,
    frame_shape_and_dtype,
    save_csv,
    sum_frame_tile,
    validate_frame_range,
    write_csv_header,
)
from .measurement import (
    annotate_previous_block_matches,
    annotate_truth_matches,
    build_temporal_peak_maps_for_tile,
    load_truth_events,
    measure_candidate_local_excess,
    measure_temporal_support_from_peak_maps,
)


MEASURED_FIELDS = [
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
    "source_threshold",
    "source_bkg_median",
    "source_bkg_sigma",
    "residual_peak_value",
    "residual_flux",
    "residual_npix",
    "flux_peak_ratio",
    "residual_prefilter_pass",
    "local_excess_flux",
    "peak_npix",
    "peak_excess_value",
    "local_bkg_median",
    "local_bkg_sigma",
    "peak_pixel_snr",
    "peak_pixel_snr_check_enabled",
    "peak_pixel_snr_threshold",
    "effective_npix_threshold",
    "local_shape_check_enabled",
    "temporal_check_enabled",
    "temporal_active_frames",
    "temporal_consecutive_active_frames",
    "temporal_max_single_frame_fraction",
    "likely_cosmic_ray",
    "previous_block_match_check_enabled",
    "previous_block_match_flag",
    "previous_block_match_dist_px",
    "previous_block_match_frame_start",
    "previous_block_match_frame_end",
    "track_length",
    "candidate_priority",
    "pass_single_stack",
    "truth_match_flag",
    "truth_event_id",
    "truth_dist_px",
    "temporal_flux_series",
]

SUMMARY_FIELDS = [
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
]

TEMPLATE_SOURCE_FIELDS = [
    "x",
    "y",
    "source_peak_value",
    "template_source_threshold",
    "template_bkg_median",
    "template_bkg_sigma",
]


@dataclass(frozen=True)
class PipelineRequest:
    input_run: Path
    output_dir: Path
    template_run: Path | None = None
    truth_events_csv: Path | None = None
    truth_match_radius_px: float = 12.0
    max_windows: int | None = None
    template_match_sources: bool = False
    template_strategy: str = "first-window"
    overwrite: bool = False


def template_window_pairs(
    ranges: list[tuple[int, int]],
    *,
    paired_template: bool,
    template_strategy: str,
    seed_template_range: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if paired_template:
        return [(window, window) for window in ranges]
    if template_strategy == "first-window":
        seed_start, seed_end = seed_template_range
        return [
            (window, (seed_start, seed_end))
            for window in ranges
            if window[0] >= seed_end and window[1] > seed_end
        ]
    if template_strategy != "rolling-previous":
        raise ValueError(f"unknown template_strategy: {template_strategy}")

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    latest_prior = None
    prior_index = 0
    for window in ranges:
        frame_start, _frame_end = window
        while prior_index < len(ranges) and ranges[prior_index][1] <= frame_start:
            latest_prior = ranges[prior_index]
            prior_index += 1
        if latest_prior is not None:
            pairs.append((window, latest_prior))
    return pairs


def build_template_sources(
    frame_paths: list[Path],
    frame_start: int,
    frame_end: int,
    shape: tuple[int, int],
    cfg: ScreenerConfig,
) -> tuple[np.ndarray, list[dict]]:
    all_sources: list[dict] = []
    for row0, row1, col0, col1 in iter_detection_tiles(shape, cfg.tile_size, cfg.use_tiles):
        row_slice, col_slice = expand_tile(row0, row1, col0, col1, shape, cfg.halo)
        tile = sum_frame_tile(frame_paths, frame_start, frame_end, row_slice, col_slice, cfg.spatial_bin)
        sources, threshold, bkg_median, bkg_sigma = detect_sources_on_tile(
            tile,
            row_origin=row_slice.start,
            col_origin=col_slice.start,
            core_bounds=(row0, row1, col0, col1),
            cfg=cfg,
        )
        for source in sources:
            source.update(
                {
                    "template_source_threshold": threshold,
                    "template_bkg_median": bkg_median,
                    "template_bkg_sigma": bkg_sigma,
                }
            )
        all_sources.extend(sources)
    xy = np.asarray([(row["x"], row["y"]) for row in all_sources], dtype=float)
    return xy, all_sources


def passes_final_candidate_gate(row: dict, cfg: ScreenerConfig) -> bool:
    if cfg.keep_all_residual_candidates:
        return True
    if not cfg.peak_pixel_snr_check:
        return True

    peak_npix = int(row.get("peak_npix", 0))
    peak_pixel_snr = float(row.get("peak_pixel_snr", np.nan))
    return bool(
        peak_npix >= cfg.effective_npix_threshold
        and np.isfinite(peak_pixel_snr)
        and peak_pixel_snr >= cfg.peak_pixel_snr_threshold
    )


def scan_window(
    frame_paths: list[Path],
    template_frame_paths: list[Path],
    template_xy: np.ndarray,
    frame_start: int,
    frame_end: int,
    shape: tuple[int, int],
    cfg: ScreenerConfig,
    *,
    template_frame_start: int,
    template_frame_end: int,
) -> tuple[list[dict], dict]:
    measured: list[dict] = []
    initial_sources = 0
    prefiltered_sources = 0
    matched_template_sources = 0
    for row0, row1, col0, col1 in iter_detection_tiles(shape, cfg.tile_size, cfg.use_tiles):
        row_slice, col_slice = expand_tile(row0, row1, col0, col1, shape, cfg.halo)
        sum_img = sum_frame_tile(frame_paths, frame_start, frame_end, row_slice, col_slice, cfg.spatial_bin)
        template_sum_img = sum_frame_tile(
            template_frame_paths,
            template_frame_start,
            template_frame_end,
            row_slice,
            col_slice,
            cfg.spatial_bin,
        )
        temporal_maps = None
        if cfg.temporal_check:
            temporal_maps = build_temporal_peak_maps_for_tile(
                frame_paths,
                frame_start,
                frame_end,
                template_sum_img,
                template_frame_end - template_frame_start,
                row_slice,
                col_slice,
                cfg,
            )

        sources, source_threshold, source_bkg_median, source_bkg_sigma = detect_residual_sources_on_tile(
            sum_img,
            template_sum_img,
            row_origin=row_slice.start,
            col_origin=col_slice.start,
            core_bounds=(row0, row1, col0, col1),
            cfg=cfg,
        )
        initial_sources += len(sources)
        for source in sources:
            source["flux_peak_ratio"] = residual_flux_peak_ratio(source)
            source["residual_prefilter_pass"] = int(passes_residual_prefilter(source, cfg))
        sources = [source for source in sources if int(source["residual_prefilter_pass"]) == 1]
        prefiltered_sources += len(sources)
        annotated = annotate_template_matches(sources, template_xy, cfg.match_radius_px)
        for row in annotated:
            if row["template_match_flag"]:
                matched_template_sources += 1
            local_x, local_y = candidate_local_xy(row, row_slice=row_slice, col_slice=col_slice)
            if cfg.local_shape_check:
                local_flux, npix, peak_excess, local_bkg_med, local_bkg_sigma = measure_candidate_local_excess(
                    sum_img,
                    template_sum_img,
                    (local_x, local_y),
                    cfg,
                )
            else:
                local_flux = int(row.get("residual_flux", 0))
                npix = int(row.get("residual_npix", 0))
                peak_excess = int(row.get("residual_peak_value", 0))
                local_bkg_med = int(source_bkg_median)
                local_bkg_sigma = int(source_bkg_sigma)
            peak_pixel_snr = (
                float(peak_excess / (local_bkg_sigma + 1e-12))
                if local_bkg_sigma > 0
                else np.nan
            )
            if cfg.temporal_check and temporal_maps is not None and (local_flux > 0 or peak_excess > 0):
                temporal = measure_temporal_support_from_peak_maps(
                    temporal_maps[0],
                    temporal_maps[1],
                    temporal_maps[2],
                    local_x,
                    local_y,
                    cfg,
                )
            else:
                temporal = {
                    "temporal_active_frames": 0,
                    "temporal_consecutive_active_frames": 0,
                    "temporal_max_single_frame_fraction": 0.0,
                    "temporal_flux_series": "[]",
                    "likely_cosmic_ray": 0,
                }
            if cfg.keep_all_residual_candidates:
                pass_single_stack = 1
            else:
                pass_single_stack = int(
                    passes_final_candidate_gate(
                        {
                            "peak_npix": npix,
                            "peak_pixel_snr": peak_pixel_snr,
                        },
                        cfg,
                    )
                )
            rec = dict(row)
            rec.update(
                {
                    "frame_start": int(frame_start),
                    "frame_end": int(frame_end - 1),
                    "window_frame_count": int(frame_end - frame_start),
                    "source_threshold": int(source_threshold),
                    "source_bkg_median": int(source_bkg_median),
                    "source_bkg_sigma": int(source_bkg_sigma),
                    "residual_peak_value": int(row.get("residual_peak_value", 0)),
                    "residual_flux": int(row.get("residual_flux", 0)),
                    "residual_npix": int(row.get("residual_npix", 0)),
                    "flux_peak_ratio": float(row.get("flux_peak_ratio", residual_flux_peak_ratio(row))),
                    "residual_prefilter_pass": int(row.get("residual_prefilter_pass", 1)),
                    "local_excess_flux": int(local_flux),
                    "peak_npix": int(npix),
                    "peak_excess_value": int(peak_excess),
                    "local_bkg_median": int(local_bkg_med),
                    "local_bkg_sigma": int(local_bkg_sigma),
                    "peak_pixel_snr": peak_pixel_snr,
                    "peak_pixel_snr_check_enabled": int(cfg.peak_pixel_snr_check),
                    "peak_pixel_snr_threshold": float(cfg.peak_pixel_snr_threshold),
                    "effective_npix_threshold": int(cfg.effective_npix_threshold),
                    "local_shape_check_enabled": int(cfg.local_shape_check),
                    "temporal_check_enabled": int(cfg.temporal_check),
                    "pass_single_stack": pass_single_stack,
                    **temporal,
                }
            )
            measured.append(rec)
    summary = {
        "frame_start": int(frame_start),
        "frame_end": int(frame_end - 1),
        "window_frame_count": int(frame_end - frame_start),
        "template_frame_start": int(template_frame_start),
        "template_frame_end": int(template_frame_end - 1),
        "initial_sources": int(initial_sources),
        "after_residual_prefilter": int(prefiltered_sources),
        "template_matched_sources_kept": int(matched_template_sources),
        "after_measurement": int(len(measured)),
        "final_candidates": int(sum(row["pass_single_stack"] == 1 for row in measured)),
    }
    return measured, summary


def run_pipeline(request: PipelineRequest, cfg: ScreenerConfig) -> dict:
    if request.output_dir.exists() and not request.overwrite:
        raise FileExistsError(f"output directory already exists: {request.output_dir}")
    request.output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = frame_paths_from_run(request.input_run)
    template_frame_paths = frame_paths_from_run(request.template_run or request.input_run)
    input_shape, dtype = frame_shape_and_dtype(frame_paths)
    template_shape, _template_dtype = frame_shape_and_dtype(template_frame_paths)
    if template_shape != input_shape:
        raise ValueError(f"template shape {template_shape} does not match input shape {input_shape}")
    validate_frame_range(frame_paths, cfg.input_bit_depth)
    shape = binned_detection_shape(input_shape, cfg.spatial_bin)
    cropped_input_shape = [
        shape[0] * cfg.spatial_bin.rows,
        shape[1] * cfg.spatial_bin.cols,
    ]

    from .geometry import window_ranges

    ranges = window_ranges(len(frame_paths), cfg.window_size, cfg.stride)
    if request.max_windows is not None:
        ranges = ranges[: int(request.max_windows)]

    paired_template = request.template_run is not None
    template_frame_start = 0
    template_frame_end = min(cfg.window_size, len(template_frame_paths))
    detection_template_ranges = template_window_pairs(
        ranges,
        paired_template=paired_template,
        template_strategy=request.template_strategy,
        seed_template_range=(template_frame_start, template_frame_end),
    )
    template_xy = np.empty((0, 2), dtype=float)
    template_sources: list[dict] = []
    if request.template_match_sources and not paired_template and request.template_strategy == "first-window":
        template_xy, template_sources = build_template_sources(
            template_frame_paths,
            0,
            min(cfg.window_size, len(template_frame_paths)),
            shape,
            cfg,
        )

    truth_path = request.truth_events_csv or default_truth_path(request.input_run)
    truth_events = load_truth_events(truth_path)

    measured_count = 0
    final_count = 0
    truth_matched_final_count = 0
    previous_window_rows: list[dict] = []
    measured_handle, measured_writer = write_csv_header(
        request.output_dir / "streaming_sum_candidates_after_measurement.csv",
        MEASURED_FIELDS,
    )
    final_handle, final_writer = write_csv_header(
        request.output_dir / "streaming_sum_transient_candidates.csv",
        MEASURED_FIELDS,
    )
    summary_handle, summary_writer = write_csv_header(
        request.output_dir / "streaming_sum_summary.csv",
        SUMMARY_FIELDS,
    )
    try:
        for (frame_start, frame_end), (current_template_start, current_template_end) in detection_template_ranges:
            if request.template_match_sources and (
                paired_template or request.template_strategy == "rolling-previous"
            ):
                template_xy, template_sources = build_template_sources(
                    template_frame_paths,
                    current_template_start,
                    current_template_end,
                    shape,
                    cfg,
                )
            rows, summary = scan_window(
                frame_paths,
                template_frame_paths,
                template_xy,
                frame_start,
                frame_end,
                shape,
                cfg,
                template_frame_start=current_template_start,
                template_frame_end=current_template_end,
            )
            if cfg.previous_block_match_check:
                rows = annotate_previous_block_matches(
                    rows,
                    previous_window_rows,
                    cfg.previous_match_radius_px,
                )
            else:
                rows = annotate_previous_block_matches(rows, [], cfg.previous_match_radius_px)
            for row in rows:
                row["previous_block_match_check_enabled"] = int(cfg.previous_block_match_check)
            if cfg.previous_block_match_check and not cfg.keep_all_residual_candidates:
                for row in rows:
                    if int(row.get("previous_block_match_flag", 0)) == 1:
                        row["pass_single_stack"] = 1
            for row in rows:
                row["candidate_priority"] = classify_candidate_priority(row, cfg)
            rows = annotate_truth_matches(rows, truth_events, request.truth_match_radius_px)
            final_rows = select_final_rows_for_window(rows, cfg)

            summary["final_candidates"] = int(len(final_rows))
            summary["previous_block_matches"] = int(
                sum(row.get("previous_block_match_flag", 0) == 1 for row in rows)
            )
            summary["template_source_count"] = int(len(template_xy))
            for row in rows:
                measured_writer.writerow(row)
            for row in final_rows:
                final_writer.writerow(row)
            summary_writer.writerow(summary)

            measured_count += len(rows)
            final_count += len(final_rows)
            truth_matched_final_count += int(
                sum(row.get("truth_match_flag", 0) == 1 for row in final_rows)
            )
            previous_window_rows = (
                [
                    {
                        "x": row["x"],
                        "y": row["y"],
                        "frame_start": row["frame_start"],
                        "frame_end": row["frame_end"],
                        "track_length": row.get("track_length", 1),
                    }
                    for row in final_rows
                ]
                if cfg.previous_block_match_check
                else []
            )
    finally:
        measured_handle.close()
        final_handle.close()
        summary_handle.close()

    template_source_catalog_written = bool(request.template_match_sources)
    if template_source_catalog_written:
        save_csv(
            request.output_dir / "template_sources.csv",
            template_sources,
            TEMPLATE_SOURCE_FIELDS,
        )

    manifest = {
        "input_run": str(request.input_run),
        "template_run": str(request.template_run) if request.template_run else None,
        "output_dir": str(request.output_dir),
        "truth_events_csv": str(truth_path) if truth_path else None,
        "shape": list(input_shape),
        "input_shape": list(input_shape),
        "detection_shape": list(shape),
        "cropped_input_shape": cropped_input_shape,
        "spatial_bin_size": cfg.spatial_bin.legacy_size,
        "spatial_bin_rows": int(cfg.spatial_bin.rows),
        "spatial_bin_cols": int(cfg.spatial_bin.cols),
        "dtype": str(dtype),
        "template_strategy": "paired-template" if paired_template else request.template_strategy,
        "template_window": [int(template_frame_start), int(template_frame_end - 1)],
        "detection_starts_after_template": bool(not paired_template),
        "window_size": cfg.window_size,
        "stride": cfg.stride,
        "use_tiles": bool(cfg.use_tiles),
        "tile_size": cfg.tile_size,
        "halo": cfg.halo,
        "source_threshold_sigma": cfg.source_threshold_sigma,
        "residual_threshold_sigma": cfg.residual_threshold_sigma,
        "residual_min_npix": cfg.residual_min_npix,
        "residual_max_npix": cfg.residual_max_npix,
        "min_residual_peak_value": cfg.min_residual_peak_value,
        "min_residual_flux": cfg.min_residual_flux,
        "min_flux_peak_ratio": cfg.min_flux_peak_ratio,
        "max_final_candidates_per_window": cfg.max_final_candidates_per_window,
        "local_threshold_sigma": cfg.local_threshold_sigma,
        "match_radius_px": cfg.match_radius_px,
        "previous_block_match_check": bool(cfg.previous_block_match_check),
        "previous_match_radius_px": cfg.previous_match_radius_px,
        "effective_npix_threshold": cfg.effective_npix_threshold,
        "peak_pixel_snr_check": bool(cfg.peak_pixel_snr_check),
        "peak_pixel_snr_threshold": cfg.peak_pixel_snr_threshold,
        "template_match_sources": bool(request.template_match_sources),
        "template_source_catalog_written": template_source_catalog_written,
        "local_shape_check": bool(cfg.local_shape_check),
        "temporal_check": bool(cfg.temporal_check),
        "keep_all_residual_candidates": bool(cfg.keep_all_residual_candidates),
        "windows_processed": len(detection_template_ranges),
        "detection_windows_processed": len(detection_template_ranges),
        "all_windows_including_template": len(ranges),
        "candidates_after_measurement": int(measured_count),
        "final_candidates": int(final_count),
        "truth_matched_final_candidates": int(truth_matched_final_count),
        "notes": (
            "Template-matched sources are kept as template_source_brightening candidates. "
            "likely_cosmic_ray is an advisory flag, not a hard rejection."
        ),
    }
    (request.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

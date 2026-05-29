"""Compatibility facade for the historical single-script API."""

from __future__ import annotations

from .cli import config_from_args, main, parse_args, parse_spatial_bin
from .config import (
    DEFAULT_INPUT_RUN,
    DEFAULT_OUTPUT_DIR,
    LOCAL_DIFF_DTYPE,
    SUM_DTYPE,
    ScreenerConfig,
    SpatialBin,
)
from .detection import (
    annotate_template_matches,
    classify_candidate_priority,
    detect_residual_sources_on_tile,
    detect_sources_on_tile,
    integer_median,
    passes_residual_prefilter,
    pick_component_label,
    residual_flux_peak_ratio,
    robust_integer_sigma,
    select_final_rows_for_window,
)
from .geometry import (
    binned_detection_shape,
    candidate_local_xy,
    detection_cutout_from_frame,
    detection_tile_from_frame,
    detector_center_from_bin,
    expand_tile,
    extract_cutout,
    iter_core_tiles,
    window_ranges,
)
from .io import (
    default_truth_path,
    frame_paths_from_run,
    frame_shape_and_dtype,
    input_range_from_bit_depth,
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
    measure_temporal_support,
    measure_temporal_support_from_peak_maps,
)
from .pipeline import (
    MEASURED_FIELDS,
    SUMMARY_FIELDS,
    TEMPLATE_SOURCE_FIELDS,
    PipelineRequest,
    build_template_sources,
    run_pipeline,
    scan_window,
)


SPATIAL_BIN_SIZE = 1

__all__ = [name for name in globals() if not name.startswith("_")]

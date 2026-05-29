#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grbfinder.cli import main
from grbfinder.config import DEFAULT_INPUT_RUN, DEFAULT_OUTPUT_DIR


SCRIPT_DEFAULTS = {
    "input_run": DEFAULT_INPUT_RUN,
    "template_run": None,
    "output_dir": Path(str(DEFAULT_OUTPUT_DIR) + "_bin2"),
    "truth_events_csv": None,
    "truth_match_radius_px": 12.0,
    "spatial_bin": "2x2",
    "window_size": 12,
    "stride": 12,
    "max_windows": 2,
    "template_strategy": "rolling-previous",
    "tile_size": 1024,
    "halo": 12,
    "input_bit_depth": 16,
    "max_filter_size": 7,
    "source_threshold_sigma": 4.0,
    "residual_threshold_sigma": 3.0,
    "residual_min_npix": 12,
    "residual_max_npix": 400,
    "min_residual_peak_value": 250000,
    "min_residual_flux": 0,
    "min_flux_peak_ratio": 2.0,
    "max_final_candidates_per_window": 5000,
    "match_radius_px": 0.75,
    "cut_half": 9,
    "annulus_r_in": 6.0,
    "annulus_r_out": 10.0,
    "local_threshold_sigma": 3.0,
    "seed_radius": 1.5,
    "effective_npix_threshold": 4,
    "temporal_cut_half": 5,
    "temporal_aperture_radius": 3.0,
    "temporal_annulus_r_in": 5.0,
    "temporal_annulus_r_out": 8.0,
    "temporal_sigma": 3.0,
    "temporal_min_active_frames": 2,
    "cosmic_single_frame_fraction": 0.80,
    "cosmic_max_active_frames": 1,
    "previous_match_radius_px": 2.0,
    "template_match_sources": False,
    "local_shape_check": True,
    "temporal_check": False,
    "keep_all_residual_candidates": False,
    "overwrite": False,
}


if __name__ == "__main__":
    raise SystemExit(main(script_defaults=SCRIPT_DEFAULTS))

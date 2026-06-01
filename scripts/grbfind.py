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
    # Input, output, and validation.
    "input_run": DEFAULT_INPUT_RUN,
    "template_run": None,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "overwrite": False,
    "truth_events_csv": None,
    "truth_match_radius_px": 12.0,

    # Window and template schedule.
    "window_size": 12,
    "stride": 12,
    "max_windows": None,
    "template_strategy": "rolling-previous",

    # Full-frame streaming and optional spatial binning.
    "spatial_bin": "1x1",
    "tile_size": 1024,
    "halo": 12,
    "input_bit_depth": 16,

    # Optional template-source annotation.
    "template_match_sources": False,
    "max_filter_size": 7,
    "source_threshold_sigma": 4.0,
    "match_radius_px": 0.75,

    # Residual-first connected-component detection and prefiltering.
    "residual_threshold_sigma": 3.0,
    "residual_min_npix": 50,
    "residual_max_npix": 400,
    "min_residual_peak_value": 50000,
    "min_residual_flux": 0,
    "min_flux_peak_ratio": 3.0,

    # Local residual cutout measurement.
    "local_shape_check": False,
    "cut_half": 9,
    "annulus_r_in": 6.0,
    "annulus_r_out": 10.0,
    "local_threshold_sigma": 3.0,
    "seed_radius": 1.5,

    # Optional local peak SNR gate.
    "peak_pixel_snr_check": False,
    "effective_npix_threshold": 4,
    "peak_pixel_snr_threshold": 5.0,

    # Optional temporal diagnostics.
    "temporal_check": False,
    "temporal_cut_half": 5,
    "temporal_aperture_radius": 3.0,
    "temporal_annulus_r_in": 5.0,
    "temporal_annulus_r_out": 8.0,
    "temporal_sigma": 3.0,
    "temporal_min_active_frames": 2,
    "cosmic_single_frame_fraction": 0.80,
    "cosmic_max_active_frames": 1,

    # Cross-window association and final output policy.
    "previous_match_radius_px": 2.0,
    "keep_all_residual_candidates": False,
    "max_final_candidates_per_window": 5000,
}


if __name__ == "__main__":
    raise SystemExit(main(script_defaults=SCRIPT_DEFAULTS))

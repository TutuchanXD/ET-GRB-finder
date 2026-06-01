from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import DEFAULT_INPUT_RUN, DEFAULT_OUTPUT_DIR, ScreenerConfig, SpatialBin
from .pipeline import PipelineRequest, run_pipeline


def parse_spatial_bin(value: str | int | SpatialBin) -> SpatialBin:
    if isinstance(value, SpatialBin):
        return value
    text = str(value).strip().lower()
    try:
        if "x" in text:
            parts = text.split("x")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError
            return SpatialBin(int(parts[0]), int(parts[1]))
        return SpatialBin.square(int(text))
    except ValueError as exc:
        raise ValueError(f"invalid spatial bin: {value!r}") from exc


def default_output_dir(default_output_suffix: str = "") -> Path:
    return Path(str(DEFAULT_OUTPUT_DIR) + default_output_suffix)


def script_default(script_defaults: dict[str, Any] | None, name: str, fallback: Any) -> Any:
    if script_defaults is not None and name in script_defaults:
        return script_defaults[name]
    return fallback


def build_parser(
    *,
    default_spatial_bin: str | SpatialBin = "1x1",
    default_output_suffix: str = "",
    default_template_strategy: str = "first-window",
    script_defaults: dict[str, Any] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Streaming 12-frame sum transient screener for current full-frame GRB products."
    )
    parser.add_argument("--input-run", type=Path, default=script_default(script_defaults, "input_run", DEFAULT_INPUT_RUN))
    parser.add_argument("--template-run", type=Path, default=script_default(script_defaults, "template_run", None))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_default(script_defaults, "output_dir", default_output_dir(default_output_suffix)),
    )
    parser.add_argument(
        "--truth-events-csv",
        type=Path,
        default=script_default(script_defaults, "truth_events_csv", None),
    )
    parser.add_argument(
        "--truth-match-radius-px",
        type=float,
        default=script_default(script_defaults, "truth_match_radius_px", 12.0),
    )
    parser.add_argument(
        "--spatial-bin",
        type=parse_spatial_bin,
        default=parse_spatial_bin(script_default(script_defaults, "spatial_bin", default_spatial_bin)),
        help="Spatial block binning as N or RxC, for example 3 or 3x4.",
    )
    parser.add_argument("--window-size", type=int, default=script_default(script_defaults, "window_size", 12))
    parser.add_argument("--stride", type=int, default=script_default(script_defaults, "stride", 12))
    parser.add_argument(
        "--template-strategy",
        choices=("first-window", "rolling-previous"),
        default=script_default(script_defaults, "template_strategy", default_template_strategy),
        help="Input-run template strategy when --template-run is not supplied.",
    )
    parser.add_argument(
        "--use-tiles",
        action=argparse.BooleanOptionalAction,
        default=script_default(script_defaults, "use_tiles", True),
        help="Process each detection window in tiles; disable to run full-frame detection per window.",
    )
    parser.add_argument("--tile-size", type=int, default=script_default(script_defaults, "tile_size", 1024))
    parser.add_argument("--halo", type=int, default=script_default(script_defaults, "halo", 12))
    parser.add_argument(
        "--input-bit-depth",
        type=int,
        default=script_default(script_defaults, "input_bit_depth", 16),
    )
    parser.add_argument(
        "--max-filter-size",
        type=int,
        default=script_default(script_defaults, "max_filter_size", 7),
    )
    parser.add_argument("--max-windows", type=int, default=script_default(script_defaults, "max_windows", None))
    parser.add_argument(
        "--source-threshold-sigma",
        type=float,
        default=script_default(script_defaults, "source_threshold_sigma", 4.0),
    )
    parser.add_argument(
        "--residual-threshold-sigma",
        type=float,
        default=script_default(script_defaults, "residual_threshold_sigma", 3.0),
    )
    parser.add_argument(
        "--residual-min-npix",
        type=int,
        default=script_default(script_defaults, "residual_min_npix", 2),
    )
    parser.add_argument(
        "--residual-max-npix",
        type=int,
        default=script_default(script_defaults, "residual_max_npix", 400),
    )
    parser.add_argument(
        "--min-residual-peak-value",
        type=int,
        default=script_default(script_defaults, "min_residual_peak_value", 10000),
    )
    parser.add_argument(
        "--min-residual-flux",
        type=int,
        default=script_default(script_defaults, "min_residual_flux", 0),
    )
    parser.add_argument(
        "--min-flux-peak-ratio",
        type=float,
        default=script_default(script_defaults, "min_flux_peak_ratio", 3.0),
    )
    parser.add_argument(
        "--max-final-candidates-per-window",
        type=int,
        default=script_default(script_defaults, "max_final_candidates_per_window", 5000),
    )
    parser.add_argument("--match-radius-px", type=float, default=script_default(script_defaults, "match_radius_px", 0.75))
    parser.add_argument("--cut-half", type=int, default=script_default(script_defaults, "cut_half", 9))
    parser.add_argument("--annulus-r-in", type=float, default=script_default(script_defaults, "annulus_r_in", 6.0))
    parser.add_argument("--annulus-r-out", type=float, default=script_default(script_defaults, "annulus_r_out", 10.0))
    parser.add_argument(
        "--local-threshold-sigma",
        type=float,
        default=script_default(script_defaults, "local_threshold_sigma", 3.0),
    )
    parser.add_argument("--seed-radius", type=float, default=script_default(script_defaults, "seed_radius", 1.5))
    parser.add_argument(
        "--effective-npix-threshold",
        type=int,
        default=script_default(script_defaults, "effective_npix_threshold", 4),
    )
    parser.add_argument(
        "--peak-pixel-snr-check",
        action=argparse.BooleanOptionalAction,
        default=script_default(script_defaults, "peak_pixel_snr_check", False),
        help="Allow strong local residual peak significance to pass final selection.",
    )
    parser.add_argument(
        "--peak-pixel-snr-threshold",
        type=float,
        default=script_default(script_defaults, "peak_pixel_snr_threshold", 5.0),
    )
    parser.add_argument(
        "--temporal-cut-half",
        type=int,
        default=script_default(script_defaults, "temporal_cut_half", 5),
    )
    parser.add_argument(
        "--temporal-aperture-radius",
        type=float,
        default=script_default(script_defaults, "temporal_aperture_radius", 3.0),
    )
    parser.add_argument(
        "--temporal-annulus-r-in",
        type=float,
        default=script_default(script_defaults, "temporal_annulus_r_in", 5.0),
    )
    parser.add_argument(
        "--temporal-annulus-r-out",
        type=float,
        default=script_default(script_defaults, "temporal_annulus_r_out", 8.0),
    )
    parser.add_argument("--temporal-sigma", type=float, default=script_default(script_defaults, "temporal_sigma", 3.0))
    parser.add_argument(
        "--temporal-min-active-frames",
        type=int,
        default=script_default(script_defaults, "temporal_min_active_frames", 2),
    )
    parser.add_argument(
        "--cosmic-single-frame-fraction",
        type=float,
        default=script_default(script_defaults, "cosmic_single_frame_fraction", 0.80),
    )
    parser.add_argument(
        "--cosmic-max-active-frames",
        type=int,
        default=script_default(script_defaults, "cosmic_max_active_frames", 1),
    )
    parser.add_argument(
        "--previous-block-match-check",
        action=argparse.BooleanOptionalAction,
        default=script_default(script_defaults, "previous_block_match_check", False),
        help="Associate candidates with previous-window final candidates and mark tracks.",
    )
    parser.add_argument(
        "--previous-match-radius-px",
        type=float,
        default=script_default(script_defaults, "previous_match_radius_px", 2.0),
    )
    parser.add_argument(
        "--template-match-sources",
        action=argparse.BooleanOptionalAction,
        default=script_default(script_defaults, "template_match_sources", False),
        help="Build and write the static template-source catalog for nearest-source annotation.",
    )
    parser.add_argument(
        "--local-shape-check",
        action=argparse.BooleanOptionalAction,
        default=script_default(script_defaults, "local_shape_check", True),
        help="Skip 19x19 local residual morphology measurement and use residual component fields directly.",
    )
    parser.add_argument(
        "--temporal-check",
        action=argparse.BooleanOptionalAction,
        default=script_default(script_defaults, "temporal_check", True),
        help="Skip per-frame temporal cutout measurement and cosmic-ray advisory fields.",
    )
    parser.add_argument(
        "--keep-all-residual-candidates",
        action=argparse.BooleanOptionalAction,
        default=script_default(script_defaults, "keep_all_residual_candidates", False),
        help="Bypass final morphology/time/SNR pass filter and output every residual candidate as final.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=script_default(script_defaults, "overwrite", False),
    )
    return parser


def parse_args(
    argv: list[str] | None = None,
    *,
    default_spatial_bin: str | SpatialBin = "1x1",
    default_output_suffix: str = "",
    default_template_strategy: str = "first-window",
    script_defaults: dict[str, Any] | None = None,
) -> argparse.Namespace:
    return build_parser(
        default_spatial_bin=default_spatial_bin,
        default_output_suffix=default_output_suffix,
        default_template_strategy=default_template_strategy,
        script_defaults=script_defaults,
    ).parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ScreenerConfig:
    return ScreenerConfig(
        spatial_bin=parse_spatial_bin(args.spatial_bin),
        window_size=args.window_size,
        stride=args.stride,
        use_tiles=args.use_tiles,
        tile_size=args.tile_size,
        halo=args.halo,
        input_bit_depth=args.input_bit_depth,
        max_filter_size=args.max_filter_size,
        source_threshold_sigma=args.source_threshold_sigma,
        residual_threshold_sigma=args.residual_threshold_sigma,
        residual_min_npix=args.residual_min_npix,
        residual_max_npix=args.residual_max_npix,
        min_residual_peak_value=args.min_residual_peak_value,
        min_residual_flux=args.min_residual_flux,
        min_flux_peak_ratio=args.min_flux_peak_ratio,
        max_final_candidates_per_window=args.max_final_candidates_per_window,
        match_radius_px=args.match_radius_px,
        cut_half=args.cut_half,
        annulus_r_in=args.annulus_r_in,
        annulus_r_out=args.annulus_r_out,
        local_threshold_sigma=args.local_threshold_sigma,
        seed_radius=args.seed_radius,
        previous_match_radius_px=args.previous_match_radius_px,
        effective_npix_threshold=args.effective_npix_threshold,
        peak_pixel_snr_check=args.peak_pixel_snr_check,
        peak_pixel_snr_threshold=args.peak_pixel_snr_threshold,
        temporal_cut_half=args.temporal_cut_half,
        temporal_aperture_radius=args.temporal_aperture_radius,
        temporal_annulus_r_in=args.temporal_annulus_r_in,
        temporal_annulus_r_out=args.temporal_annulus_r_out,
        temporal_sigma=args.temporal_sigma,
        temporal_min_active_frames=args.temporal_min_active_frames,
        cosmic_single_frame_fraction=args.cosmic_single_frame_fraction,
        cosmic_max_active_frames=args.cosmic_max_active_frames,
        previous_block_match_check=args.previous_block_match_check,
        local_shape_check=args.local_shape_check,
        temporal_check=args.temporal_check,
        keep_all_residual_candidates=args.keep_all_residual_candidates,
    )


def request_from_args(args: argparse.Namespace) -> PipelineRequest:
    return PipelineRequest(
        input_run=args.input_run,
        template_run=args.template_run,
        output_dir=args.output_dir,
        truth_events_csv=args.truth_events_csv,
        truth_match_radius_px=args.truth_match_radius_px,
        max_windows=args.max_windows,
        template_match_sources=args.template_match_sources,
        template_strategy=args.template_strategy,
        overwrite=args.overwrite,
    )


def main(
    argv: list[str] | None = None,
    *,
    default_spatial_bin: str | SpatialBin = "1x1",
    default_output_suffix: str = "",
    default_template_strategy: str = "first-window",
    script_defaults: dict[str, Any] | None = None,
) -> int:
    args = parse_args(
        argv,
        default_spatial_bin=default_spatial_bin,
        default_output_suffix=default_output_suffix,
        default_template_strategy=default_template_strategy,
        script_defaults=script_defaults,
    )
    cfg = config_from_args(args)
    manifest = run_pipeline(request_from_args(args), cfg)
    print(f"Input run: {args.input_run}")
    print(f"Template run: {args.template_run or f'({args.template_strategy})'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Windows processed: {manifest['windows_processed']}")
    print(f"Candidates after measurement: {manifest['candidates_after_measurement']}")
    print(f"Final candidates: {manifest['final_candidates']}")
    print(f"Truth-matched final candidates: {manifest['truth_matched_final_candidates']}")
    return 0

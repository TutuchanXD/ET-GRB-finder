from __future__ import annotations

import argparse
from pathlib import Path

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


def build_parser(
    *,
    default_spatial_bin: str | SpatialBin = "1x1",
    default_output_suffix: str = "",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Streaming 12-frame sum transient screener for current full-frame GRB products."
    )
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--template-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(default_output_suffix))
    parser.add_argument("--truth-events-csv", type=Path, default=None)
    parser.add_argument("--truth-match-radius-px", type=float, default=12.0)
    parser.add_argument(
        "--spatial-bin",
        default=str(parse_spatial_bin(default_spatial_bin)),
        help="Spatial block binning as N or RxC, for example 3 or 3x4.",
    )
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--halo", type=int, default=12)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--source-threshold-sigma", type=float, default=4.0)
    parser.add_argument("--residual-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--residual-min-npix", type=int, default=2)
    parser.add_argument("--residual-max-npix", type=int, default=400)
    parser.add_argument("--min-residual-peak-value", type=int, default=10000)
    parser.add_argument("--min-residual-flux", type=int, default=0)
    parser.add_argument("--min-flux-peak-ratio", type=float, default=3.0)
    parser.add_argument("--max-final-candidates-per-window", type=int, default=5000)
    parser.add_argument("--local-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=0.75)
    parser.add_argument("--previous-match-radius-px", type=float, default=2.0)
    parser.add_argument("--effective-npix-threshold", type=int, default=4)
    parser.add_argument(
        "--template-match-sources",
        action="store_true",
        help="Build and write the static template-source catalog for nearest-source annotation.",
    )
    parser.add_argument(
        "--no-local-shape-check",
        dest="local_shape_check",
        action="store_false",
        help="Skip 19x19 local residual morphology measurement and use residual component fields directly.",
    )
    parser.add_argument(
        "--no-temporal-check",
        dest="temporal_check",
        action="store_false",
        help="Skip per-frame temporal cutout measurement and cosmic-ray advisory fields.",
    )
    parser.add_argument(
        "--keep-all-residual-candidates",
        action="store_true",
        help="Bypass final morphology/time/SNR pass filter and output every residual candidate as final.",
    )
    parser.set_defaults(local_shape_check=True, temporal_check=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(
    argv: list[str] | None = None,
    *,
    default_spatial_bin: str | SpatialBin = "1x1",
    default_output_suffix: str = "",
) -> argparse.Namespace:
    return build_parser(
        default_spatial_bin=default_spatial_bin,
        default_output_suffix=default_output_suffix,
    ).parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ScreenerConfig:
    return ScreenerConfig(
        spatial_bin=parse_spatial_bin(args.spatial_bin),
        window_size=args.window_size,
        stride=args.stride,
        tile_size=args.tile_size,
        halo=args.halo,
        source_threshold_sigma=args.source_threshold_sigma,
        residual_threshold_sigma=args.residual_threshold_sigma,
        residual_min_npix=args.residual_min_npix,
        residual_max_npix=args.residual_max_npix,
        min_residual_peak_value=args.min_residual_peak_value,
        min_residual_flux=args.min_residual_flux,
        min_flux_peak_ratio=args.min_flux_peak_ratio,
        max_final_candidates_per_window=args.max_final_candidates_per_window,
        local_threshold_sigma=args.local_threshold_sigma,
        match_radius_px=args.match_radius_px,
        previous_match_radius_px=args.previous_match_radius_px,
        effective_npix_threshold=args.effective_npix_threshold,
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
        overwrite=args.overwrite,
    )


def main(
    argv: list[str] | None = None,
    *,
    default_spatial_bin: str | SpatialBin = "1x1",
    default_output_suffix: str = "",
) -> int:
    args = parse_args(
        argv,
        default_spatial_bin=default_spatial_bin,
        default_output_suffix=default_output_suffix,
    )
    cfg = config_from_args(args)
    manifest = run_pipeline(request_from_args(args), cfg)
    print(f"Input run: {args.input_run}")
    print(f"Template run: {args.template_run or '(first input window)'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Windows processed: {manifest['windows_processed']}")
    print(f"Candidates after measurement: {manifest['candidates_after_measurement']}")
    print(f"Final candidates: {manifest['final_candidates']}")
    print(f"Truth-matched final candidates: {manifest['truth_matched_final_candidates']}")
    return 0

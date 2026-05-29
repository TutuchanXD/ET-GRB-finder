#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.grbfinder_analysis.core import AnalysisRequest, run_analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze existing grbfinder search results.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--events-csv", type=Path, default=None)
    parser.add_argument("--input-run", type=Path, default=None)
    parser.add_argument("--template-sources-csv", type=Path, default=None)
    parser.add_argument("--cosmic-truth", type=Path, default=None)
    parser.add_argument("--cosmic-frame-summaries", type=Path, default=None)
    parser.add_argument("--truth-match-radius-px", type=float, default=12.0)
    parser.add_argument("--cosmic-match-radius-px", type=float, default=2.0)
    parser.add_argument("--star-match-radius-px", type=float, default=1.5)
    parser.add_argument("--star-source-threshold-sigma", type=float, default=4.0)
    parser.add_argument("--template-tile-size", type=int, default=1024)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_analysis(
        AnalysisRequest(
            result_dir=args.result_dir,
            output_dir=args.output_dir,
            events_csv=args.events_csv,
            input_run=args.input_run,
            template_sources_csv=args.template_sources_csv,
            cosmic_truth=args.cosmic_truth,
            cosmic_frame_summaries=args.cosmic_frame_summaries,
            truth_match_radius_px=args.truth_match_radius_px,
            cosmic_match_radius_px=args.cosmic_match_radius_px,
            star_match_radius_px=args.star_match_radius_px,
            star_source_threshold_sigma=args.star_source_threshold_sigma,
            template_tile_size=args.template_tile_size,
            top_n=args.top_n,
        )
    )
    print(f"Wrote analysis to {result.output_dir}")
    print(
        "truth_recall="
        f"{result.summary['truth_events_detected']}/{result.summary['truth_events_in_windows']} "
        f"false_candidates={result.summary['false_candidates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

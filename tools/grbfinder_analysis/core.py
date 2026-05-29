from __future__ import annotations

import ast
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AnalysisRequest:
    result_dir: Path
    output_dir: Path | None = None
    events_csv: Path | None = None
    input_run: Path | None = None
    template_sources_csv: Path | None = None
    cosmic_truth: Path | None = None
    cosmic_frame_summaries: Path | None = None
    truth_match_radius_px: float = 12.0
    cosmic_match_radius_px: float = 2.0
    star_match_radius_px: float = 1.5
    star_source_threshold_sigma: float = 4.0
    template_tile_size: int = 1024
    top_n: int = 50
    write_csv: bool = True


@dataclass(frozen=True)
class AnalysisResult:
    summary: dict
    output_dir: Path


def run_analysis(request: AnalysisRequest) -> AnalysisResult:
    result_dir = Path(request.result_dir)
    output_dir = Path(request.output_dir) if request.output_dir is not None else result_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(result_dir / "manifest.json")
    input_run = Path(request.input_run or manifest.get("input_run", ""))
    events_csv = Path(request.events_csv or manifest.get("truth_events_csv") or input_run / "events.csv")
    cosmic_dir = Path(request.cosmic_truth) if request.cosmic_truth else input_run / "copied_source_aux" / "cosmic_events"

    windows = read_csv_rows(result_dir / "streaming_sum_summary.csv")
    candidates = read_csv_rows(result_dir / "streaming_sum_transient_candidates.csv")
    measured_path = result_dir / "streaming_sum_candidates_after_measurement.csv"
    measured = read_csv_rows(measured_path) if measured_path.exists() else candidates
    truth_events = read_csv_rows(events_csv)

    add_candidate_ranks(candidates)
    template_sources = load_or_build_template_sources(
        request=request,
        input_run=input_run,
        output_dir=output_dir,
        windows=windows,
    )
    template_tree = build_template_tree(template_sources)

    cosmic_cache: dict[int, np.ndarray] = {}
    truth_rows, window_rows = evaluate_truth_events(
        windows=windows,
        candidates=candidates,
        measured=measured,
        truth_events=truth_events,
        truth_match_radius_px=request.truth_match_radius_px,
    )
    false_rows = diagnose_false_positives(
        candidates=candidates,
        cosmic_dir=cosmic_dir,
        cosmic_cache=cosmic_cache,
        cosmic_match_radius_px=request.cosmic_match_radius_px,
        template_sources=template_sources,
        template_tree=template_tree,
        star_match_radius_px=request.star_match_radius_px,
    )
    group_rows = group_false_positives(false_rows)
    top_rows = sorted_by_rank_metric(candidates, "peak_pixel_snr")[: request.top_n]

    summary = build_summary(
        result_dir=result_dir,
        input_run=input_run,
        manifest=manifest,
        windows=windows,
        candidates=candidates,
        truth_rows=truth_rows,
        false_rows=false_rows,
    )

    if request.write_csv:
        write_csv_rows(output_dir / "window_grb_recall.csv", window_rows)
        write_csv_rows(output_dir / "truth_event_ranks.csv", truth_rows)
        write_csv_rows(output_dir / "false_positive_diagnostics.csv", false_rows)
        write_csv_rows(output_dir / "false_positive_groups.csv", group_rows)
        write_csv_rows(output_dir / "top_candidates_by_snr.csv", top_rows)
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output_dir / "analysis_report.md").write_text(
        render_report(
            summary=summary,
            manifest=manifest,
            window_rows=window_rows,
            truth_rows=truth_rows,
            false_rows=false_rows,
            group_rows=group_rows,
        )
    )
    return AnalysisResult(summary=summary, output_dir=output_dir)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default: int = 0) -> int:
    number = to_float(value)
    if math.isnan(number):
        return default
    return int(number)


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return "nan"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.6g}"
    return str(value)


def block_key(row: dict) -> tuple[int, int]:
    return to_int(row.get("frame_start")), to_int(row.get("frame_end"))


def add_candidate_ranks(candidates: list[dict]) -> None:
    by_window: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in candidates:
        by_window[block_key(row)].append(row)
    for rows in by_window.values():
        assign_rank(rows, "peak_pixel_snr", "snr_rank", tie_breaker="residual_flux")
        assign_rank(rows, "residual_flux", "residual_flux_rank")
        assign_rank(rows, "residual_peak_value", "residual_peak_rank")


def assign_rank(rows: list[dict], metric: str, output_field: str, tie_breaker: str | None = None) -> None:
    sorted_rows = sorted_by_rank_metric(rows, metric, tie_breaker=tie_breaker)
    for rank, row in enumerate(sorted_rows, start=1):
        row[output_field] = str(rank)


def sorted_by_rank_metric(rows: list[dict], metric: str, tie_breaker: str | None = None) -> list[dict]:
    def key(row: dict) -> tuple[int, float, float]:
        value = to_float(row.get(metric))
        tie = to_float(row.get(tie_breaker)) if tie_breaker else 0.0
        if math.isnan(value):
            return (1, 0.0, 0.0)
        if math.isnan(tie):
            tie = 0.0
        return (0, -value, -tie)

    return sorted(rows, key=key)


def evaluate_truth_events(
    *,
    windows: list[dict],
    candidates: list[dict],
    measured: list[dict],
    truth_events: list[dict],
    truth_match_radius_px: float,
) -> tuple[list[dict], list[dict]]:
    final_by_window: dict[tuple[int, int], list[dict]] = defaultdict(list)
    measured_by_window: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in candidates:
        final_by_window[block_key(row)].append(row)
    for row in measured:
        measured_by_window[block_key(row)].append(row)

    truth_rows: list[dict] = []
    window_rows: list[dict] = []
    for window in windows:
        start, end = block_key(window)
        in_block = [event for event in truth_events if event_overlaps_window(event, start, end)]
        detected = 0
        for event in in_block:
            matched = find_truth_matched_candidate(final_by_window[(start, end)], event)
            if matched is None:
                matched = nearest_truth_candidate(final_by_window[(start, end)], event, truth_match_radius_px)
            is_detected = matched is not None
            detected += int(is_detected)
            visible_frames = visible_frames_in_window(event, start, end)
            row = {
                "event_id": str(event.get("event_id", "")),
                "frame_start": str(start),
                "frame_end": str(end),
                "first_visible_frame": event.get("first_visible_frame", ""),
                "last_visible_frame": event.get("last_visible_frame", ""),
                "visible_frames_in_block": str(visible_frames),
                "requested_detector_xpix": event.get("requested_detector_xpix", ""),
                "requested_detector_ypix": event.get("requested_detector_ypix", ""),
                "detected": str(int(is_detected)),
                "matched_candidate_x": matched.get("x", "") if matched else "",
                "matched_candidate_y": matched.get("y", "") if matched else "",
                "truth_dist_px": matched.get("truth_dist_px", "") if matched else "",
                "snr_rank": matched.get("snr_rank", "") if matched else "",
                "residual_flux_rank": matched.get("residual_flux_rank", "") if matched else "",
                "residual_peak_rank": matched.get("residual_peak_rank", "") if matched else "",
                "peak_pixel_snr": matched.get("peak_pixel_snr", "") if matched else "",
                "residual_flux": matched.get("residual_flux", "") if matched else "",
                "residual_peak_value": matched.get("residual_peak_value", "") if matched else "",
                "temporal_active_frames": matched.get("temporal_active_frames", "") if matched else "",
                "likely_cosmic_ray": matched.get("likely_cosmic_ray", "") if matched else "",
                "miss_reason": "" if is_detected else classify_miss_reason(
                    event=event,
                    window_start=start,
                    window_end=end,
                    template_start=to_int(window.get("template_frame_start"), 10**12),
                    template_end=to_int(window.get("template_frame_end"), -1),
                    final_rows=final_by_window[(start, end)],
                    measured_rows=measured_by_window[(start, end)],
                    truth_match_radius_px=truth_match_radius_px,
                ),
            }
            truth_rows.append(row)
        final_count = len(final_by_window[(start, end)])
        window_rows.append(
            {
                "frame_start": str(start),
                "frame_end": str(end),
                "template_frame_start": window.get("template_frame_start", ""),
                "template_frame_end": window.get("template_frame_end", ""),
                "truth_events_in_block": str(len(in_block)),
                "truth_events_detected": str(detected),
                "truth_recall": fmt(detected / len(in_block) if in_block else 0.0),
                "final_candidates": str(final_count),
                "false_candidates": str(max(final_count - detected, 0)),
            }
        )
    return truth_rows, window_rows


def event_overlaps_window(event: dict, start: int, end: int) -> bool:
    return to_int(event.get("first_visible_frame"), 10**12) <= end and to_int(event.get("last_visible_frame"), -1) >= start


def visible_frames_in_window(event: dict, start: int, end: int) -> int:
    overlap_start = max(to_int(event.get("first_visible_frame")), start)
    overlap_end = min(to_int(event.get("last_visible_frame")), end)
    return max(0, overlap_end - overlap_start + 1)


def find_truth_matched_candidate(candidates: list[dict], event: dict) -> dict | None:
    event_id = str(event.get("event_id", ""))
    for row in candidates:
        if truthy(row.get("truth_match_flag")) and str(row.get("truth_event_id", "")) == event_id:
            return row
    return None


def nearest_truth_candidate(candidates: list[dict], event: dict, radius: float) -> dict | None:
    event_x = to_float(event.get("requested_detector_xpix"))
    event_y = to_float(event.get("requested_detector_ypix"))
    best_row = None
    best_dist = math.inf
    for row in candidates:
        dist = distance(to_float(row.get("x")), to_float(row.get("y")), event_x, event_y)
        if dist <= radius and dist < best_dist:
            best_row = row
            best_dist = dist
    if best_row is not None and not best_row.get("truth_dist_px"):
        best_row["truth_dist_px"] = fmt(best_dist)
    return best_row


def classify_miss_reason(
    *,
    event: dict,
    window_start: int,
    window_end: int,
    template_start: int,
    template_end: int,
    final_rows: list[dict],
    measured_rows: list[dict],
    truth_match_radius_px: float,
) -> str:
    if not event_overlaps_window(event, window_start, window_end):
        return "outside_detection_window"
    if truthy(event.get("edge_clipped")):
        return "edge_or_stamp_clipped"
    if event_overlaps_window(event, template_start, template_end):
        return "present_in_template_window"
    if visible_frames_in_window(event, window_start, window_end) < 2:
        return "too_few_visible_frames"
    if nearest_truth_candidate(final_rows, event, truth_match_radius_px) is not None:
        return "candidate_ranked_but_not_truth_matched"
    if nearest_truth_candidate(measured_rows, event, truth_match_radius_px) is not None:
        return "candidate_generated_but_filtered"
    return "below_candidate_threshold"


def diagnose_false_positives(
    *,
    candidates: list[dict],
    cosmic_dir: Path,
    cosmic_cache: dict[int, np.ndarray],
    cosmic_match_radius_px: float,
    template_sources: list[dict],
    template_tree: cKDTree | None,
    star_match_radius_px: float,
) -> list[dict]:
    false_rows: list[dict] = []
    for row in candidates:
        if truthy(row.get("truth_match_flag")):
            continue
        cosmic = match_cosmic_candidate(row, cosmic_dir, cosmic_cache, cosmic_match_radius_px)
        star = match_template_star(row, template_sources, template_tree, star_match_radius_px)
        if truthy(row.get("likely_cosmic_ray")) or cosmic["cosmic_truth_match_flag"] == "1":
            fp_class = "false_likely_cosmic"
        elif star["nearest_template_star_dist_px"] not in {"", "nan"}:
            fp_class = "false_template_star_like"
        elif truthy(row.get("previous_block_match_flag")):
            fp_class = "false_repeated_previous_block"
        else:
            fp_class = "false_unknown_residual"
        out = dict(row)
        out.update(cosmic)
        out.update(star)
        out["false_positive_class"] = fp_class
        out["diagnostic_tags"] = diagnostic_tags(row, cosmic, star)
        false_rows.append(out)
    return false_rows


def diagnostic_tags(row: dict, cosmic: dict, star: dict) -> str:
    tags: list[str] = []
    if truthy(row.get("likely_cosmic_ray")):
        tags.append("finder_likely_cosmic")
    if cosmic["cosmic_truth_match_flag"] == "1":
        tags.append("cosmic_truth_footprint")
    if cosmic["cosmic_truth_same_frame_flag"] == "1":
        tags.append("cosmic_truth_same_frame")
    if star["nearest_template_star_dist_px"] not in {"", "nan"}:
        tags.append("template_star_nearby")
    return ";".join(tags)


def parse_temporal_flux_series(value) -> list[float]:
    if value in (None, ""):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [to_float(item) for item in parsed]


def strongest_temporal_frame(row: dict) -> int | None:
    series = parse_temporal_flux_series(row.get("temporal_flux_series"))
    finite = [(idx, val) for idx, val in enumerate(series) if not math.isnan(val)]
    if not finite:
        return None
    strongest_idx = max(finite, key=lambda item: item[1])[0]
    return to_int(row.get("frame_start")) + strongest_idx


def match_cosmic_candidate(
    row: dict,
    cosmic_dir: Path,
    cosmic_cache: dict[int, np.ndarray],
    radius: float,
) -> dict:
    start, end = block_key(row)
    x = to_float(row.get("x"))
    y = to_float(row.get("y"))
    best: tuple[float, int, np.void] | None = None
    strongest = strongest_temporal_frame(row)
    same_frame = False
    for frame in range(start, end + 1):
        events = load_cosmic_events(cosmic_dir, frame, cosmic_cache)
        for event in events:
            dist = distance_to_box(x, y, float(event["x0"]), float(event["y0"]), float(event["x1"]), float(event["y1"]))
            if dist <= radius and (best is None or dist < best[0]):
                best = (dist, frame, event)
            if strongest == frame and dist <= radius:
                same_frame = True
    if best is None:
        return {
            "cosmic_truth_match_flag": "0",
            "cosmic_truth_same_frame_flag": "0",
            "cosmic_truth_dist_px": "nan",
            "cosmic_truth_frame": "",
            "cosmic_truth_x0": "",
            "cosmic_truth_y0": "",
            "cosmic_truth_x1": "",
            "cosmic_truth_y1": "",
            "cosmic_truth_total_adu": "",
            "cosmic_truth_peak_adu": "",
        }
    dist, frame, event = best
    return {
        "cosmic_truth_match_flag": "1",
        "cosmic_truth_same_frame_flag": str(int(same_frame)),
        "cosmic_truth_dist_px": fmt(dist),
        "cosmic_truth_frame": str(frame),
        "cosmic_truth_x0": fmt(event["x0"]),
        "cosmic_truth_y0": fmt(event["y0"]),
        "cosmic_truth_x1": fmt(event["x1"]),
        "cosmic_truth_y1": fmt(event["y1"]),
        "cosmic_truth_total_adu": fmt(event["total_adu"]),
        "cosmic_truth_peak_adu": fmt(event["peak_adu"]),
    }


def load_cosmic_events(cosmic_dir: Path, frame: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    if frame not in cache:
        path = cosmic_dir / f"frame_{frame:06d}_events.npy"
        if path.exists():
            cache[frame] = np.load(path, allow_pickle=False)
        else:
            cache[frame] = np.array([], dtype=[])
    return cache[frame]


def distance_to_box(x: float, y: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    if any(math.isnan(value) for value in [x1, y1, x2, y2]):
        return math.inf
    return math.hypot(x1 - x2, y1 - y2)


def build_template_tree(template_sources: list[dict]) -> cKDTree | None:
    if not template_sources:
        return None
    xy = np.asarray([(to_float(row.get("x")), to_float(row.get("y"))) for row in template_sources], dtype=float)
    if xy.size == 0:
        return None
    return cKDTree(xy)


def match_template_star(
    row: dict,
    template_sources: list[dict],
    template_tree: cKDTree | None,
    radius: float,
) -> dict:
    if truthy(row.get("template_match_flag")) and not math.isnan(to_float(row.get("nearest_template_dist_px"))):
        return {
            "nearest_template_star_dist_px": row.get("nearest_template_dist_px", ""),
            "nearest_template_star_x": row.get("nearest_template_x", ""),
            "nearest_template_star_y": row.get("nearest_template_y", ""),
            "template_star_peak_value": "",
        }
    if template_tree is None:
        return {
            "nearest_template_star_dist_px": "",
            "nearest_template_star_x": "",
            "nearest_template_star_y": "",
            "template_star_peak_value": "",
        }
    dist, idx = template_tree.query([to_float(row.get("x")), to_float(row.get("y"))], distance_upper_bound=radius)
    if math.isinf(float(dist)) or idx >= len(template_sources):
        return {
            "nearest_template_star_dist_px": "",
            "nearest_template_star_x": "",
            "nearest_template_star_y": "",
            "template_star_peak_value": "",
        }
    source = template_sources[int(idx)]
    return {
        "nearest_template_star_dist_px": fmt(dist),
        "nearest_template_star_x": source.get("x", ""),
        "nearest_template_star_y": source.get("y", ""),
        "template_star_peak_value": source.get("source_peak_value", ""),
    }


def load_or_build_template_sources(
    *,
    request: AnalysisRequest,
    input_run: Path,
    output_dir: Path,
    windows: list[dict],
) -> list[dict]:
    output_path = output_dir / "template_sources.csv"
    if request.template_sources_csv:
        sources = read_csv_rows(Path(request.template_sources_csv))
        write_csv_rows(output_path, sources)
        return sources
    if output_path.exists():
        return read_csv_rows(output_path)
    cached_sources = load_template_sources_from_aux_cache(input_run)
    if cached_sources:
        write_csv_rows(output_path, cached_sources)
        return cached_sources
    sources = build_template_sources_from_windows(
        input_run=input_run,
        windows=windows,
        threshold_sigma=request.star_source_threshold_sigma,
        tile_size=request.template_tile_size,
    )
    write_csv_rows(output_path, sources)
    return sources


def load_template_sources_from_aux_cache(input_run: Path) -> list[dict]:
    cache_dir = input_run / "copied_source_aux" / "cache"
    if not cache_dir.exists():
        return []
    npz_files = sorted(cache_dir.glob("stars_*.npz"))
    if not npz_files:
        return []
    path = npz_files[0]
    with np.load(path, allow_pickle=False) as data:
        x_key = first_existing_key(data.files, ["detector_xpix", "detector_xpix_shifted", "x", "x0"])
        y_key = first_existing_key(data.files, ["detector_ypix", "detector_ypix_shifted", "y", "y0"])
        if x_key is None or y_key is None:
            return []
        flux_key = first_existing_key(data.files, ["total_flux", "source_peak_value", "flux"])
        source_id_key = first_existing_key(data.files, ["source_id", "gaia_source_id"])
        xs = np.asarray(data[x_key], dtype=float)
        ys = np.asarray(data[y_key], dtype=float)
        fluxes = np.asarray(data[flux_key], dtype=float) if flux_key else np.full(xs.shape, math.nan)
        source_ids = data[source_id_key] if source_id_key else np.full(xs.shape, "")
        sources: list[dict] = []
        for idx, (x, y) in enumerate(zip(xs, ys, strict=False)):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            sources.append(
                {
                    "x": fmt(float(x)),
                    "y": fmt(float(y)),
                    "source_peak_value": fmt(float(fluxes[idx])) if idx < len(fluxes) else "",
                    "template_source_threshold": "",
                    "template_bkg_median": "",
                    "template_bkg_sigma": "",
                    "source_id": fmt(source_ids[idx]) if idx < len(source_ids) else "",
                    "catalog_source": "copied_source_aux_cache",
                }
            )
    return sources


def first_existing_key(keys: Iterable[str], candidates: list[str]) -> str | None:
    available = set(keys)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def build_template_sources_from_windows(
    *,
    input_run: Path,
    windows: list[dict],
    threshold_sigma: float,
    tile_size: int,
) -> list[dict]:
    frame_paths = sorted((input_run / "frames").glob("frame_*.npy"))
    if not frame_paths or not windows:
        return []
    first = np.load(frame_paths[0], mmap_mode="r")
    rows, cols = first.shape
    seen: set[tuple[int, int]] = set()
    sources: list[dict] = []
    for window in unique_template_windows(windows):
        start = to_int(window.get("template_frame_start"))
        end = to_int(window.get("template_frame_end"))
        if end < start:
            continue
        for row0 in range(0, rows, tile_size):
            row1 = min(row0 + tile_size, rows)
            for col0 in range(0, cols, tile_size):
                col1 = min(col0 + tile_size, cols)
                tile = sum_tile(frame_paths, start, end, row0, row1, col0, col1)
                sources.extend(
                    detect_template_sources_in_tile(
                        tile,
                        row0=row0,
                        col0=col0,
                        threshold_sigma=threshold_sigma,
                        seen=seen,
                    )
                )
    return sources


def unique_template_windows(windows: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for row in windows:
        key = (to_int(row.get("template_frame_start")), to_int(row.get("template_frame_end")))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def sum_tile(frame_paths: list[Path], start: int, end: int, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
    total: np.ndarray | None = None
    for frame_index in range(start, end + 1):
        if frame_index < 0 or frame_index >= len(frame_paths):
            continue
        frame = np.load(frame_paths[frame_index], mmap_mode="r")
        tile = np.asarray(frame[row0:row1, col0:col1], dtype=np.uint32)
        total = tile.copy() if total is None else total + tile
    if total is None:
        return np.zeros((row1 - row0, col1 - col0), dtype=np.uint32)
    return total


def detect_template_sources_in_tile(
    tile: np.ndarray,
    *,
    row0: int,
    col0: int,
    threshold_sigma: float,
    seen: set[tuple[int, int]],
) -> list[dict]:
    if tile.size == 0:
        return []
    median = float(np.median(tile))
    mad = float(np.median(np.abs(tile.astype(float) - median)))
    sigma = 1.4826 * mad
    if sigma == 0.0:
        sigma = float(np.std(tile))
    threshold = median + threshold_sigma * sigma
    mask = tile > threshold
    if not np.any(mask):
        return []
    maxima = tile == ndimage.maximum_filter(tile, size=3, mode="nearest")
    labels, count = ndimage.label(mask & maxima)
    sources: list[dict] = []
    for label_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == label_id)
        if ys.size == 0:
            continue
        values = tile[ys, xs]
        peak_index = int(np.argmax(values))
        y = int(row0 + ys[peak_index])
        x = int(col0 + xs[peak_index])
        if (x, y) in seen:
            continue
        seen.add((x, y))
        sources.append(
            {
                "x": str(x),
                "y": str(y),
                "source_peak_value": fmt(int(values[peak_index])),
                "template_source_threshold": fmt(threshold),
                "template_bkg_median": fmt(median),
                "template_bkg_sigma": fmt(sigma),
            }
        )
    return sources


def group_false_positives(false_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in false_rows:
        groups[(row.get("frame_start", ""), row.get("frame_end", ""), row.get("false_positive_class", ""))].append(row)
    output: list[dict] = []
    total_by_window = Counter((row.get("frame_start", ""), row.get("frame_end", "")) for row in false_rows)
    for (start, end, fp_class), rows in sorted(groups.items()):
        flux = [to_float(row.get("residual_flux")) for row in rows if not math.isnan(to_float(row.get("residual_flux")))]
        output.append(
            {
                "frame_start": start,
                "frame_end": end,
                "false_positive_class": fp_class,
                "count": str(len(rows)),
                "fraction_of_false_candidates": fmt(len(rows) / total_by_window[(start, end)]),
                "max_peak_pixel_snr": fmt(max((to_float(row.get("peak_pixel_snr")) for row in rows), default=math.nan)),
                "max_residual_flux": fmt(max(flux) if flux else math.nan),
                "median_residual_flux": fmt(float(np.median(flux)) if flux else math.nan),
            }
        )
    return output


def build_summary(
    *,
    result_dir: Path,
    input_run: Path,
    manifest: dict,
    windows: list[dict],
    candidates: list[dict],
    truth_rows: list[dict],
    false_rows: list[dict],
) -> dict:
    detected = sum(1 for row in truth_rows if row.get("detected") == "1")
    truth_count = len(truth_rows)
    false_classes = Counter(row.get("false_positive_class", "") for row in false_rows)
    return {
        "result_dir": str(result_dir),
        "input_run": str(input_run),
        "template_strategy": manifest.get("template_strategy", ""),
        "spatial_bin": f"{manifest.get('spatial_bin_rows', '')}x{manifest.get('spatial_bin_cols', '')}",
        "windows": len(windows),
        "truth_events_in_windows": truth_count,
        "truth_events_detected": detected,
        "truth_recall": detected / truth_count if truth_count else 0.0,
        "final_candidates": len(candidates),
        "false_candidates": len(false_rows),
        "false_cosmic_truth": sum(1 for row in false_rows if row.get("cosmic_truth_match_flag") == "1"),
        "false_likely_cosmic": false_classes["false_likely_cosmic"],
        "false_template_star_like": false_classes["false_template_star_like"],
        "false_unknown": false_classes["false_unknown_residual"],
    }


def render_report(
    *,
    summary: dict,
    manifest: dict,
    window_rows: list[dict],
    truth_rows: list[dict],
    false_rows: list[dict],
    group_rows: list[dict],
) -> str:
    lines = [
        "# GRB Finder 结果分析报告",
        "",
        "## 1. 输入和运行配置",
        "",
        f"- result_dir: `{summary['result_dir']}`",
        f"- input_run: `{summary['input_run']}`",
        f"- template_strategy: `{summary.get('template_strategy', '')}`",
        f"- spatial_bin: `{summary.get('spatial_bin', '')}`",
        "",
        "## 2. 检测块摘要",
        "",
        "| frame_start | frame_end | GRB truth | detected | recall | final | false |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in window_rows:
        lines.append(
            f"| {row['frame_start']} | {row['frame_end']} | {row['truth_events_in_block']} | "
            f"{row['truth_events_detected']} | {row['truth_recall']} | {row['final_candidates']} | {row['false_candidates']} |"
        )
    lines.extend(
        [
            "",
            "## 3. GRB truth 覆盖和命中率",
            "",
            f"- truth_events_in_windows: {summary['truth_events_in_windows']}",
            f"- truth_events_detected: {summary['truth_events_detected']}",
            f"- truth_recall: {summary['truth_recall']:.6g}",
            "",
            "## 4. 命中 GRB 的候选排名",
            "",
            "`snr_rank` 按 `peak_pixel_snr` 降序排序；`residual_flux_rank` 按 `residual_flux` 降序排序；"
            "`residual_peak_rank` 按 `residual_peak_value` 降序排序。三个排名都在每个检测块内单独计算。",
            "",
            "| event_id | detected | snr_rank | residual_flux_rank | residual_peak_rank | miss_reason |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in truth_rows:
        lines.append(
            f"| {row['event_id']} | {row['detected']} | {row['snr_rank']} | "
            f"{row['residual_flux_rank']} | {row['residual_peak_rank']} | {row['miss_reason']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 未命中 GRB 诊断",
            "",
        ]
    )
    missed = [row for row in truth_rows if row.get("detected") != "1"]
    if missed:
        for row in missed:
            lines.append(f"- event {row['event_id']}: `{row['miss_reason']}`")
    else:
        lines.append("- 无未命中 GRB。")
    lines.extend(
        [
            "",
            "## 6. 误检总体统计",
            "",
            f"- false_candidates: {summary['false_candidates']}",
            f"- false_cosmic_truth: {summary['false_cosmic_truth']}",
            f"- false_template_star_like: {summary['false_template_star_like']}",
            f"- false_unknown: {summary['false_unknown']}",
            "",
            "| class | count | fraction |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in group_rows:
        lines.append(f"| {row['false_positive_class']} | {row['count']} | {row['fraction_of_false_candidates']} |")
    lines.extend(
        [
            "",
            "## 7. 宇宙线类误检",
            "",
            "`cosmic_truth_match_flag` 是检测块级 footprint 匹配；`cosmic_truth_same_frame_flag` 是候选最强 raw frame 与 cosmic footprint 同帧匹配，证据更强。",
            "",
            f"- cosmic_truth_same_frame_flag count: {sum(1 for row in false_rows if row.get('cosmic_truth_same_frame_flag') == '1')}",
            "",
            "## 8. 星点/模板源类误检",
            "",
            "模板星源 catalog 写出为 `template_sources.csv`，后续可用 `--template-sources-csv` 复用。",
            "",
            "## 9. 其它高 SNR 误检",
            "",
            "详见 `top_candidates_by_snr.csv` 和 `false_positive_diagnostics.csv`。",
            "",
            "## 10. 数据缺失和诊断限制",
            "",
            "当前第一版只输出 Markdown 和 CSV，不生成图表或 cutout 图片。",
        ]
    )
    return "\n".join(lines) + "\n"

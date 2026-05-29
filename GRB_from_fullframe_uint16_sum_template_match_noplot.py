#!/usr/bin/env python3
"""
Streaming 12-frame sum transient screener for current full-frame GRB products.

This is a copy/adaptation of the 501x501 integer prototype. It does not replace
GRB_from_integer20_npy_template_match_sum_noplot.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import find_objects, label, maximum_filter
from scipy.spatial import cKDTree


DEFAULT_INPUT_RUN = Path(
    "/home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_sum12_streaming"
)

SUM_DTYPE = np.uint32
LOCAL_DIFF_DTYPE = np.int64


@dataclass(frozen=True)
class ScreenerConfig:
    window_size: int = 12
    stride: int = 12
    tile_size: int = 1024
    halo: int = 12
    input_bit_depth: int = 16
    max_filter_size: int = 7
    source_threshold_sigma: float = 4.0
    residual_threshold_sigma: float = 3.0
    residual_min_npix: int = 2
    residual_max_npix: int = 400
    match_radius_px: float = 0.75
    cut_half: int = 9
    annulus_r_in: float = 6.0
    annulus_r_out: float = 10.0
    local_threshold_sigma: float = 3.0
    seed_radius: float = 1.5
    effective_npix_threshold: int = 4
    temporal_cut_half: int = 5
    temporal_aperture_radius: float = 3.0
    temporal_annulus_r_in: float = 5.0
    temporal_annulus_r_out: float = 8.0
    temporal_sigma: float = 3.0
    temporal_min_active_frames: int = 2
    cosmic_single_frame_fraction: float = 0.80
    cosmic_max_active_frames: int = 1
    local_shape_check: bool = True
    temporal_check: bool = True
    keep_all_residual_candidates: bool = False


def input_range_from_bit_depth(bit_depth: int) -> tuple[int, int]:
    if bit_depth <= 0:
        raise ValueError(f"input bit depth must be positive, got {bit_depth}")
    return 0, (1 << bit_depth) - 1


def frame_paths_from_run(run_dir: str | Path) -> list[Path]:
    frames_dir = Path(run_dir) / "frames"
    paths = sorted(frames_dir.glob("frame_*.npy"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.npy files found in {frames_dir}")
    return paths


def frame_shape_and_dtype(frame_paths: list[Path]) -> tuple[tuple[int, int], np.dtype]:
    arr = np.load(frame_paths[0], mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D full frame, got shape={arr.shape} from {frame_paths[0]}")
    return (int(arr.shape[0]), int(arr.shape[1])), np.dtype(arr.dtype)


def validate_frame_range(frame_paths: list[Path], bit_depth: int) -> None:
    lo, hi = input_range_from_bit_depth(bit_depth)
    for path in (frame_paths[0], frame_paths[-1]):
        arr = np.load(path, mmap_mode="r")
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"Expected integer frame dtype, got {arr.dtype} from {path}")
        arr_min = int(arr.min())
        arr_max = int(arr.max())
        if arr_min < lo or arr_max > hi:
            raise ValueError(
                f"{path} has values outside {bit_depth}-bit unsigned range "
                f"[{lo}, {hi}]: min={arr_min}, max={arr_max}"
            )


def window_ranges(n_frames: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    ranges = []
    start = 0
    while start < n_frames:
        end = min(n_frames, start + window_size)
        ranges.append((start, end))
        if end == n_frames:
            break
        start += stride
    return ranges


def iter_core_tiles(shape: tuple[int, int], tile_size: int) -> Iterable[tuple[int, int, int, int]]:
    n_rows, n_cols = shape
    for row0 in range(0, n_rows, tile_size):
        row1 = min(n_rows, row0 + tile_size)
        for col0 in range(0, n_cols, tile_size):
            col1 = min(n_cols, col0 + tile_size)
            yield row0, row1, col0, col1


def expand_tile(
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    shape: tuple[int, int],
    halo: int,
) -> tuple[slice, slice]:
    n_rows, n_cols = shape
    return (
        slice(max(0, row0 - halo), min(n_rows, row1 + halo)),
        slice(max(0, col0 - halo), min(n_cols, col1 + halo)),
    )


def sum_frame_tile(
    frame_paths: list[Path],
    frame_start: int,
    frame_end: int,
    row_slice: slice,
    col_slice: slice,
) -> np.ndarray:
    if frame_start < 0 or frame_end > len(frame_paths) or frame_start >= frame_end:
        raise ValueError(f"invalid frame window {frame_start}:{frame_end}")
    out = None
    for path in frame_paths[frame_start:frame_end]:
        frame = np.load(path, mmap_mode="r")
        tile = np.asarray(frame[row_slice, col_slice], dtype=SUM_DTYPE)
        if out is None:
            out = tile.copy()
        else:
            out += tile
    return out


def integer_median(values: np.ndarray) -> int:
    vals = np.asarray(values).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0
    k = vals.size // 2
    if vals.size % 2:
        return int(np.partition(vals, k)[k])
    part = np.partition(vals, (k - 1, k))
    return (int(part[k - 1]) + int(part[k]) + 1) // 2


def robust_integer_sigma(values: np.ndarray) -> tuple[int, int]:
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0, 0
    med = integer_median(vals)
    abs_dev = np.abs(vals.astype(LOCAL_DIFF_DTYPE, copy=False) - med)
    mad = integer_median(abs_dev)
    sigma = (int(mad) * 14826 + 5000) // 10000
    return int(med), int(sigma)


def detect_sources_on_tile(
    sum_tile: np.ndarray,
    *,
    row_origin: int,
    col_origin: int,
    core_bounds: tuple[int, int, int, int],
    cfg: ScreenerConfig,
) -> tuple[list[dict], int, int, int]:
    bkg_median, bkg_sigma = robust_integer_sigma(sum_tile)
    threshold = int(bkg_median + cfg.source_threshold_sigma * bkg_sigma)
    is_local_max = sum_tile == maximum_filter(
        sum_tile,
        size=cfg.max_filter_size,
        mode="nearest",
    )
    peak_mask = is_local_max & (sum_tile > threshold) & (sum_tile > 0)
    peak_labels, n_peak_labels = label(peak_mask, structure=np.ones((3, 3), dtype=int))
    row0, row1, col0, col1 = core_bounds
    rows = []
    peak_objects = find_objects(peak_labels, max_label=n_peak_labels)
    for peak_label, peak_slice in enumerate(peak_objects, start=1):
        if peak_slice is None:
            continue
        y_slice, x_slice = peak_slice
        local_peak_labels = peak_labels[peak_slice]
        local_mask = local_peak_labels == peak_label
        if not np.any(local_mask):
            continue
        local_ys, local_xs = np.where(local_mask)
        ys = local_ys + int(y_slice.start)
        xs = local_xs + int(x_slice.start)
        vals = sum_tile[peak_slice][local_mask]
        peak_value = int(vals.max())
        max_idx = np.flatnonzero(vals == peak_value)
        if max_idx.size > 1:
            y_center = float(np.mean(ys[max_idx]))
            x_center = float(np.mean(xs[max_idx]))
            best_idx = max_idx[
                int(np.argmin((ys[max_idx] - y_center) ** 2 + (xs[max_idx] - x_center) ** 2))
            ]
        else:
            best_idx = int(max_idx[0])
        y_local = int(ys[best_idx])
        x_local = int(xs[best_idx])
        y = int(row_origin + y_local)
        x = int(col_origin + x_local)
        if not (row0 <= y < row1 and col0 <= x < col1):
            continue
        rows.append(
            {
                "x": x,
                "y": y,
                "source_peak_value": peak_value,
            }
        )
    rows.sort(key=lambda row: row["source_peak_value"], reverse=True)
    return rows, threshold, bkg_median, bkg_sigma


def detect_residual_sources_on_tile(
    sum_tile: np.ndarray,
    template_sum_tile: np.ndarray,
    *,
    row_origin: int,
    col_origin: int,
    core_bounds: tuple[int, int, int, int],
    cfg: ScreenerConfig,
) -> tuple[list[dict], int, int, int]:
    diff = sum_tile.astype(LOCAL_DIFF_DTYPE, copy=False) - template_sum_tile.astype(
        LOCAL_DIFF_DTYPE,
        copy=False,
    )
    bkg_median, bkg_sigma = robust_integer_sigma(diff)
    if bkg_sigma <= 0:
        threshold = max(int(bkg_median), 0)
    else:
        threshold = int(bkg_median + cfg.residual_threshold_sigma * bkg_sigma)

    mask = (diff > threshold) & (diff > 0)
    labels, nlabels = label(mask, structure=np.ones((3, 3), dtype=int))
    objects = find_objects(labels, max_label=nlabels)
    row0, row1, col0, col1 = core_bounds
    rows = []
    for lab, lab_slice in enumerate(objects, start=1):
        if lab_slice is None:
            continue
        y_slice, x_slice = lab_slice
        local_labels = labels[lab_slice]
        local_mask = local_labels == lab
        if not np.any(local_mask):
            continue
        vals = diff[lab_slice][local_mask]
        npix = int(vals.size)
        if npix < cfg.residual_min_npix or npix > cfg.residual_max_npix:
            continue

        local_ys, local_xs = np.where(local_mask)
        ys = local_ys + int(y_slice.start)
        xs = local_xs + int(x_slice.start)
        peak_value = int(vals.max())
        max_idx = np.flatnonzero(vals == peak_value)
        if max_idx.size > 1:
            y_center = float(np.mean(ys[max_idx]))
            x_center = float(np.mean(xs[max_idx]))
            best_idx = max_idx[
                int(np.argmin((ys[max_idx] - y_center) ** 2 + (xs[max_idx] - x_center) ** 2))
            ]
        else:
            best_idx = int(max_idx[0])

        y_local = int(ys[best_idx])
        x_local = int(xs[best_idx])
        y = int(row_origin + y_local)
        x = int(col_origin + x_local)
        if not (row0 <= y < row1 and col0 <= x < col1):
            continue
        rows.append(
            {
                "x": x,
                "y": y,
                "source_peak_value": int(sum_tile[y_local, x_local]),
                "residual_peak_value": peak_value,
                "residual_flux": int(
                    max(int(vals.sum(dtype=LOCAL_DIFF_DTYPE)) - int(bkg_median) * npix, 0)
                ),
                "residual_npix": npix,
            }
        )
    rows.sort(key=lambda row: row["residual_flux"], reverse=True)
    return rows, threshold, bkg_median, bkg_sigma


def annotate_template_matches(
    candidates: list[dict],
    template_xy: np.ndarray,
    match_radius_px: float,
) -> list[dict]:
    if template_xy.size == 0:
        out = []
        for row in candidates:
            rec = dict(row)
            rec.update(
                {
                    "nearest_template_dist_px": np.nan,
                    "nearest_template_x": np.nan,
                    "nearest_template_y": np.nan,
                    "template_match_flag": 0,
                    "candidate_channel": "new_source",
                }
            )
            out.append(rec)
        return out

    tree = cKDTree(template_xy)
    out = []
    for row in candidates:
        dist, idx = tree.query([row["x"], row["y"]], k=1)
        matched = int(float(dist) <= float(match_radius_px))
        rec = dict(row)
        rec.update(
            {
                "nearest_template_dist_px": float(dist),
                "nearest_template_x": float(template_xy[idx, 0]),
                "nearest_template_y": float(template_xy[idx, 1]),
                "template_match_flag": matched,
                "candidate_channel": "template_source_brightening" if matched else "new_source",
            }
        )
        out.append(rec)
    return out


def candidate_local_xy(row: dict, *, row_slice: slice, col_slice: slice) -> tuple[int, int]:
    return int(row["x"]) - int(col_slice.start), int(row["y"]) - int(row_slice.start)


def extract_cutout(frame: np.ndarray, x: float, y: float, half: int):
    h, w = frame.shape
    x0 = max(0, int(np.floor(x)) - half)
    x1 = min(w, int(np.floor(x)) + half + 1)
    y0 = max(0, int(np.floor(y)) - half)
    y1 = min(h, int(np.floor(y)) + half + 1)
    return frame[y0:y1, x0:x1], x0, y0


def pick_component_label(
    labels: np.ndarray,
    diff_patch: np.ndarray,
    seed_mask: np.ndarray,
    seed_mask_fallback: np.ndarray,
) -> int:
    seed_labels = np.unique(labels[seed_mask])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        seed_labels = np.unique(labels[seed_mask_fallback])
        seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return 0

    best_label = 0
    best_peak = -np.inf
    for lab in seed_labels:
        peak = int(np.max(diff_patch[labels == lab]))
        if peak > best_peak:
            best_peak = peak
            best_label = int(lab)
    return best_label


def measure_candidate_local_excess(
    sum_img: np.ndarray,
    template_sum_img: np.ndarray,
    cand_xy,
    cfg: ScreenerConfig,
):
    xc, yc = cand_xy
    tmpl_cut, x0, y0 = extract_cutout(template_sum_img, xc, yc, cfg.cut_half)
    h, w = tmpl_cut.shape
    x1 = x0 + w
    y1 = y0 + h
    img_cut = sum_img[y0:y1, x0:x1]
    if img_cut.shape != tmpl_cut.shape:
        return 0, 0, 0, 0, 0

    xc_local = float(xc - x0)
    yc_local = float(yc - y0)
    yy, xx = np.indices(tmpl_cut.shape)
    rr = np.hypot(xx - xc_local, yy - yc_local)
    annulus = (rr >= cfg.annulus_r_in) & (rr <= cfg.annulus_r_out)
    seed_mask = rr <= cfg.seed_radius
    seed_mask_fallback = rr <= (cfg.seed_radius + 1.0)
    structure = np.ones((3, 3), dtype=int)

    diff = img_cut.astype(LOCAL_DIFF_DTYPE, copy=False) - tmpl_cut.astype(
        LOCAL_DIFF_DTYPE,
        copy=False,
    )
    bkg_med, bkg_sigma = robust_integer_sigma(diff[annulus])
    if bkg_sigma <= 0:
        bkg_med, bkg_sigma = robust_integer_sigma(diff)
    if bkg_sigma <= 0:
        threshold = max(int(bkg_med), 0)
    else:
        threshold = int(bkg_med + cfg.local_threshold_sigma * bkg_sigma)
    mask = diff > threshold
    if not np.any(mask):
        return 0, 0, 0, int(bkg_med), int(bkg_sigma)

    labels, nlab = label(mask, structure=structure)
    if nlab == 0:
        return 0, 0, 0, int(bkg_med), int(bkg_sigma)

    chosen = pick_component_label(labels, diff, seed_mask, seed_mask_fallback)
    if chosen <= 0:
        return 0, 0, 0, int(bkg_med), int(bkg_sigma)

    comp_vals = diff[labels == chosen]
    npix = int(comp_vals.size)
    local_excess_flux = int(
        max(int(comp_vals.sum(dtype=LOCAL_DIFF_DTYPE)) - int(bkg_med) * npix, 0)
    )
    peak_excess_value = int(comp_vals.max() - int(bkg_med))
    return local_excess_flux, npix, peak_excess_value, int(bkg_med), int(bkg_sigma)


def measure_temporal_support(
    frame_paths: list[Path],
    template_frame_paths: list[Path] | None,
    frame_start: int,
    frame_end: int,
    x: int,
    y: int,
    cfg: ScreenerConfig,
) -> dict:
    fluxes: list[int] = []
    for idx in range(frame_start, frame_end):
        frame = np.load(frame_paths[idx], mmap_mode="r")
        cut, x0, y0 = extract_cutout(frame, x, y, cfg.temporal_cut_half)
        cut_data = cut.astype(LOCAL_DIFF_DTYPE, copy=False)
        if template_frame_paths is not None and idx < len(template_frame_paths):
            tmpl = np.load(template_frame_paths[idx], mmap_mode="r")
            tmpl_cut, _, _ = extract_cutout(tmpl, x, y, cfg.temporal_cut_half)
            if tmpl_cut.shape == cut.shape:
                cut_data = cut_data - tmpl_cut.astype(LOCAL_DIFF_DTYPE, copy=False)

        yy, xx = np.indices(cut.shape)
        rr = np.hypot(xx - float(x - x0), yy - float(y - y0))
        aperture = rr <= cfg.temporal_aperture_radius
        annulus = (rr >= cfg.temporal_annulus_r_in) & (rr <= cfg.temporal_annulus_r_out)
        bkg_med = integer_median(cut_data[annulus]) if np.any(annulus) else integer_median(cut_data)
        flux = int(cut_data[aperture].sum(dtype=LOCAL_DIFF_DTYPE) - bkg_med * int(aperture.sum()))
        fluxes.append(max(flux, 0))

    if not fluxes:
        return {
            "temporal_active_frames": 0,
            "temporal_consecutive_active_frames": 0,
            "temporal_max_single_frame_fraction": 0.0,
            "temporal_flux_series": "[]",
            "likely_cosmic_ray": 0,
        }

    flux_arr = np.asarray(fluxes, dtype=np.int64)
    med, sigma = robust_integer_sigma(flux_arr)
    threshold = max(1, int(med + cfg.temporal_sigma * sigma))
    active = flux_arr > threshold
    best_run = 0
    run = 0
    for val in active:
        run = run + 1 if val else 0
        best_run = max(best_run, run)
    total = int(flux_arr.sum(dtype=np.int64))
    max_fraction = float(flux_arr.max() / total) if total > 0 else 0.0
    likely_cosmic = int(
        int(np.count_nonzero(active)) <= cfg.cosmic_max_active_frames
        and max_fraction >= cfg.cosmic_single_frame_fraction
    )
    return {
        "temporal_active_frames": int(np.count_nonzero(active)),
        "temporal_consecutive_active_frames": int(best_run),
        "temporal_max_single_frame_fraction": max_fraction,
        "temporal_flux_series": json.dumps([int(v) for v in flux_arr], separators=(",", ":")),
        "likely_cosmic_ray": likely_cosmic,
    }


def load_truth_events(path: str | Path | None) -> list[dict]:
    if path is None:
        return []
    truth_path = Path(path)
    if not truth_path.is_file():
        return []
    rows = []
    with truth_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def annotate_truth_matches(
    candidates: list[dict],
    truth_events: list[dict],
    radius_px: float,
) -> list[dict]:
    if not truth_events:
        out = []
        for row in candidates:
            rec = dict(row)
            rec.update({"truth_match_flag": 0, "truth_event_id": "", "truth_dist_px": np.nan})
            out.append(rec)
        return out

    truth_xy = np.asarray(
        [
            [float(row["requested_detector_xpix"]), float(row["requested_detector_ypix"])]
            for row in truth_events
        ],
        dtype=float,
    )
    tree = cKDTree(truth_xy)
    out = []
    for row in candidates:
        rec = dict(row)
        dist, idx = tree.query([row["x"], row["y"]], k=1)
        truth = truth_events[int(idx)]
        frame_start = int(row["frame_start"])
        frame_end = int(row["frame_end"])
        first = int(float(truth.get("first_visible_frame", -10**9)))
        last = int(float(truth.get("last_visible_frame", -10**9)))
        time_overlap = frame_start <= last and frame_end >= first
        matched = bool(float(dist) <= float(radius_px) and time_overlap)
        rec.update(
            {
                "truth_match_flag": int(matched),
                "truth_event_id": truth.get("event_id", "") if matched else "",
                "truth_dist_px": float(dist) if matched else np.nan,
            }
        )
        out.append(rec)
    return out


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_template_sources(
    frame_paths: list[Path],
    frame_start: int,
    frame_end: int,
    shape: tuple[int, int],
    cfg: ScreenerConfig,
) -> tuple[np.ndarray, list[dict]]:
    all_sources: list[dict] = []
    for row0, row1, col0, col1 in iter_core_tiles(shape, cfg.tile_size):
        row_slice, col_slice = expand_tile(row0, row1, col0, col1, shape, cfg.halo)
        tile = sum_frame_tile(frame_paths, frame_start, frame_end, row_slice, col_slice)
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


def scan_window(
    frame_paths: list[Path],
    template_frame_paths: list[Path],
    template_xy: np.ndarray,
    frame_start: int,
    frame_end: int,
    shape: tuple[int, int],
    cfg: ScreenerConfig,
    *,
    paired_template: bool,
) -> tuple[list[dict], dict]:
    measured: list[dict] = []
    initial_sources = 0
    matched_template_sources = 0
    for row0, row1, col0, col1 in iter_core_tiles(shape, cfg.tile_size):
        row_slice, col_slice = expand_tile(row0, row1, col0, col1, shape, cfg.halo)
        sum_img = sum_frame_tile(frame_paths, frame_start, frame_end, row_slice, col_slice)
        if paired_template:
            template_sum_img = sum_frame_tile(
                template_frame_paths,
                frame_start,
                frame_end,
                row_slice,
                col_slice,
            )
        else:
            template_sum_img = sum_frame_tile(
                template_frame_paths,
                0,
                min(cfg.window_size, len(template_frame_paths)),
                row_slice,
                col_slice,
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
            if cfg.temporal_check and (local_flux > 0 or peak_excess > 0):
                temporal = measure_temporal_support(
                    frame_paths,
                    template_frame_paths if paired_template else None,
                    frame_start,
                    frame_end,
                    int(row["x"]),
                    int(row["y"]),
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
                    npix >= cfg.effective_npix_threshold
                    or temporal["temporal_active_frames"] >= cfg.temporal_min_active_frames
                    or (np.isfinite(peak_pixel_snr) and peak_pixel_snr >= 5.0)
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
                    "local_excess_flux": int(local_flux),
                    "peak_npix": int(npix),
                    "peak_excess_value": int(peak_excess),
                    "local_bkg_median": int(local_bkg_med),
                    "local_bkg_sigma": int(local_bkg_sigma),
                    "peak_pixel_snr": peak_pixel_snr,
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
        "initial_sources": int(initial_sources),
        "template_matched_sources_kept": int(matched_template_sources),
        "after_measurement": int(len(measured)),
        "final_candidates": int(sum(row["pass_single_stack"] == 1 for row in measured)),
    }
    return measured, summary


def default_truth_path(input_run: Path) -> Path | None:
    path = input_run / "events.csv"
    return path if path.is_file() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, default=DEFAULT_INPUT_RUN)
    parser.add_argument("--template-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--truth-events-csv", type=Path, default=None)
    parser.add_argument("--truth-match-radius-px", type=float, default=12.0)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--halo", type=int, default=12)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--source-threshold-sigma", type=float, default=4.0)
    parser.add_argument("--residual-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--residual-min-npix", type=int, default=2)
    parser.add_argument("--residual-max-npix", type=int, default=400)
    parser.add_argument("--local-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--match-radius-px", type=float, default=0.75)
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
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ScreenerConfig:
    return ScreenerConfig(
        window_size=args.window_size,
        stride=args.stride,
        tile_size=args.tile_size,
        halo=args.halo,
        source_threshold_sigma=args.source_threshold_sigma,
        residual_threshold_sigma=args.residual_threshold_sigma,
        residual_min_npix=args.residual_min_npix,
        residual_max_npix=args.residual_max_npix,
        local_threshold_sigma=args.local_threshold_sigma,
        match_radius_px=args.match_radius_px,
        effective_npix_threshold=args.effective_npix_threshold,
        local_shape_check=args.local_shape_check,
        temporal_check=args.temporal_check,
        keep_all_residual_candidates=args.keep_all_residual_candidates,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = frame_paths_from_run(args.input_run)
    template_frame_paths = frame_paths_from_run(args.template_run or args.input_run)
    shape, dtype = frame_shape_and_dtype(frame_paths)
    template_shape, _template_dtype = frame_shape_and_dtype(template_frame_paths)
    if template_shape != shape:
        raise ValueError(f"template shape {template_shape} does not match input shape {shape}")
    validate_frame_range(frame_paths, cfg.input_bit_depth)

    ranges = window_ranges(len(frame_paths), cfg.window_size, cfg.stride)
    if args.max_windows is not None:
        ranges = ranges[: int(args.max_windows)]

    paired_template = args.template_run is not None
    template_xy = np.empty((0, 2), dtype=float)
    template_sources: list[dict] = []
    if args.template_match_sources and not paired_template:
        template_xy, template_sources = build_template_sources(
            template_frame_paths,
            0,
            min(cfg.window_size, len(template_frame_paths)),
            shape,
            cfg,
        )

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for frame_start, frame_end in ranges:
        if args.template_match_sources and paired_template:
            template_xy, template_sources = build_template_sources(
                template_frame_paths,
                frame_start,
                frame_end,
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
            paired_template=paired_template,
        )
        summary["template_source_count"] = int(len(template_xy))
        all_rows.extend(rows)
        summary_rows.append(summary)

    final_rows = [row for row in all_rows if row["pass_single_stack"] == 1]
    truth_path = args.truth_events_csv or default_truth_path(args.input_run)
    truth_events = load_truth_events(truth_path)
    all_rows = annotate_truth_matches(all_rows, truth_events, args.truth_match_radius_px)
    final_rows = annotate_truth_matches(final_rows, truth_events, args.truth_match_radius_px)

    measured_fields = [
        "frame_start",
        "frame_end",
        "window_frame_count",
        "x",
        "y",
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
        "local_excess_flux",
        "peak_npix",
        "peak_excess_value",
        "local_bkg_median",
        "local_bkg_sigma",
        "peak_pixel_snr",
        "effective_npix_threshold",
        "local_shape_check_enabled",
        "temporal_check_enabled",
        "temporal_active_frames",
        "temporal_consecutive_active_frames",
        "temporal_max_single_frame_fraction",
        "likely_cosmic_ray",
        "pass_single_stack",
        "truth_match_flag",
        "truth_event_id",
        "truth_dist_px",
        "temporal_flux_series",
    ]
    summary_fields = [
        "frame_start",
        "frame_end",
        "window_frame_count",
        "template_source_count",
        "initial_sources",
        "template_matched_sources_kept",
        "after_measurement",
        "final_candidates",
    ]
    save_csv(args.output_dir / "streaming_sum_candidates_after_measurement.csv", all_rows, measured_fields)
    save_csv(args.output_dir / "streaming_sum_transient_candidates.csv", final_rows, measured_fields)
    save_csv(args.output_dir / "streaming_sum_summary.csv", summary_rows, summary_fields)
    template_source_catalog_written = bool(args.template_match_sources)
    if template_source_catalog_written:
        save_csv(
            args.output_dir / "template_sources.csv",
            template_sources,
            [
                "x",
                "y",
                "source_peak_value",
                "template_source_threshold",
                "template_bkg_median",
                "template_bkg_sigma",
            ],
        )
    manifest = {
        "input_run": str(args.input_run),
        "template_run": str(args.template_run) if args.template_run else None,
        "output_dir": str(args.output_dir),
        "truth_events_csv": str(truth_path) if truth_path else None,
        "shape": list(shape),
        "dtype": str(dtype),
        "window_size": cfg.window_size,
        "stride": cfg.stride,
        "tile_size": cfg.tile_size,
        "halo": cfg.halo,
        "source_threshold_sigma": cfg.source_threshold_sigma,
        "residual_threshold_sigma": cfg.residual_threshold_sigma,
        "residual_min_npix": cfg.residual_min_npix,
        "residual_max_npix": cfg.residual_max_npix,
        "local_threshold_sigma": cfg.local_threshold_sigma,
        "match_radius_px": cfg.match_radius_px,
        "effective_npix_threshold": cfg.effective_npix_threshold,
        "template_match_sources": bool(args.template_match_sources),
        "template_source_catalog_written": template_source_catalog_written,
        "local_shape_check": bool(cfg.local_shape_check),
        "temporal_check": bool(cfg.temporal_check),
        "keep_all_residual_candidates": bool(cfg.keep_all_residual_candidates),
        "windows_processed": len(ranges),
        "candidates_after_measurement": len(all_rows),
        "final_candidates": len(final_rows),
        "truth_matched_final_candidates": int(sum(row.get("truth_match_flag", 0) == 1 for row in final_rows)),
        "notes": (
            "Template-matched sources are kept as template_source_brightening candidates. "
            "likely_cosmic_ray is an advisory flag, not a hard rejection."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Input run: {args.input_run}")
    print(f"Template run: {args.template_run or '(first input window)'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Windows processed: {len(ranges)}")
    print(f"Candidates after measurement: {len(all_rows)}")
    print(f"Final candidates: {len(final_rows)}")
    print(f"Truth-matched final candidates: {manifest['truth_matched_final_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

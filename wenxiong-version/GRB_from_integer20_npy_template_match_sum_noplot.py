#!/usr/bin/env python3
"""
No-argument transient screener for integer detector simulation.

This script is for onboard-style tests. It consumes one 12-frame integer NPY
slice at a time, forms a uint32 sum image, detects sources directly on that
current sum image, and removes sources already present in the first-slice sum
template. It does not build or cache a full-frame difference image.
"""
import csv
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.spatial import cKDTree


# Fixed onboard-test configuration. No command-line arguments are used.
INPUT_NPY_DIR = Path("/Users/lwx-mac/Documents/ET/GRB模拟/批量下载_uint16/501x501_GRB_x0p01_slices_12")
OUTPUT_DIR = Path("/Users/lwx-mac/Documents/ET/GRB模拟/grb_x0p01_integer20_sum_template_match_noplot")

INPUT_BIT_DEPTH = 20
EXPECTED_FRAMES_PER_SLICE = 12
EXPECTED_IMAGE_SHAPE = (501, 501)
SUM_DTYPE = np.uint32
LOCAL_DIFF_DTYPE = np.int64

MAX_FILTER_SIZE = 7
SOURCE_THRESHOLD_SIGMA = 5
MATCH_RADIUS_PX = 1.5
CUT_HALF = 9
ANNULUS_R_IN = 6.0
ANNULUS_R_OUT = 10.0
LOCAL_THRESHOLD_SIGMA = 5
SEED_RADIUS = 1.5
EFFECTIVE_NPIX_THRESHOLD = 10

TARGET_X = 388
TARGET_Y = 457
TARGET_MATCH_RADIUS_PX = 1.5


def input_range_from_bit_depth(bit_depth: int):
    if bit_depth <= 0:
        raise ValueError(f"input bit depth must be positive, got {bit_depth}")
    return 0, (1 << bit_depth) - 1


def parse_slice_info(path: Path):
    m = re.search(r"frames_(\d+)_(\d+)\.npy$", path.name)
    if not m:
        return -1, -1, -1
    frame_start = int(m.group(1))
    frame_end = int(m.group(2))
    slice_idx = frame_start // EXPECTED_FRAMES_PER_SLICE
    return slice_idx, frame_start, frame_end


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


def robust_integer_sigma(values: np.ndarray):
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0, 0
    med = integer_median(vals)
    abs_dev = np.abs(vals.astype(LOCAL_DIFF_DTYPE, copy=False) - med)
    mad = integer_median(abs_dev)
    sigma = (int(mad) * 14826 + 5000) // 10000
    return int(med), int(sigma)


def load_integer_npy_cube(path: Path) -> np.ndarray:
    cube = np.load(path, mmap_mode="r")
    if cube.ndim != 3:
        raise ValueError(f"Expected 3-D NPY cube, got shape={cube.shape} from {path}")
    if cube.shape != (EXPECTED_FRAMES_PER_SLICE, *EXPECTED_IMAGE_SHAPE):
        raise ValueError(
            f"Expected shape={(EXPECTED_FRAMES_PER_SLICE, *EXPECTED_IMAGE_SHAPE)}, "
            f"got shape={cube.shape} from {path}"
        )
    if not np.issubdtype(cube.dtype, np.integer):
        raise TypeError(f"Expected integer NPY input, got dtype={cube.dtype} from {path}")

    lo, hi = input_range_from_bit_depth(INPUT_BIT_DEPTH)
    arr_min = int(cube.min())
    arr_max = int(cube.max())
    if arr_min < lo or arr_max > hi:
        raise ValueError(
            f"{path} has values outside {INPUT_BIT_DEPTH}-bit unsigned range "
            f"[{lo}, {hi}]: min={arr_min}, max={arr_max}"
        )
    return cube


def sum_stack_from_npy(path: Path) -> np.ndarray:
    cube = load_integer_npy_cube(path)
    sum_img = np.asarray(cube, dtype=SUM_DTYPE).sum(axis=0, dtype=SUM_DTYPE)
    if sum_img.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"Expected 2-D sum image from {path}, got shape={sum_img.shape}")
    return sum_img


def detect_sources_on_sum_image(sum_img: np.ndarray):
    bkg_median, bkg_sigma = robust_integer_sigma(sum_img)
    threshold = int(bkg_median + SOURCE_THRESHOLD_SIGMA * bkg_sigma)
    is_local_max = sum_img == maximum_filter(sum_img, size=MAX_FILTER_SIZE, mode="nearest")
    ys, xs = np.where(is_local_max & (sum_img > threshold) & (sum_img > 0))

    rows = []
    for y, x in zip(ys, xs):
        rows.append(
            {
                "x": int(x),
                "y": int(y),
                "source_peak_value": int(sum_img[y, x]),
            }
        )
    rows.sort(key=lambda r: r["source_peak_value"], reverse=True)
    return rows, threshold, bkg_median, bkg_sigma


def template_match_filter_candidates(candidates, template_xy: np.ndarray, match_radius_px: float):
    if template_xy.size == 0:
        return candidates, []

    tree = cKDTree(template_xy)
    kept = []
    dropped = []
    for row in candidates:
        dist, idx = tree.query([row["x"], row["y"]], k=1)
        rec = dict(row)
        rec.update(
            {
                "nearest_template_dist_px": float(dist),
                "nearest_template_x": float(template_xy[idx, 0]),
                "nearest_template_y": float(template_xy[idx, 1]),
            }
        )
        if dist <= match_radius_px:
            dropped.append(rec)
        else:
            kept.append(rec)
    return kept, dropped


def extract_cutout(frame: np.ndarray, x: float, y: float, half: int):
    h, w = frame.shape
    x0 = max(0, int(np.floor(x)) - half)
    x1 = min(w, int(np.floor(x)) + half + 1)
    y0 = max(0, int(np.floor(y)) - half)
    y1 = min(h, int(np.floor(y)) + half + 1)
    return frame[y0:y1, x0:x1], x0, y0


def pick_component_label(labels: np.ndarray, diff_patch: np.ndarray, seed_mask: np.ndarray, seed_mask_fallback: np.ndarray) -> int:
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
    cut_half: int,
    annulus_r_in: float,
    annulus_r_out: float,
    threshold_sigma: int,
    seed_radius: float,
):
    xc, yc = cand_xy
    tmpl_cut, x0, y0 = extract_cutout(template_sum_img, xc, yc, cut_half)
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
    annulus = (rr >= annulus_r_in) & (rr <= annulus_r_out)
    seed_mask = rr <= seed_radius
    seed_mask_fallback = rr <= (seed_radius + 1.0)
    structure = np.ones((3, 3), dtype=int)

    # The only subtraction in this script is local to the small candidate cutout.
    diff = img_cut.astype(LOCAL_DIFF_DTYPE, copy=False) - tmpl_cut.astype(LOCAL_DIFF_DTYPE, copy=False)
    bkg_med, bkg_sigma = robust_integer_sigma(diff[annulus])
    if bkg_sigma <= 0:
        bkg_med, bkg_sigma = robust_integer_sigma(diff)
    if bkg_sigma <= 0:
        return 0, 0, 0, int(bkg_med), int(bkg_sigma)

    threshold = int(bkg_med + threshold_sigma * bkg_sigma)
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
    local_excess_flux = int(max(int(comp_vals.sum(dtype=LOCAL_DIFF_DTYPE)) - int(bkg_med) * npix, 0))
    peak_excess_value = int(comp_vals.max() - int(bkg_med))
    return local_excess_flux, npix, peak_excess_value, int(bkg_med), int(bkg_sigma)


def save_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_target_hits(rows):
    hits = []
    for row in rows:
        dx = float(row["x"]) - TARGET_X
        dy = float(row["y"]) - TARGET_Y
        dist = float(np.hypot(dx, dy))
        if dist <= TARGET_MATCH_RADIUS_PX:
            rec = dict(row)
            rec["target_dist_px"] = dist
            hits.append(rec)
    return hits


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slice_paths = sorted(INPUT_NPY_DIR.glob("*.npy"))
    if not slice_paths:
        raise FileNotFoundError(f"No NPY slices found in {INPUT_NPY_DIR}")

    template_path = slice_paths[0]
    template_sum_img = sum_stack_from_npy(template_path)
    template_sources, template_threshold, template_bkg_median, template_bkg_sigma = detect_sources_on_sum_image(
        template_sum_img
    )
    template_xy = np.asarray([(r["x"], r["y"]) for r in template_sources], dtype=float)

    all_rows = []
    dropped_rows = []
    final_rows = []
    summary_rows = []

    for slice_path in slice_paths:
        slice_idx, frame_start, frame_end = parse_slice_info(slice_path)
        sum_img = sum_stack_from_npy(slice_path)
        if sum_img.shape != template_sum_img.shape:
            raise ValueError(f"Shape mismatch: {slice_path} -> {sum_img.shape}, template -> {template_sum_img.shape}")

        sources, source_threshold, source_bkg_median, source_bkg_sigma = detect_sources_on_sum_image(sum_img)
        kept, dropped = template_match_filter_candidates(sources, template_xy, MATCH_RADIUS_PX)

        measured = []
        for row in kept:
            local_flux, npix, peak_excess, local_bkg_med, local_bkg_sigma = measure_candidate_local_excess(
                sum_img,
                template_sum_img,
                (row["x"], row["y"]),
                cut_half=CUT_HALF,
                annulus_r_in=ANNULUS_R_IN,
                annulus_r_out=ANNULUS_R_OUT,
                threshold_sigma=LOCAL_THRESHOLD_SIGMA,
                seed_radius=SEED_RADIUS,
            )
            peak_pixel_snr = float(peak_excess / (local_bkg_sigma + 1e-12)) if local_bkg_sigma > 0 else np.nan
            rec = dict(row)
            rec.update(
                {
                    "stack_file": slice_path.name,
                    "stack_idx": slice_idx,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "source_threshold": int(source_threshold),
                    "source_bkg_median": int(source_bkg_median),
                    "source_bkg_sigma": int(source_bkg_sigma),
                    "template_file": template_path.name,
                    "local_excess_flux": int(local_flux),
                    "peak_npix": int(npix),
                    "peak_excess_value": int(peak_excess),
                    "local_bkg_median": int(local_bkg_med),
                    "local_bkg_sigma": int(local_bkg_sigma),
                    "peak_pixel_snr": peak_pixel_snr,
                    "effective_npix_threshold": int(EFFECTIVE_NPIX_THRESHOLD),
                    "pass_single_stack": int(npix >= EFFECTIVE_NPIX_THRESHOLD),
                }
            )
            measured.append(rec)

        final = [r for r in measured if r["pass_single_stack"] == 1]
        all_rows.extend(measured)
        dropped_rows.extend(
            [
                {
                    "stack_file": slice_path.name,
                    "stack_idx": slice_idx,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "source_threshold": int(source_threshold),
                    "source_bkg_median": int(source_bkg_median),
                    "source_bkg_sigma": int(source_bkg_sigma),
                    **r,
                }
                for r in dropped
            ]
        )
        final_rows.extend(final)
        summary_rows.append(
            {
                "stack_file": slice_path.name,
                "stack_idx": slice_idx,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "source_threshold": int(source_threshold),
                "source_bkg_median": int(source_bkg_median),
                "source_bkg_sigma": int(source_bkg_sigma),
                "initial_sources": len(sources),
                "dropped_by_template_match": len(dropped),
                "after_template_match": len(measured),
                "final_candidates": len(final),
            }
        )

    after_match_csv = OUTPUT_DIR / "streaming_sum_candidates_after_template_match.csv"
    dropped_csv = OUTPUT_DIR / "streaming_sum_candidates_dropped_by_template_match.csv"
    final_csv = OUTPUT_DIR / "streaming_sum_transient_candidates.csv"
    summary_csv = OUTPUT_DIR / "streaming_sum_summary.csv"

    measured_fields = [
        "stack_file",
        "stack_idx",
        "frame_start",
        "frame_end",
        "x",
        "y",
        "source_peak_value",
        "nearest_template_dist_px",
        "nearest_template_x",
        "nearest_template_y",
        "source_threshold",
        "source_bkg_median",
        "source_bkg_sigma",
        "template_file",
        "local_excess_flux",
        "peak_npix",
        "peak_excess_value",
        "local_bkg_median",
        "local_bkg_sigma",
        "peak_pixel_snr",
        "effective_npix_threshold",
        "pass_single_stack",
    ]
    dropped_fields = [
        "stack_file",
        "stack_idx",
        "frame_start",
        "frame_end",
        "x",
        "y",
        "source_peak_value",
        "nearest_template_dist_px",
        "nearest_template_x",
        "nearest_template_y",
        "source_threshold",
        "source_bkg_median",
        "source_bkg_sigma",
    ]
    summary_fields = [
        "stack_file",
        "stack_idx",
        "frame_start",
        "frame_end",
        "source_threshold",
        "source_bkg_median",
        "source_bkg_sigma",
        "initial_sources",
        "dropped_by_template_match",
        "after_template_match",
        "final_candidates",
    ]

    save_csv(after_match_csv, all_rows, measured_fields)
    save_csv(dropped_csv, dropped_rows, dropped_fields)
    save_csv(final_csv, final_rows, measured_fields)
    save_csv(summary_csv, summary_rows, summary_fields)

    target_hits = find_target_hits(final_rows)

    print(f"Input NPY dir: {INPUT_NPY_DIR}")
    print(f"Template NPY slice: {template_path}")
    print(f"Input bit depth: {INPUT_BIT_DEPTH}")
    print(f"Template sources: {len(template_sources)}")
    print(f"Template source threshold: {template_threshold}")
    print(f"Template background median/sigma: {template_bkg_median}/{template_bkg_sigma}")
    print(f"Slices processed: {len(slice_paths)}")
    print(f"Total candidates after template match: {len(all_rows)}")
    print(f"Total final streaming candidates: {len(final_rows)}")
    print(f"Target check ({TARGET_X}, {TARGET_Y}), radius={TARGET_MATCH_RADIUS_PX}: {'FOUND' if target_hits else 'NOT FOUND'}")
    for hit in target_hits:
        print(
            "Target hit: "
            f"stack={hit['stack_file']} frames={hit['frame_start']}-{hit['frame_end']} "
            f"xy=({hit['x']},{hit['y']}) dist={hit['target_dist_px']:.3f} "
            f"local_excess_flux={hit['local_excess_flux']} peak_npix={hit['peak_npix']} "
            f"peak_excess_value={hit['peak_excess_value']} peak_pixel_snr={hit['peak_pixel_snr']:.3f}"
        )
    print(f"Saved: {after_match_csv}")
    print(f"Saved: {dropped_csv}")
    print(f"Saved: {final_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()

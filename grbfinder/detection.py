from __future__ import annotations

import numpy as np
from scipy.ndimage import find_objects, label, maximum_filter
from scipy.spatial import cKDTree

from .config import LOCAL_DIFF_DTYPE, ScreenerConfig
from .geometry import detector_center_from_bin


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
        bin_y = int(row_origin + y_local)
        bin_x = int(col_origin + x_local)
        if not (row0 <= bin_y < row1 and col0 <= bin_x < col1):
            continue
        x, y = detector_center_from_bin(bin_x, bin_y, cfg.spatial_bin)
        rows.append(
            {
                "x": x,
                "y": y,
                "bin_x": bin_x,
                "bin_y": bin_y,
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
        bin_y = int(row_origin + y_local)
        bin_x = int(col_origin + x_local)
        if not (row0 <= bin_y < row1 and col0 <= bin_x < col1):
            continue
        x, y = detector_center_from_bin(bin_x, bin_y, cfg.spatial_bin)
        rows.append(
            {
                "x": x,
                "y": y,
                "bin_x": bin_x,
                "bin_y": bin_y,
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


def residual_flux_peak_ratio(row: dict) -> float:
    peak = max(int(row.get("residual_peak_value", 0)), 1)
    return float(int(row.get("residual_flux", 0)) / peak)


def passes_residual_prefilter(row: dict, cfg: ScreenerConfig) -> bool:
    if cfg.keep_all_residual_candidates:
        return True
    if int(row.get("residual_peak_value", 0)) < int(cfg.min_residual_peak_value):
        return False
    if int(row.get("residual_flux", 0)) < int(cfg.min_residual_flux):
        return False
    if residual_flux_peak_ratio(row) < float(cfg.min_flux_peak_ratio):
        return False
    return True


def select_final_rows_for_window(rows: list[dict], cfg: ScreenerConfig) -> list[dict]:
    final_rows = [row for row in rows if int(row.get("pass_single_stack", 0)) == 1]
    limit = int(cfg.max_final_candidates_per_window)
    if cfg.keep_all_residual_candidates or limit <= 0 or len(final_rows) <= limit:
        return final_rows
    return sorted(
        final_rows,
        key=lambda row: (
            int(row.get("residual_flux", 0)),
            int(row.get("residual_peak_value", 0)),
        ),
        reverse=True,
    )[:limit]


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


def classify_candidate_priority(row: dict, cfg: ScreenerConfig) -> str:
    if int(row.get("previous_block_match_flag", 0)) == 1:
        return "confirmed_previous_block"
    if int(row.get("temporal_active_frames", 0)) >= int(cfg.temporal_min_active_frames):
        return "confirmed_current_block_temporal"
    if int(row.get("likely_cosmic_ray", 0)) == 1:
        return "low_priority_likely_cosmic"
    if int(row.get("pass_single_stack", 0)) == 1:
        return "single_block_psf_like"
    return "filtered"

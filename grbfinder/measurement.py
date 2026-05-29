from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from .config import LOCAL_DIFF_DTYPE, ScreenerConfig
from .detection import integer_median, pick_component_label, robust_integer_sigma
from .geometry import detection_cutout_from_frame, extract_cutout
from .io import sum_frame_tile


def detector_xy_to_bin_xy(x: int | float, y: int | float, cfg: ScreenerConfig) -> tuple[int, int]:
    return (
        int(np.floor(float(x) / cfg.spatial_bin.cols)),
        int(np.floor(float(y) / cfg.spatial_bin.rows)),
    )


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
    template_frame_start: int,
    template_frame_end: int,
    frame_start: int,
    frame_end: int,
    x: int | float,
    y: int | float,
    cfg: ScreenerConfig,
) -> dict:
    bin_x, bin_y = detector_xy_to_bin_xy(x, y, cfg)
    template_mean = None
    if template_frame_paths is not None and template_frame_start < template_frame_end:
        template_sum = None
        template_count = 0
        for idx in range(template_frame_start, min(template_frame_end, len(template_frame_paths))):
            tmpl = np.load(template_frame_paths[idx], mmap_mode="r")
            tmpl_cut, _, _ = detection_cutout_from_frame(
                tmpl,
                bin_x,
                bin_y,
                cfg.temporal_cut_half,
                cfg.spatial_bin,
            )
            if template_sum is None:
                template_sum = tmpl_cut.astype(np.float64, copy=True)
            elif template_sum.shape == tmpl_cut.shape:
                template_sum += tmpl_cut.astype(np.float64, copy=False)
            else:
                continue
            template_count += 1
        if template_sum is not None and template_count > 0:
            template_mean = template_sum / float(template_count)

    fluxes: list[int] = []
    for idx in range(frame_start, frame_end):
        frame = np.load(frame_paths[idx], mmap_mode="r")
        cut, x0, y0 = detection_cutout_from_frame(
            frame,
            bin_x,
            bin_y,
            cfg.temporal_cut_half,
            cfg.spatial_bin,
        )
        cut_data = cut.astype(np.float64, copy=False)
        if template_mean is not None and template_mean.shape == cut.shape:
            cut_data = cut_data - template_mean

        yy, xx = np.indices(cut.shape)
        rr = np.hypot(xx - float(bin_x - x0), yy - float(bin_y - y0))
        aperture = rr <= cfg.temporal_aperture_radius
        annulus = (rr >= cfg.temporal_annulus_r_in) & (rr <= cfg.temporal_annulus_r_out)
        bkg_med = integer_median(cut_data[annulus]) if np.any(annulus) else integer_median(cut_data)
        flux = int(round(float(cut_data[aperture].sum()) - bkg_med * int(aperture.sum())))
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


def build_temporal_peak_maps_for_tile(
    frame_paths: list[Path],
    frame_start: int,
    frame_end: int,
    template_sum_img: np.ndarray,
    template_frame_count: int,
    row_slice: slice,
    col_slice: slice,
    cfg: ScreenerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if template_frame_count <= 0:
        raise ValueError("template_frame_count must be positive")
    active_count = np.zeros(template_sum_img.shape, dtype=np.uint8)
    positive_total = np.zeros(template_sum_img.shape, dtype=np.int64)
    positive_peak = np.zeros(template_sum_img.shape, dtype=np.int64)
    template_scaled = template_sum_img.astype(LOCAL_DIFF_DTYPE, copy=False)
    for idx in range(frame_start, frame_end):
        current = sum_frame_tile(frame_paths, idx, idx + 1, row_slice, col_slice, cfg.spatial_bin).astype(
            LOCAL_DIFF_DTYPE,
            copy=False,
        )
        diff_scaled = current * int(template_frame_count) - template_scaled
        med, sigma = robust_integer_sigma(diff_scaled)
        if sigma <= 0:
            threshold = max(int(med), 0)
        else:
            threshold = int(med + cfg.temporal_sigma * sigma)
        positive = np.where(diff_scaled > threshold, diff_scaled, 0)
        positive_total += positive
        positive_peak = np.maximum(positive_peak, positive)
        active_count += positive > 0
    return active_count, positive_total, positive_peak


def measure_temporal_support_from_peak_maps(
    active_count: np.ndarray,
    positive_total: np.ndarray,
    positive_peak: np.ndarray,
    local_x: int,
    local_y: int,
    cfg: ScreenerConfig,
) -> dict:
    if (
        local_y < 0
        or local_y >= active_count.shape[0]
        or local_x < 0
        or local_x >= active_count.shape[1]
    ):
        return {
            "temporal_active_frames": 0,
            "temporal_consecutive_active_frames": 0,
            "temporal_max_single_frame_fraction": 0.0,
            "temporal_flux_series": "[]",
            "likely_cosmic_ray": 0,
        }
    total = int(positive_total[local_y, local_x])
    peak = int(positive_peak[local_y, local_x])
    active = int(active_count[local_y, local_x])
    max_fraction = float(peak / total) if total > 0 else 0.0
    likely_cosmic = int(
        active <= cfg.cosmic_max_active_frames
        and max_fraction >= cfg.cosmic_single_frame_fraction
    )
    return {
        "temporal_active_frames": active,
        "temporal_consecutive_active_frames": active,
        "temporal_max_single_frame_fraction": max_fraction,
        "temporal_flux_series": "[]",
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


def annotate_previous_block_matches(
    candidates: list[dict],
    previous_candidates: list[dict],
    radius_px: float,
) -> list[dict]:
    if not previous_candidates:
        out = []
        for row in candidates:
            rec = dict(row)
            rec.update(
                {
                    "previous_block_match_flag": 0,
                    "previous_block_match_dist_px": np.nan,
                    "previous_block_match_frame_start": "",
                    "previous_block_match_frame_end": "",
                    "track_length": 1,
                }
            )
            out.append(rec)
        return out

    prev_xy = np.asarray(
        [(float(row["x"]), float(row["y"])) for row in previous_candidates],
        dtype=float,
    )
    tree = cKDTree(prev_xy)
    out = []
    for row in candidates:
        dist, idx = tree.query([float(row["x"]), float(row["y"])], k=1)
        matched = bool(float(dist) <= float(radius_px))
        prev = previous_candidates[int(idx)]
        rec = dict(row)
        rec.update(
            {
                "previous_block_match_flag": int(matched),
                "previous_block_match_dist_px": float(dist) if matched else np.nan,
                "previous_block_match_frame_start": prev.get("frame_start", "") if matched else "",
                "previous_block_match_frame_end": prev.get("frame_end", "") if matched else "",
                "track_length": int(prev.get("track_length", 1)) + 1 if matched else 1,
            }
        )
        out.append(rec)
    return out

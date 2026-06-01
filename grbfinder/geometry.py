from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import SUM_DTYPE, SpatialBin


def binned_detection_shape(
    frame_shape: tuple[int, int],
    spatial_bin: SpatialBin = SpatialBin(),
) -> tuple[int, int]:
    return int(frame_shape[0]) // spatial_bin.rows, int(frame_shape[1]) // spatial_bin.cols


def detector_center_from_bin(
    bin_x: int | float,
    bin_y: int | float,
    spatial_bin: SpatialBin = SpatialBin(),
) -> tuple[float | int, float | int]:
    x = float(bin_x) * spatial_bin.cols + (spatial_bin.cols - 1) / 2.0
    y = float(bin_y) * spatial_bin.rows + (spatial_bin.rows - 1) / 2.0
    if spatial_bin.rows == 1 and spatial_bin.cols == 1:
        return int(round(x)), int(round(y))
    return x, y


def detection_tile_from_frame(
    frame: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    spatial_bin: SpatialBin = SpatialBin(),
) -> np.ndarray:
    if spatial_bin.rows == 1 and spatial_bin.cols == 1:
        return np.asarray(frame[row_slice, col_slice], dtype=SUM_DTYPE)

    raw_row_slice = slice(int(row_slice.start) * spatial_bin.rows, int(row_slice.stop) * spatial_bin.rows)
    raw_col_slice = slice(int(col_slice.start) * spatial_bin.cols, int(col_slice.stop) * spatial_bin.cols)
    raw = np.asarray(frame[raw_row_slice, raw_col_slice], dtype=SUM_DTYPE)
    out_h = int(row_slice.stop) - int(row_slice.start)
    out_w = int(col_slice.stop) - int(col_slice.start)
    expected_shape = (out_h * spatial_bin.rows, out_w * spatial_bin.cols)
    if raw.shape != expected_shape:
        raise ValueError(
            f"cannot block-bin raw tile shape {raw.shape} into "
            f"{out_h}x{out_w} blocks of {spatial_bin.rows}x{spatial_bin.cols}"
        )
    return raw.reshape(out_h, spatial_bin.rows, out_w, spatial_bin.cols).sum(axis=(1, 3), dtype=SUM_DTYPE)


def detection_cutout_from_frame(
    frame: np.ndarray,
    x: int | float,
    y: int | float,
    half: int,
    spatial_bin: SpatialBin = SpatialBin(),
) -> tuple[np.ndarray, int, int]:
    det_h, det_w = binned_detection_shape((int(frame.shape[0]), int(frame.shape[1])), spatial_bin)
    x0 = max(0, int(np.floor(x)) - half)
    x1 = min(det_w, int(np.floor(x)) + half + 1)
    y0 = max(0, int(np.floor(y)) - half)
    y1 = min(det_h, int(np.floor(y)) + half + 1)
    tile = detection_tile_from_frame(frame, slice(y0, y1), slice(x0, x1), spatial_bin)
    return tile, x0, y0


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


def iter_detection_tiles(
    shape: tuple[int, int],
    tile_size: int,
    use_tiles: bool = True,
) -> Iterable[tuple[int, int, int, int]]:
    if not use_tiles:
        n_rows, n_cols = shape
        yield 0, n_rows, 0, n_cols
        return
    yield from iter_core_tiles(shape, tile_size)


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


def candidate_local_xy(row: dict, *, row_slice: slice, col_slice: slice) -> tuple[int, int]:
    x = int(row.get("bin_x", row["x"]))
    y = int(row.get("bin_y", row["y"]))
    return x - int(col_slice.start), y - int(row_slice.start)


def extract_cutout(frame: np.ndarray, x: float, y: float, half: int):
    h, w = frame.shape
    x0 = max(0, int(np.floor(x)) - half)
    x1 = min(w, int(np.floor(x)) + half + 1)
    y0 = max(0, int(np.floor(y)) - half)
    y1 = min(h, int(np.floor(y)) + half + 1)
    return frame[y0:y1, x0:x1], x0, y0

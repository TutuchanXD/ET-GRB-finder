from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import SUM_DTYPE, SpatialBin
from .geometry import detection_tile_from_frame


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


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_csv_header(path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def sum_frame_tile(
    frame_paths: list[Path],
    frame_start: int,
    frame_end: int,
    row_slice: slice,
    col_slice: slice,
    spatial_bin: SpatialBin = SpatialBin(),
) -> np.ndarray:
    if frame_start < 0 or frame_end > len(frame_paths) or frame_start >= frame_end:
        raise ValueError(f"invalid frame window {frame_start}:{frame_end}")
    out = None
    for path in frame_paths[frame_start:frame_end]:
        frame = np.load(path, mmap_mode="r")
        tile = detection_tile_from_frame(frame, row_slice, col_slice, spatial_bin)
        if out is None:
            out = tile.copy()
        else:
            out += tile
    return out


def default_truth_path(input_run: Path) -> Path | None:
    path = input_run / "events.csv"
    return path if path.is_file() else None

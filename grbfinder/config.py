from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_INPUT_RUN = Path(
    "/home/cxgao/Results/GRB/grb_injected/main_rd_g17_120x10s_grb_seed20260529"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/cxgao/Results/GRB/grb_search/main_rd_g17_120x10s_grb_seed20260529_sum12_streaming"
)

SUM_DTYPE = np.uint32
LOCAL_DIFF_DTYPE = np.int64


@dataclass(frozen=True)
class SpatialBin:
    rows: int = 1
    cols: int = 1

    def __post_init__(self) -> None:
        if int(self.rows) <= 0 or int(self.cols) <= 0:
            raise ValueError(f"spatial bin dimensions must be positive, got {self.rows}x{self.cols}")
        object.__setattr__(self, "rows", int(self.rows))
        object.__setattr__(self, "cols", int(self.cols))

    @property
    def legacy_size(self) -> int | None:
        return self.rows if self.rows == self.cols else None

    @classmethod
    def square(cls, size: int) -> "SpatialBin":
        return cls(size, size)

    def __str__(self) -> str:
        return f"{self.rows}x{self.cols}"


@dataclass(frozen=True)
class ScreenerConfig:
    spatial_bin: SpatialBin = SpatialBin()
    window_size: int = 12
    stride: int = 12
    use_tiles: bool = True
    tile_size: int = 1024
    halo: int = 12
    input_bit_depth: int = 16
    max_filter_size: int = 7
    source_threshold_sigma: float = 4.0
    residual_threshold_sigma: float = 3.0
    residual_min_npix: int = 2
    residual_max_npix: int = 400
    min_residual_peak_value: int = 10000
    min_residual_flux: int = 0
    min_flux_peak_ratio: float = 3.0
    max_final_candidates_per_window: int = 5000
    match_radius_px: float = 0.75
    cut_half: int = 9
    annulus_r_in: float = 6.0
    annulus_r_out: float = 10.0
    local_threshold_sigma: float = 3.0
    seed_radius: float = 1.5
    effective_npix_threshold: int = 4
    peak_pixel_snr_check: bool = False
    peak_pixel_snr_threshold: float = 5.0
    temporal_cut_half: int = 5
    temporal_aperture_radius: float = 3.0
    temporal_annulus_r_in: float = 5.0
    temporal_annulus_r_out: float = 8.0
    temporal_sigma: float = 3.0
    temporal_min_active_frames: int = 2
    cosmic_single_frame_fraction: float = 0.80
    cosmic_max_active_frames: int = 1
    previous_block_match_check: bool = False
    previous_match_radius_px: float = 2.0
    local_shape_check: bool = True
    temporal_check: bool = True
    keep_all_residual_candidates: bool = False

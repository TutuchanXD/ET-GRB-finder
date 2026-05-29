import pytest

from grbfinder.cli import parse_args, parse_spatial_bin
from grbfinder.config import SpatialBin
import numpy as np

from grbfinder.geometry import (
    binned_detection_shape,
    detection_tile_from_frame,
    detector_center_from_bin,
)


def test_parse_spatial_bin_accepts_square_and_rectangular_values():
    assert parse_spatial_bin("3") == SpatialBin(3, 3)
    assert parse_spatial_bin("3x4") == SpatialBin(3, 4)
    assert parse_spatial_bin("3X4") == SpatialBin(3, 4)


def test_parse_spatial_bin_rejects_invalid_values():
    for value in ["0", "-1", "3x0", "3x", "x4", "3*4", "abc"]:
        with pytest.raises(ValueError):
            parse_spatial_bin(value)


def test_parse_args_rejects_invalid_spatial_bin_as_argparse_error():
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--spatial-bin", "abc"])

    assert excinfo.value.code == 2


def test_rectangular_spatial_bin_geometry():
    spatial_bin = SpatialBin(3, 4)

    assert binned_detection_shape((9121, 8903), spatial_bin) == (3040, 2225)
    assert detector_center_from_bin(2, 3, spatial_bin) == (9.5, 10.0)


def test_rectangular_spatial_bin_tile_sums_blocks():
    spatial_bin = SpatialBin(3, 4)
    frame = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)

    tile = detection_tile_from_frame(frame, slice(0, 2), slice(0, 2), spatial_bin)

    assert tile.shape == (2, 2)
    assert tile.dtype == np.uint32
    assert tile[0, 0] == int(frame[0:3, 0:4].sum())
    assert tile[0, 1] == int(frame[0:3, 4:8].sum())
    assert tile[1, 0] == int(frame[3:6, 0:4].sum())
    assert tile[1, 1] == int(frame[3:6, 4:8].sum())

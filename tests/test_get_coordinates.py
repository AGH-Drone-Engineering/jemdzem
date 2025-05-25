import math

import numpy as np
import pytest

from modules import get_coordinates


@pytest.mark.parametrize("deg", [0, -15.5, 37.7749, 179.9999])
def test_round_trip_degrees_to_dms_and_back(deg: float) -> None:
    d, m, s = get_coordinates.degrees_to_d_m_s(deg)
    result = get_coordinates.d_m_s_to_degrees(d, m, s)
    assert math.isclose(result, deg, rel_tol=0, abs_tol=1e-6)


def make_camera():
    # Camera parameters roughly matching the ones used in module examples
    img_width = 5472.0
    img_height = 3648.0
    fx = fy = 4955.0
    cx = img_width / 2.0
    cy = img_height / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)
    center = (cx, cy)
    return camera_matrix, dist_coeffs, center


def test_translate_center_near_zero() -> None:
    camera_matrix, dist_coeffs, center = make_camera()
    altitude = 100.0
    x, y = center

    dx, dy = get_coordinates.translate(x, y, altitude, camera_matrix, dist_coeffs)

    assert abs(dx) < 1e-6
    assert abs(dy) < 1e-6


def test_calculate_new_coordinates() -> None:
    camera_matrix, dist_coeffs, center = make_camera()
    altitude = 100.0
    lat, lng = 50.0, 19.0

    # Zero deltas should preserve coordinates
    new_lat, new_lng = get_coordinates.calculate_new_coordinates(lat, lng, 0.0, 0.0)
    assert math.isclose(new_lat, lat, abs_tol=1e-6)
    assert math.isclose(new_lng, lng, abs_tol=1e-6)

    # Point offset from centre
    x_off = center[0] + 100
    y_off = center[1] + 50

    dx, dy = get_coordinates.translate(x_off, y_off, altitude, camera_matrix, dist_coeffs)
    off_lat, off_lng = get_coordinates.calculate_new_coordinates(lat, lng, dx, dy)

    assert not math.isclose(off_lat, lat, abs_tol=1e-6)
    assert not math.isclose(off_lng, lng, abs_tol=1e-6)

    # Expect movement south (negative dy) => latitude decreases
    assert off_lat < lat
    # Expect movement east (positive dx) => longitude increases
    assert off_lng > lng

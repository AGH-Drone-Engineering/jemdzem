import math
from geographiclib.geodesic import Geodesic
import numpy as np
import pytest
from typing import Tuple

from modules import get_coordinates


@pytest.mark.parametrize("deg", [0, -15.5, 37.7749, 179.9999, 179.999999])
def test_round_trip_degrees_to_dms_and_back(deg: float) -> None:
    sign, d, m, s = get_coordinates.degrees_to_d_m_s(deg)
    result = get_coordinates.d_m_s_to_degrees(sign, d, m, s)
    assert math.isclose(result, deg, rel_tol=0, abs_tol=1e-6)


def test_seconds_rounding_carry() -> None:
    deg = 179.9999999999
    sign, d, m, s = get_coordinates.degrees_to_d_m_s(deg)
    assert (sign, d, m, s) == (1, 180, 0, 0.0)
    back = get_coordinates.d_m_s_to_degrees(sign, d, m, s)
    assert math.isclose(back, deg, rel_tol=0, abs_tol=1e-6)


def test_sign_handling() -> None:
    sign, d, m, s = get_coordinates.degrees_to_d_m_s(-15.5)
    assert (sign, d, m, s) == (-1, 15, 30, 0)

    result = get_coordinates.d_m_s_to_degrees(sign, d, m, s)
    assert math.isclose(result, -15.5, rel_tol=0, abs_tol=1e-6)


@pytest.mark.parametrize(
    "deg, expected",
    [
        (0.5, (1, 0, 30, 0)),
        (-0.5, (-1, 0, 30, 0)),
        (0.25, (1, 0, 15, 0)),
        (-0.75, (-1, 0, 45, 0)),
    ],
)
def test_under_one_degree(deg: float, expected: Tuple[int, int, int, int]) -> None:
    result = get_coordinates.degrees_to_d_m_s(deg)
    assert result == expected
    back = get_coordinates.d_m_s_to_degrees(*result)
    assert math.isclose(back, deg, rel_tol=0, abs_tol=1e-6)


def make_camera() -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Return camera intrinsics used by the translate tests.

    The returned ``(camera_matrix, dist_coeffs, center)`` tuple approximates
    the parameters from the example modules. ``center`` is the principal point
    referenced by ``translate`` and ``calculate_new_coordinates``.
    """
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


def test_translate_offset_not_zero() -> None:
    camera_matrix, dist_coeffs, center = make_camera()
    altitude = 100.0

    x_off = center[0] + 100
    y_off = center[1] + 50

    dx, dy = get_coordinates.translate(x_off, y_off, altitude, camera_matrix, dist_coeffs)

    assert not math.isclose(dx, 0.0, abs_tol=1e-6) or not math.isclose(dy, 0.0, abs_tol=1e-6)


def test_translate_rotation() -> None:
    camera_matrix, dist_coeffs, center = make_camera()
    altitude = 100.0

    x_off = center[0] + 100
    y_off = center[1] + 50
    angle = math.radians(30)

    dx0, dy0 = get_coordinates.translate(
        x_off, y_off, altitude, camera_matrix, dist_coeffs
    )
    dx_rot, dy_rot = get_coordinates.translate(
        x_off, y_off, altitude, camera_matrix, dist_coeffs, angle
    )

    expected_dx = dx0 * math.cos(angle) + dy0 * math.sin(angle)
    expected_dy = dy0 * math.cos(angle) - dx0 * math.sin(angle)

    assert math.isclose(dx_rot, expected_dx, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(dy_rot, expected_dy, rel_tol=0, abs_tol=1e-6)


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


# Base coordinate for tests
LAT = 50.0
LNG = 19.0

@pytest.mark.parametrize(
    "dx, dy, lat_rel, lng_rel",
    [
        (0.0, 10.0, 1, 0),   # north -> lat increases
        (0.0, -10.0, -1, 0), # south -> lat decreases
        (10.0, 0.0, 0, 1),   # east -> lon increases
        (-10.0, 0.0, 0, -1), # west -> lon decreases
    ],
)
def test_cardinal_directions(dx: float, dy: float, lat_rel: int, lng_rel: int) -> None:
    new_lat, new_lng = get_coordinates.calculate_new_coordinates(LAT, LNG, dx, dy)

    if lat_rel == 0:
        assert math.isclose(new_lat, LAT, abs_tol=1e-6)
    elif lat_rel > 0:
        assert new_lat > LAT
    else:
        assert new_lat < LAT

    if lng_rel == 0:
        assert math.isclose(new_lng, LNG, abs_tol=1e-6)
    elif lng_rel > 0:
        assert new_lng > LNG
    else:
        assert new_lng < LNG

    inv = Geodesic.WGS84.Inverse(LAT, LNG, new_lat, new_lng)
    expected_dist = math.hypot(dx, dy)
    assert math.isclose(inv["s12"], expected_dist, rel_tol=0, abs_tol=1e-6)


def test_distance_and_bearing() -> None:
    dx, dy = 30.0, 40.0  # 3-4-5 triangle -> distance 50
    new_lat, new_lng = get_coordinates.calculate_new_coordinates(LAT, LNG, dx, dy)

    inv = Geodesic.WGS84.Inverse(LAT, LNG, new_lat, new_lng)
    assert math.isclose(inv["s12"], 50.0, rel_tol=0, abs_tol=1e-6)

    expected_az = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    result_az = (inv["azi1"] + 360) % 360
    assert math.isclose(result_az, expected_az, rel_tol=0, abs_tol=1e-6)


@pytest.mark.parametrize(
    "pixels, expected",
    [
        (0, 0.0),
        (1, 2.0),
        (5, 10.0),
        (10, 20.0),
        (50, 100.0),
    ],
)
def test_pixels_to_meters_basic(pixels: int, expected: float) -> None:
    assert math.isclose(
        get_coordinates.pixels_to_meters(pixels, altitude=100.0, focal_length_px=50.0),
        expected,
        rel_tol=0,
        abs_tol=1e-6,
    )

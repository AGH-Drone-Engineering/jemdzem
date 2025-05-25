import math
import pytest
from geographiclib.geodesic import Geodesic

from modules import get_coordinates

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

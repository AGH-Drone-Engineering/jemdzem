import math
from geographiclib.geodesic import Geodesic
import cv2
import numpy as np
from typing import Tuple


def degrees_to_d_m_s(degrees: float) -> Tuple[int, int, int, float]:
    """Convert decimal degrees to a ``(sign, degrees, minutes, seconds)`` tuple.

    Args:
        degrees: Geographic coordinate in decimal degrees.

    Returns:
        A tuple ``(sign, d, m, s)`` where ``sign`` is either ``1`` or ``-1`` and
        ``d`` and ``m`` are the absolute degree and minute components. ``s`` is
        the seconds component. ``d``, ``m`` and ``s`` are always non-negative.
    """

    sign = -1 if degrees < 0 else 1
    abs_deg = abs(degrees)

    d = int(abs_deg)
    minutes_decimal = (abs_deg - d) * 60
    m = int(minutes_decimal)
    s = (minutes_decimal - m) * 60

    # Round seconds to fixed precision to avoid floating point noise
    s = round(s, 6)

    # Handle carry-over from rounding
    if s >= 60.0:
        s = 0.0
        m += 1

    if m >= 60:
        m = 0
        d += 1

    return sign, d, m, s


def d_m_s_to_degrees(sign: int, d: int, m: int, s: float) -> float:
    """Convert a ``(sign, degrees, minutes, seconds)`` tuple to decimal degrees.

    Args:
        sign: ``1`` or ``-1`` specifying the hemisphere.
        d: Degrees component (non‐negative).
        m: Minutes component (non‐negative).
        s: Seconds component (non‐negative).

    Returns:
        Geographic coordinate expressed in decimal degrees.
    """

    deg_abs = d + (m / 60) + (s / 3600)
    return sign * deg_abs


def pixels_to_meters(
    pixels: float,
    altitude: float,
    focal_length_px: float,
) -> float:
    """Convert a distance in pixels to meters using camera intrinsics.

    Args:
        pixels: Distance in image pixels.
        altitude: Altitude of the camera above the ground in meters.
        focal_length_px: Focal length in *pixels* (fx or fy from the camera matrix).

    Returns:
        Distance expressed in meters.
    """
    meters_per_pixel = altitude / focal_length_px
    return pixels * meters_per_pixel


def translate(
    x: float,
    y: float,
    altitude: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    angle: float = 0.0,
) -> Tuple[float, float]:
    """Calculate the metric offset from the image centre using camera intrinsics.

    Args:
        x: X pixel coordinate of the detected object.
        y: Y pixel coordinate of the detected object.
        altitude: Altitude of the camera in meters.
        camera_matrix: 3x3 intrinsic camera matrix ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
        dist_coeffs: Distortion coefficients for the lens.
        angle: Optional rotation of the image around the optical axis (radians).

    Returns:
        ``delta_x_meters`` and ``delta_y_meters`` describing the offset from the
        image centre in metres.
    """

    pts = np.array([[[x, y]]], dtype=np.float32)
    norm = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)
    xn, yn = norm[0, 0]

    # Coordinates on the ground plane assuming the camera looks straight down.
    X = xn * altitude
    Y = yn * altitude

    # Apply optional rotation around the Z axis.
    Xr = X * math.cos(angle) - Y * math.sin(angle)
    Yr = X * math.sin(angle) + Y * math.cos(angle)

    delta_x_meters = Xr
    # Image Y coordinates increase downward, so negate the value so that
    # positive ``delta_y_meters`` corresponds to northward movement.
    delta_y_meters = -Yr

    return delta_x_meters, delta_y_meters


def calculate_new_coordinates(
    lat: float,  # lateral position of drone
    lng: float,  # longitudinal position of drone
    delta_x_meters: float,  # meters distance from middle on x axis
    delta_y_meters: float,  # meters distance from middle on y axis
) -> Tuple[float, float]:  # tuple containing the object's latitude and longitude
    """Compute the object's latitude and longitude from metric offsets.

    The function uses ``geographiclib`` to translate the planar offsets
    from the drone position into WGS84 coordinates.

    Args:
        lat: Current latitude of the drone in decimal degrees.
        lng: Current longitude of the drone in decimal degrees.
        delta_x_meters: Offset from image centre along the x-axis in meters.
        delta_y_meters: Offset from image centre along the y-axis in meters.

    Returns:
        A tuple ``(new_lat, new_lng)`` with the coordinates of the detected
        object in decimal degrees.
    """
    geod = Geodesic.WGS84

    # Compute the total offset distance in the ground plane
    distance = math.hypot(delta_x_meters, delta_y_meters)

    if math.isclose(distance, 0.0, abs_tol=1e-9):
        return lat, lng

    # Bearing from North (0°) clockwise to East (90°)
    azimuth_deg = math.degrees(math.atan2(delta_x_meters, delta_y_meters))

    res = geod.Direct(lat, lng, azimuth_deg, distance)
    new_lat = res["lat2"]
    new_lng = res["lon2"]
    return new_lat, new_lng

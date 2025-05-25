import math
from geographiclib.geodesic import Geodesic
from typing import Tuple
import numpy as np


def calibration(calib_data_path: str) -> Tuple[np.ndarray]:
    """
    Function with give path to camera calibration returns tuple
    of calibration data

    Args:
        calib_data_path - path to calibration file .npz

    Returns:
        cam_mat, dist_coef, r_vectors, t_vectors - tuple of
        numpy.ndarrays for further calibration and length
        estimation
    """
    calib_data = np.load(calib_data_path)
    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]
    r_vectors = calib_data["rVector"]
    t_vectors = calib_data["tVector"]

    return cam_mat, dist_coef, r_vectors, t_vectors


def degrees_to_d_m_s(degrees: float) -> Tuple[int, float]:
    """
    Function converts decimal degrees to degrees, minutes, and seconds.

    Args:
        degrees - floating value of geo postion in one var

    Returns:
        d, m, s - tuple of degrees, minutes and seconds that are (int, int, float)
    """
    d = int(degrees)
    minutes_decimal = (degrees - d) * 60
    m = int(minutes_decimal)
    s = (minutes_decimal - m) * 60
    return d, m, s


def d_m_s_to_degrees(d: int, m: int, s: float) -> float:
    """
    Function converts degrees, minutes, and seconds to decimal degrees.

    Args:
        d - degrees
        m - minutes
        s - seconds

    Returns:
        degrees - floating value of geo postion in one var
    """
    degrees = d + (m / 60) + (s / 3600)
    return degrees


def pixels_to_meters(
    pixels: int,  # distance in pixels that will be converted to meters
    altitude: float,  # drone altitude
    img_width: int,  # image width in pixels
    cam_mat: np.ndarray = None,  # calibration camera matrix
    focal_length: float = None,  # focal length (camera dependency)
    sensor_width: float = None,  # sensor width (camera dependency)
) -> float:  # meters received from pixels
    """
    Function estimates the meter distance based on pixel distance
    and parameters that affects scaling.

    Args:
        pixels - distance in pixels that needs to be converted to meters
        altitude - altitude of drone in meters
        img_width - image width of a frame in pixels
        cam_mat - calibration camera matrix created by testing camera with
        chessboard squares (https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
        focal_length - camera parameter provided by manufacturer default None because cam_mat is used
        sensor_width - camera parameter provided by manufacturer default None because cam_mat is used
    Returns:
        meters - converted pixels to the meters unit
    """
    if cam_mat is None:
        meters_per_pixel = (altitude * sensor_width) / (focal_length * img_width)
    else:
        fx = cam_mat[0][0]
        meters_per_pixel = altitude / fx
    meters = pixels * meters_per_pixel
    return meters


def translate(
    x: float,
    y: float,
    altitude: float,
    angle=0.0,
    img_height=3648.0,
    img_width=5472.0,
    cam_mat=None,
    focal_length=None,
    sensor_width=None,
):
    """
    Function calculates the offset in meters from the image center for a detected object,
    based on its pixel position, drone altitude, camera parameters, and optional image rotation.


    Args:
        x - x coordinate (in meters or pixels, depending on context) of the detected object
        y - y coordinate (in meters or pixels, depending on context) of the detected object
        altitude - altitude of the drone in meters
        angle - rotation angle of the image (radians), default 0.0
        img_height - image height in pixels, default 3648.0
        img_width - image width in pixels, default 5472.0
        cam_mat - calibration camera matrix created by testing camera with
        chessboard squares (https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
        focal_length - camera focal length, default None because cam_mat is used
        sensor_width - camera sensor width, default None because cam_mat is used

    Returns:
        delta_x_meters: Offset in meters from the image center along the X axis.
        delta_y_meters: Offset in meters from the image center along the Y axis.
    """
    x_n = x * math.cos(angle) - y * math.sin(angle)
    y_n = x * math.sin(angle) + y * math.cos(angle)
    x_center = pixels_to_meters(
        img_width / 2.0,
        altitude,
        img_width,
        cam_mat=cam_mat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )
    y_center = pixels_to_meters(
        img_height / 2.0,
        altitude,
        img_width,
        cam_mat=cam_mat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )
    delta_x_meters = x_n - x_center
    delta_y_meters = -(y_n - y_center)

    return delta_x_meters, delta_y_meters


def calculate_new_coordinates(
    lat: float,  # lateral position of drone
    lng: float,  # longitudinal position of drone
    delta_x_meters: float,  # meters distance from middle on x axis
    delta_y_meters: float,  # meters distance from middle on y axis
) -> Tuple[float]:  # tuple of lateral and longitudinal positions of detected object
    """
    Function uses geographiclib library to estimate geo position of
    detected object.

    Args:
        lat - lateral position of drone in geo notation (degrees)
        lng - longitudinal position of drone in geo notation (degrees)
        delta_x_meters - distance from the middle on x axis with meters
        delta_y_meters - distance from the middle on y axis with meters
    Returns:
        new_lat, new_lng - lateral and longitudinal position of detected object (degrees)
    """
    geod = Geodesic.WGS84
    result_lat = geod.Direct(lat, lng, 0, delta_y_meters)
    new_lat = result_lat["lat2"]
    lat_rad = math.radians(lat)
    result_lon = geod.Direct(lat, lng, 90, delta_x_meters * math.cos(lat_rad))
    new_lng = result_lon["lon2"]
    return new_lat, new_lng


def unit_test_get_coordinates():
    """
    Unit test for pixels_to_meters, translate, and calculate_new_coordinates functions.
    Checks if the coordinate transformation pipeline works as expected.

    (Generated by AI because lack of test data)
    """
    # Example parameters
    x_pixel = 2736  # center of image width for 5472px
    y_pixel = 1824  # center of image height for 3648px
    altitude = 100.0  # meters
    lat = 50.0
    lng = 19.0
    angle = 0.0
    img_height = 3648.0
    img_width = 5472.0
    focal_length = 23.0
    sensor_width = 25.4

    cam_mat, dist_coef, r_vectors, t_vectors = calibration(
        "../camera_dependencies/Yuneec.npz"
    )

    # Test pixels_to_meters for center (should be 0 offset)
    meters_x = pixels_to_meters(
        x_pixel,
        altitude,
        img_width,
        cam_mat=cam_mat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )
    meters_y = pixels_to_meters(
        y_pixel,
        altitude,
        img_width,
        cam_mat = cam_mat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )
    print(f"meters_x: {meters_x}, meters_y: {meters_y}")

    # Test translate for center (should be close to 0,0)
    delta_x, delta_y = translate(
        meters_x,
        meters_y,
        altitude,
        angle,
        img_height,
        img_width,
        cam_mat=cam_mat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )
    print(f"delta_x: {delta_x}, delta_y: {delta_y}")
    assert abs(delta_x) < 1e-6, "delta_x should be near zero for image center"
    assert abs(delta_y) < 1e-6, "delta_y should be near zero for image center"

    # Test calculate_new_coordinates (should return original lat/lng for center)
    new_lat, new_lng = calculate_new_coordinates(lat, lng, delta_x, delta_y)
    print(f"new_lat: {new_lat}, new_lng: {new_lng}")
    assert abs(new_lat - lat) < 1e-6, "Latitude should not change for image center"
    assert abs(new_lng - lng) < 1e-6, "Longitude should not change for image center"

    # Test for a point away from center
    x_pixel2 = 3000
    y_pixel2 = 2000
    meters_x2 = pixels_to_meters(
        x_pixel2,
        altitude,
        img_width,
        cam_mat=cam_mat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )
    meters_y2 = pixels_to_meters(
        y_pixel2,
        altitude,
        img_width,
        cam_mat=cam_mat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )

    delta_x2, delta_y2 = translate(
        meters_x2,
        meters_y2,
        altitude,
        angle,
        img_height,
        img_width,
        cam_mat=cam_mat,
        focal_length=focal_length,
        sensor_width=sensor_width,
    )
    new_lat2, new_lng2 = calculate_new_coordinates(lat, lng, delta_x2, delta_y2)
    print(f"Offset lat/lng: {new_lat2}, {new_lng2}")

    print("All tests passed.")


if __name__ == "__main__":
    # Define params for Yuneec camera
    (IMG_HEIGHT, IMG_WIDTH) = (3648.0, 5472.0)
    SENSOR_WIDTH = 25.4
    FOCAL_LENGTH = 23.0

    # Zero step: call d_m_s_to_degrees() if the geo position is in degrees/minutes/seconds and not in
    # degrees only (Yuneec is storing that informations as deg/min/sec in the metadata of it's pictures).
    # Fist call pixels_to_meters() to rescale position of detected object to the meter unit value
    # Then translate() it to the delta distance from the middle
    # On the end use calculate_new_coordinates() to receive new lateral and longitudinal positions
    unit_test_get_coordinates()

import argparse
import cv2
import datetime
import os
import piexif
import rosbag

from cv_bridge import CvBridge
from fractions import Fraction
from libxmp import XMPFiles, XMPMeta
from PIL import Image as PILImage
from pyproj import Transformer, CRS
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image

def add_exif_xmp_tags(image_filename, exif_data, xmp_data):
    # Add XMP data
    xmp_file = XMPFiles(file_path=image_filename, open_forupdate=True)
    xmp = XMPMeta()

    # Register the Pix4D namespace
    pix4d_ns = "http://pix4d.com/camera/1.0/"
    XMPMeta.register_namespace(pix4d_ns, "Camera")

    # Set XMP properties under the Pix4D namespace
    for key, value in xmp_data.items():
        xmp.set_property(pix4d_ns, key, value)

    xmp_file.put_xmp(xmp)
    xmp_file.close_file()

    # Convert EXIF data dictionary to bytes
    exif_bytes = piexif.dump(exif_data)
    piexif.insert(exif_bytes, image_filename)

def adjust_brightness(image, brightness_factor):
    """
    Adjusts the brightness of the image by multiplying the pixel values by the given factor.
    """
    # Ensure the factor is a float and perform the brightness adjustment
    brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return brightened_image

def extract_images_from_bag(bag_file):
    # Create a CvBridge object for converting ROS Image messages to OpenCV images
    bridge = CvBridge()

    # Create output directory based on the bag file name
    output_dir = os.path.splitext(bag_file)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the bag file
    print(f"Processing bag {bag_file}")

    current_lat, current_lon, current_alt = None, None, None
    # ETH ML, but honestly, doesn't matter so much. Relative position is important
    origin_lat, origin_lon, origin_alt = 47.377757, 8.547512, 452.9
    current_roll_deg, current_pitch_deg, current_yaw_deg = 0.0, 0.0, 0.0

    image_count = -1

    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate through the messages in the bag
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['/image_raw', '/rio/odometry_navigation'])):
            if topic == '/image_raw':
                image_count += 1
                if image_count % 3 > 0:
                    continue

                # Convert the ROS Image message to a CV2 image
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Adjust the brightness of the image
                # cv_image = adjust_brightness(cv_image, 3.0)

                # Generate image filename based on timestamp
                image_filename = os.path.join(output_dir, f"{t.to_nsec()}.jpg")

                # Save the image to the specified directory
                cv2.imwrite(image_filename, cv_image)

                # Convert ROS timestamp to datetime format
                timestamp = datetime.datetime.fromtimestamp(msg.header.stamp.to_sec())
                datetime_original = timestamp.strftime("%Y:%m:%d %H:%M:%S")

                # Prepare EXIF data with correct types
                exif_data = {
                    "0th": {
                        piexif.ImageIFD.Make: "FLIR",
                        piexif.ImageIFD.Model: "FireflyS"
                    },
                    "Exif": {
                        piexif.ExifIFD.FocalLength: (36, 10),  # Focal length 3.6mm as a rational number (36/10)
                        piexif.ExifIFD.DateTimeOriginal: datetime_original,
                        piexif.ExifIFD.FocalPlaneXResolution: (1000000, 3450),  # 1 / 3.45 as a rational number
                        piexif.ExifIFD.FocalPlaneYResolution: (1000000, 3450),  # 1 / 3.45 as a rational number
                        piexif.ExifIFD.FocalPlaneResolutionUnit: 3  # 3 represents "centimetres"
                    }
                }

                # Add GPS info to the EXIF data if coordinates are available
                if current_lat is not None and current_lon is not None and current_alt is not None:
                    lat_deg = to_deg(current_lat, ["S", "N"])
                    lng_deg = to_deg(current_lon, ["W", "E"])

                    exiv_lat = (change_to_rational(lat_deg[0]), change_to_rational(lat_deg[1]), change_to_rational(lat_deg[2]))
                    exiv_lng = (change_to_rational(lng_deg[0]), change_to_rational(lng_deg[1]), change_to_rational(lng_deg[2]))

                    gps_ifd = {
                        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
                        piexif.GPSIFD.GPSAltitudeRef: 0,
                        piexif.GPSIFD.GPSAltitude: change_to_rational(round(current_alt)),
                        piexif.GPSIFD.GPSLatitudeRef: lat_deg[3],
                        piexif.GPSIFD.GPSLatitude: exiv_lat,
                        piexif.GPSIFD.GPSLongitudeRef: lng_deg[3],
                        piexif.GPSIFD.GPSLongitude: exiv_lng,
                    }
                    exif_data["GPS"] = gps_ifd

                # Prepare XMP data with the correct Pix4D namespace
                focal_length_mm = (1200.4311769445228 + 1201.8315992165312) / 2.0 * 0.00345
                principal_point_x = 634.1037111885645 * 0.00345
                principal_point_y = 432.2169659507848 * 0.00345
                xmp_data = {
                    "ModelType": "perspective",
                    "PrincipalPoint": f"{principal_point_x}, {principal_point_y}",  # Principal point (in mm)
                    "PerspectiveFocalLength": f"{focal_length_mm}",
                    "PerspectiveDistortion": "0,0,0,0,0",
                    "Roll": f"{current_roll_deg}",
                    "Pitch": f"{current_pitch_deg}",
                    "Yaw": f"{current_yaw_deg}",
                    "GPSXYAccuracy": f"{0.5}",
                    "GPSZAccuracy": f"{0.5}",
                    "IMUYawAccuracy": f"{5.0}",
                    "IMUPitchAccuracy": f"{5.0}",
                    "IMURollAccuracy": f"{5.0}",
                    "HorizCS": "EPSG:4326",
                    "VertCS": "ellipsoidal"
                }

                # Add EXIF and XMP tags to the image
                add_exif_xmp_tags(image_filename, exif_data, xmp_data)

            elif topic == '/rio/odometry_navigation':
                position = msg.pose.pose.position
                current_lat, current_lon, current_alt = cartesian_to_geodetic(position.x, position.y, position.z, origin_lat, origin_lon, origin_alt)
                orientation = msg.pose.pose.orientation
                r = R.from_quat((orientation.x, orientation.y, orientation.z, orientation.w))
                euler = r.as_euler('XYZ', degrees=True)
                current_roll_deg = euler[0]
                current_pitch_deg = euler[1]
                current_yaw_deg = euler[2]


def to_deg(value, loc):
    """convert decimal coordinates into degrees, munutes and seconds tuple
    Keyword arguments: value is float gps-value, loc is direction list ["S", "N"] or ["W", "E"]
    return: tuple like (25, 13, 48.343 ,'N')
    """
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg =  int(abs_value)
    t1 = (abs_value-deg)*60
    min = int(t1)
    sec = round((t1 - min)* 60, 5)
    return (deg, min, sec, loc_value)


def change_to_rational(number):
    """convert a number to rantional
    Keyword arguments: number
    return: tuple like (1, 2), (numerator, denominator)
    """
    f = Fraction(str(number))
    return (f.numerator, f.denominator)


def cartesian_to_geodetic(x, y, z, lat0, lon0, alt0):
    """
    Convert local Cartesian coordinates (x, y, z) to geographic coordinates (latitude, longitude, altitude).
    
    Parameters:
    - x, y, z: Local Cartesian coordinates in meters
    - lat0, lon0, alt0: Origin coordinates in latitude, longitude, and altitude
    
    Returns:
    - Latitude, Longitude, Altitude in degrees and meters
    """
    # Define CRS for WGS84 (geodetic coordinates)
    crs_geodetic = CRS.from_epsg(4326)  # EPSG:4326 for WGS84

    # Define ECEF CRS
    crs_ecef = CRS.from_string('EPSG:4978')  # ECEF CRS doesn't have a traditional EPSG code, but use EPSG:4978 for ECEF in WGS84

    # Transformer from geodetic to ECEF
    transformer_to_ecef = Transformer.from_crs(crs_geodetic, crs_ecef)

    # Convert origin lat/lon/alt to ECEF
    origin_ecef_x, origin_ecef_y, origin_ecef_z = transformer_to_ecef.transform(lat0, lon0, alt0)

    # Compute new ECEF coordinates
    ecef_x = origin_ecef_x + x
    ecef_y = origin_ecef_y + y
    ecef_z = origin_ecef_z + z

    # Transformer from ECEF back to geodetic
    transformer_to_geodetic = Transformer.from_crs(crs_ecef, crs_geodetic)

    # Convert the ECEF coordinates back to geodetic (lat, lon, alt)
    lat, lon, alt = transformer_to_geodetic.transform(ecef_x, ecef_y, ecef_z)

    return lat, lon, alt

def extract_images_from_folder(folder_path):
    # Iterate over all files in the provided folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.bag'):
            bag_file_path = os.path.join(folder_path, filename)
            extract_images_from_bag(bag_file_path)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract images from /image_raw topic in all ROS bags in a folder and add EXIF/XMP metadata.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing ROS bag files.")
    args = parser.parse_args()

    # Call the function to extract images from all .bag files in the folder
    extract_images_from_folder(args.folder_path)
import logging
import yaml
import random
import numpy as np
import re

from math import sin, cos, pi
from mathutils import *
from scipy.spatial.transform import Rotation as R


logger = logging.getLogger('blender')
logger.setLevel(logging.DEBUG)


class Paths:
    def __init__(self, config_file="config/config.yml"):
        # Load the config
        try:
            config = yaml.safe_load(open(config_file))
        except Exception as e:
            logger.error("Could not load config: {}".format(e))
            raise RuntimeError(
                "Ensure that the config.yml is created and filled")

        # Extract the config values
        self.config = config

        # Seed random with known seed
        random.seed(self.config["paths"].get("seed", 1))

        # Extract config for demo
        self.demo_radius = self.config["paths"].get("demo_radius", 10.0)
        self.demo_x = self.config["paths"].get("demo_x", 0.0)
        self.demo_y = self.config["paths"].get("demo_y", 0.0)
        self.demo_z = self.config["paths"].get("demo_z", 0.0)

        # Number of samples to be drawn
        self.number_of_samples = self.config["paths"].get(
            "number_of_samples", 0)

        # Path poses
        self.path_poses = []

    def __len__(self):
        return len(self.path_poses)

    def get_next_pose(self):
        for p in self.path_poses:
            yield p

    def gen_demo_circle(self, start_att):
        # Start pos
        start_pos = Vector((self.demo_x, self.demo_y, self.demo_z))

        # Increase in angle to reach spacing
        incr = 2 * pi / self.number_of_samples

        # Yaw
        phi = start_att.z

        # Loop over angle
        theta = 0.0
        while theta <= 2.0 * pi:
            x = -self.demo_radius * cos(theta)
            y = self.demo_radius * sin(theta)

            # Rotate coords
            rot_x = x * cos(phi) - y * sin(phi) + start_pos.x
            rot_y = x * sin(phi) + y * cos(phi) + start_pos.y

            # Append pose
            self.path_poses.append((Vector((rot_x, rot_y, start_pos.z)),
                                    Vector((0.0, 0.0, phi - theta))))

            # Increase angle
            theta += incr

    def gen_dataset(self):
        # Get the position min/max config
        x_min = self.config["paths"].get("x_min", 0.0)
        x_max = self.config["paths"].get("x_max", 0.0)
        y_min = self.config["paths"].get("y_min", 0.0)
        y_max = self.config["paths"].get("y_max", 0.0)
        z_min = self.config["paths"].get("z_min", 0.0)
        z_max = self.config["paths"].get("z_max", 0.0)

        # Get the euler angle min/max config
        euler_x_min = self.config["paths"].get("euler_x_min", 0.0)
        euler_x_max = self.config["paths"].get("euler_x_max", 0.0)
        euler_y_min = self.config["paths"].get("euler_y_min", 0.0)
        euler_y_max = self.config["paths"].get("euler_y_max", 0.0)
        euler_z_min = self.config["paths"].get("euler_z_min", 0.0)
        euler_z_max = self.config["paths"].get("euler_z_max", 0.0)

        # Fill the attitude list, draw some safety random samples to discard
        # faulty ones
        for a in range(1000 * self.number_of_samples):
            # Position
            x = random.random() * (x_max - x_min) + x_min
            y = random.random() * (y_max - y_min) + y_min
            z = random.random() * (z_max - z_min) + z_min

            # Attitude
            euler_x = random.random() * (euler_x_max - euler_x_min) + euler_x_min
            euler_y = random.random() * (euler_y_max - euler_y_min) + euler_y_min
            euler_z = random.random() * (euler_z_max - euler_z_min) + euler_z_min

            # Append to poses
            self.path_poses.append((Vector((x, y, z)),
                                    Vector((euler_x, euler_y, euler_z))))

    def generate_validation_set(self):
        lines = []
        extrinsics_file = self.config["paths"].get("pix4d_trajectory", None)
        ros_image_mask = r'([0-9]*)\.jpg ([\-0-9.]*) ([\-0-9\.]*) ([\-0-9\.]*) ([\-0-9\.]*) ([\-0-9\.]*) ([\-0-9\.]*)'

        if extrinsics_file is not None:
            logger.info(f"Using {extrinsics_file}")
            with open(extrinsics_file, "r") as f:
                lines = f.readlines()

        for line in lines:
            out = re.search(ros_image_mask, line)
            if out is not None:
                x, y, z = float(out.group(2)), float(out.group(3)), float(out.group(4))
                ex, ey, ez = float(out.group(5)), float(out.group(6)), float(out.group(7))

                x -= self.config["paths"].get("offset_x", 0.0)
                y -= self.config["paths"].get("offset_y", 0.0)
                z -= self.config["paths"].get("offset_z", 0.0)

                rotation = self.get_R_pix4d(ex, ey, ez)
                quat = R.from_matrix(rotation).as_quat()

                timestamp = int(out.group(1))
                if timestamp > self.config["paths"].get("min_timestamp", 0):
                    self.path_poses.append((Vector((x, y, z)), Vector((quat[1], quat[2], quat[3], quat[0])), timestamp))


    def get_R_pix4d(self, o_deg, p_deg, k_deg):
        o = np.radians(o_deg)
        p = np.radians(p_deg)
        k = np.radians(k_deg)
        rotation = np.zeros((3, 3))
        sino = np.sin(o)
        coso = np.cos(o)
        sinp = np.sin(p)
        cosp = np.cos(p)
        sink = np.sin(k)
        cosk = np.cos(k)

        # Defined accd. to https://support.pix4d.com/hc/en-us/articles/202559089
        rotation0 = np.zeros((3, 3))
        rotation0[0, :] = [1.0, 0.0, 0.0]
        rotation0[1, :] = [0.0, coso, -sino]
        rotation0[2, :] = [0.0, sino, coso]

        rotation1 = np.zeros((3, 3))
        rotation1[0, :] = [cosp, 0.0, sinp]
        rotation1[1, :] = [0.0, 1.0, 0.0]
        rotation1[2, :] = [-sinp, 0.0, cosp]

        rotation2 = np.zeros((3, 3))
        rotation2[0, :] = [cosk, -sink, 0.0]
        rotation2[1, :] = [sink, cosk, 0.0]
        rotation2[2, :] = [0.0, 0.0, 1.0]

        return rotation0 @ rotation1 @ rotation2

    def rad_to_deg(self, rad):
        return rad * 180.0 / pi

import bpy
import logging
import yaml
import re
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import random

from addon_utils import check, paths, enable
from contextlib import contextmanager
from datetime import timedelta
from math import radians
from mathutils import *
from pathlib import Path

from .paths import Paths
from .sun_calc import SunCalc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Blender:
    def __init__(self, config_file="config/config.yml"):
        # Load the config
        try:
            config = yaml.safe_load(open(config_file))
        except Exception as e:
            logger.error("Could not load config: {}".format(e))
            raise RuntimeError("Ensure that the config.yml is created and filled")

        # Activate CUDA devices if possible
        self.enable_gpus("NONE")

        # Extract the config values
        self.config = config

        # Get and setup the camera
        self.setup_camera()

        # Setup the render
        self.setup_render()

        # Setup the light
        self.setup_light()

        # Setup the output directory
        self.setup_output_dir(overwrite=False)

        # Generate the dataset
        self.paths = Paths(config_file=config_file)

        # Select mode
        mode = config.get("mode", "training")
        if mode == "training":
            self.paths.gen_dataset()
        elif mode == "validation":
            self.paths.generate_validation_set()
        elif mode == "demo":
            self.paths.gen_demo_circle(self.camera.rotation_euler)

        self.mode = mode

        # Time variables
        self.render_window = list()
        self.render_dur = 0
        self.render_rem_time = 0
        self.render_start = None
        self.initial_altitude = None

    def start(self):
        self.i = 0
        number_of_samples = self.config["paths"].get("number_of_samples", 0)
        use_second_view = self.config["output"].get("use_second_view", False)
        logger.info("Set use second view to: {}".format(use_second_view))
        ret = 0

        for p in self.paths.get_next_pose():
            # Do terrain following
            if self.config["mode"] != "validation":
                print("following terrain")
                p_tf = self.follow_terrain(p)
                if p_tf is None:
                    continue
            else:
                p_tf = p

            # Set camera pose
            q = self.set_pose(p_tf)

            # Set sun intensity
            if self.mode == "dataset":
                self.set_sun_angle()

            # Add keyframe for visualization
            # self.add_keyframe(self.i)

            # Render rgbd
            ret = self.render_rgbd(self.i)

            if (ret == 0) or (self.config["mode"] is "validation" or self.config["mode"] is "demo"):
                # Log pose
                if len(p) > 2:
                    self.log_pose(p_tf, q, timestamp=p[2])
                else:
                    self.log_pose(p_tf, q)

                # Successfully rendered RGB, increase image count
                self.i += 1

            else:
                logger.warning("Faulty RGB rendering detected")

            # Timing
            if self.render_start:
                # Duration in seconds
                self.render_dur = round(time.monotonic() - self.render_start)

                # LP filter
                self.render_dur, self.render_window = Blender.update_sliding_window(
                    self.render_dur, self.render_window)

                self.render_rem_time = (
                    number_of_samples - self.i) * self.render_dur

            # Log it
            logger.info("Rendered {}/{}, rem. time: {}".format(self.i,
                                                               number_of_samples, Blender.format_t(self.render_rem_time)))

            # Break from loop if enough samples successfully rendered
            if self.i >= number_of_samples:
                logger.info("Rendered {} samples, finished".format(self.i))
                break

            # Restart the timer
            self.render_start = time.monotonic()

        # Too many errors
        if self.i < number_of_samples:
            logger.error("Failed to render RGB, investigate!")
        else:
            logger.info("Success, finished")

    def render_rgbd(self, i):
        filepath = self.project_dir / "rgb" / f"{i:05d}_rgb.jpg"
        if filepath.is_file():
            # Check first if the depth is valid, otherwise need to redo it
            depth_file = self.project_dir / "depth" / f"{i:05d}_d.npy"
            if depth_file.is_file():
                depth = np.load(depth_file)
                if np.abs(depth).max() < 1000:
                    logger.warning(f"Skipping: {filepath}")
                    return 0

        filepath = str(filepath)

        bpy.context.scene.render.filepath = filepath

        for n in self.tree.nodes:
            self.tree.nodes.remove(n)
        rl = self.tree.nodes.new('CompositorNodeRLayers')

        vl = self.tree.nodes.new('CompositorNodeViewer')
        vl.use_alpha = True
        self.links.new(rl.outputs[0], vl.inputs[0])
        self.links.new(rl.outputs[2], vl.inputs[1])

        # Render
        bpy.context.scene.render.resolution_percentage = 100
        bpy.ops.render.render(write_still=True)

        # Get pixels
        pixels = np.array(bpy.data.images['Viewer Node'].pixels)

        # Reshape
        image = pixels.reshape(
            self.render.resolution_y,
            self.render.resolution_x,
            4)

        # Depth
        dmap = image[:, :, 3]
        dmap = np.flip(dmap)
        dmap = np.flip(dmap, 1)

        # Check that depth is correct
        max_depth = np.max(np.abs(dmap))
        if (max_depth > 1000) and not (self.config["mode"] == "validation" or self.config["mode"] == "demo"):
            logger.warning("Max depth too large: {}".format(max_depth))
            return 1

        # Save the depth image in pretty img format and binary matrix
        plt.imsave(str(self.project_dir / "depth" / "{:05d}_d.png".format(i)), dmap, cmap='viridis')
        np.save(str(self.project_dir / "depth" / "{:05d}_d".format(i)), dmap)

        # All good
        return 0

    def set_pose(self, p):
        # Set camera position
        self.set_camera_position(p[0])

        # Set camera pose
        if (len(p[1]) == 3):
            self.set_camera_rotation_euler(p[1])
        else:
            print("using quat")
            self.set_camera_rotation_quaternion(p[1])

        # Return quaternion
        return self.camera.rotation_euler.to_quaternion()

    def get_second_pose(self, p):
        p_cp = (p[0].copy(), p[1].copy())

        # Set pose to further down
        down_vel_min, down_vel_max = 0.025, 0.1
        side_vel_min, side_vel_max = -0.1, 0.1
        fps = 1.0
        p_cp[0].x += 1.0 / fps * random.random() * (side_vel_max - side_vel_min) + side_vel_min  # NOQA
        p_cp[0].y += 1.0 / fps * random.random() * (side_vel_max - side_vel_min) + side_vel_min  # NOQA
        p_cp[0].z += 1.0 / fps * random.random() * (down_vel_max - down_vel_min) + down_vel_min  # NOQA

        # Set attitude
        euler_min, euler_max = -deg_to_rad(5), deg_to_rad(5)
        p_cp[1].x += random.random() * (euler_max - euler_min) + euler_min
        p_cp[1].y += random.random() * (euler_max - euler_min) + euler_min
        p_cp[1].z += random.random() * (euler_max - euler_min) + euler_min

        # Return the new pose
        return p_cp

    def add_keyframe(self, i):
        new_obj = self.camera.copy()
        new_obj.data = self.camera.data.copy()
        new_obj.animation_data_clear()
        bpy.context.collection.objects.link(new_obj)

    def set_camera_position(self, p):
        self.camera.location = p
        logger.info("Set camera position to: {}".format(p))

    def set_camera_rotation_euler(self, r):
        self.camera.rotation_euler = r
        logger.info("Set camera rotation to: {}".format(r))

    def set_camera_rotation_quaternion(self, r):
        self.camera.rotation_mode = "QUATERNION"
        self.camera.rotation_quaternion = r
        logger.info(f"Set camera rotation to: {r}")

    def follow_terrain(self, p):
        # Do not modify old vector
        p = (p[0].copy(), p[1].copy())

        # Look at the point above the target
        point_above_target = Vector((p[0].x, p[0].y, 0.0))
        downwards = Vector((0, 0, -1))

        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                mw = obj.matrix_world
                mwi = mw.inverted()
                origin = mwi @ point_above_target
                hit, loc, _, _ = obj.ray_cast(origin, downwards)

                if hit:
                    # Get the global hit of the surface
                    global_loc = mw @ loc
                    dist = global_loc.z
                    # Modify the pose
                    p[0].z += dist

                else:
                    p = None

        return p

    def enable_gpus(self, device_type="CUDA"):
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

        for scene in bpy.data.scenes:
            scene.render.engine = 'CYCLES' 
            bpy.data.scenes["Scene"].cycles.device='GPU'
            scene.cycles.device = 'GPU'

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.compute_device_type = device_type
        devices = cycles_preferences.devices
        
        for device in devices:
            print(f"Device: {device.name}")
            print(f"Device type: {device.type}")
            if device.type == device_type:
                device.use = True
            else:
                device.use = False
            print(f"Device use: {device.use}")

        return True

    def log_pose(self, p, q, use_second_view=False, timestamp=None):
        # Index
        index = self.i

        # For second view
        if use_second_view:
            index = "{}_2".format(self.i)

        # Append the pose to the file
        with self.poses_log_file.open('a') as f:
            f.write(f"{index}, {p[0].x}, {p[0].y}, {p[0].z}, {q.x}, {q.y}, {q.z}")
            if timestamp is not None:
                 f.write(f", {timestamp}\n")
            else:
                f.write("\n")

    def setup_camera(self):
        self.camera = bpy.data.objects["Camera"]
        if self.camera is None:
            raise RuntimeError("Could not find a camera")

        # Get the values from the config, if the field is empty
        # set standard values for Flir Firefly S camera
        self.camera.data.lens = self.config["camera"].get("f", 3.5)
        self.camera.data.sensor_width = self.config["camera"].get("sx", 4.96)
        self.camera.data.sensor_height = self.config["camera"].get("sy", 3.72)

        logger.info("Camera focal length: {}".format(self.camera.data.lens))
        logger.info(
            "Camera sensor width: {}".format(
                self.camera.data.sensor_width))
        logger.info(
            "Camera sensor height: {}".format(
                self.camera.data.sensor_height))

        K = get_calibration_matrix_K_from_blender(self.camera.data)
        logger.info("K is: {}".format(K))

    def setup_render(self):
        self.scene = bpy.data.scenes["Scene"]
        self.render = self.scene.render

        # Samples
        self.scene.cycles.samples = 128

        # Setup based on config
        self.render.resolution_x = self.config["output"].get(
            "resolution_x", 640)
        self.render.resolution_y = self.config["output"].get(
            "resolution_y", 480)
        self.render.image_settings.file_format = self.config["output"].get(
            "format", "JPEG")

        logger.info("Resolution x: {}".format(self.render.resolution_x))
        logger.info("Resolution y: {}".format(self.render.resolution_y))
        logger.info(
            "Output format: {}".format(
                self.render.image_settings.file_format))

        # Other setup
        self.render.use_compositing = True
        self.scene.use_nodes = True

        self.tree = bpy.context.scene.node_tree
        self.links = self.tree.links

        # Set cycles render
        render_name = self.config["output"].get("render", "CYCLES")
        bpy.context.scene.render.engine = render_name
        logger.info("Using render: {}".format(render_name))

    def setup_light(self):
        self.sun = bpy.data.objects["Sun"]

        # Min / max energy of sun
        self.sun_energy_min = self.config["sun"].get("energy_min", 1.0)
        self.sun_energy_max = self.config["sun"].get("energy_max", 10.0)

        # Random
        random.seed(10)

        logger.info(
            "Sun energy min, max: {}, {} W/m2".format(self.sun_energy_min, self.sun_energy_max))

        # Create sun calc instance
        self.sc = SunCalc()

    def get_sun_intensity(self):
        energy = random.random() * (self.sun_energy_max - self.sun_energy_min) + \
            self.sun_energy_min
        logger.info("Get sun energy: {} W/m2".format(energy))
        return energy

    def set_sun_angle(self):
        objs = bpy.data.objects
        objs.remove(objs['Sun'], do_unlink=True)

        sun_data = bpy.data.lights.new(name="Sun", type="SUN")
        sun_data.energy = self.get_sun_intensity()

        self.sun = bpy.data.objects.new(name="Sun", object_data=sun_data)

        # Set light angle
        self.sun.data.angle = self.config["sun"].get("angle", radians(45.0))

        bpy.context.collection.objects.link(self.sun)
        bpy.context.view_layer.objects.active = self.sun

        # Get the position based on the local time
        local_time_min = self.config["sun"].get("local_time_min", 10.0)
        local_time_max = self.config["sun"].get("local_time_max", 16.0)
        local_time = random.random() * (local_time_max - local_time_min) + local_time_min

        rot_euler = self.sc.get_sun_pose(local_time=local_time)
        logger.info("Set local time to: {}".format(local_time))

        rotation_quaternion = rot_euler.to_quaternion()
        self.sun.rotation_quaternion = rotation_quaternion

        if self.sun.rotation_mode in {'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'}:
            self.sun.rotation_euler = rotation_quaternion.to_euler(
                self.sun.rotation_mode)
        else:
            self.sun.rotation_euler = rot_euler

        rotation_axis_angle = self.sun.rotation_quaternion.to_axis_angle()
        self.sun.rotation_axis_angle = (
            rotation_axis_angle[1], *rotation_axis_angle[0])

        # Set position also
        self.sun.location = self.sc.get_sun_position()
        logger.info("Set sun location to {}".format(self.sun.location))

        # update scene, if needed
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

    def setup_output_dir(self, overwrite=False):
        output_dir = Path(self.config["output"].get("dir", "output"))
        if not output_dir.is_dir():
            os.mkdir(output_dir)
            logger.warn("Output directory does not exist")

        # For every run folder structure is 001/<images>, 002/<images>,
        proj_count = 0
        for f in output_dir.iterdir():
            out = re.search("[^a-zA-Z]([0-9][0-9][0-9])", str(f))
            if out:
                proj_count_cur = int(out.group(1))
                if proj_count_cur > proj_count:
                    proj_count = proj_count_cur

        # Create the initial one
        if not (output_dir / "000").exists():
            os.mkdir(str(output_dir / "000"))

            # In this case overwrite anyways
            overwrite = True

        # If overwrite is False, create new project folder
        if "pickup" in self.config["output"].keys():
            self.project_dir = output_dir / f'{self.config["output"]["pickup"]}'
            print(f"Resuming in directory {self.project_dir}")

        elif not overwrite:
            # Safe for all OS
            proj_count += 1
            self.project_dir = output_dir / "{:03d}".format(proj_count)
            os.mkdir(str(self.project_dir))
        else:
            self.project_dir = output_dir / "{:03d}".format(proj_count)

        # Check that all subfolders exist
        if not (self.project_dir / "rgb").exists():
            os.mkdir(str(self.project_dir / "rgb"))

        if not (self.project_dir / "depth").exists():
            os.mkdir(str(self.project_dir / "depth"))

        if not (self.project_dir / "segmentation").exists():
            os.mkdir(str(self.project_dir / "segmentation"))

        if not (self.project_dir / "depth_prior").exists():
            os.mkdir(str(self.project_dir / "depth_prior"))

        # Poses log file, preparation
        self.poses_log_file = self.project_dir / "poses.txt"
        self.poses_log_file.touch(exist_ok=True)

        # Write a header
        with self.poses_log_file.open("w") as f:
            f.write(
                "Poses of project {}, format: i, x, y, z, quat x, quat y, quat z, quat w\n".format(proj_count))

    @staticmethod
    def update_sliding_window(val, window, window_size=1000):
        # Add the new value
        window.append(val)

        # Kick out the old value
        if len(window) >= window_size:
            del window[0]

        # Compute average
        return sum(window) / len(window), window

    @staticmethod
    def format_t(seconds):
        seconds = round(seconds)
        return str(timedelta(seconds=seconds))


def deg_to_rad(deg):
    return np.pi * deg / 180.0

# ---------------------------------------------------------------
# 3x4 P matrix from Blender camera
# ---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model


def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew, u_0),
         (0, alpha_v, v_0),
         (0, 0, 1)))
    return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction


def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner


def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return Vector(
        (co_2d.x *
         render_size[0],
         render_size[1] -
         co_2d.y *
         render_size[1]))

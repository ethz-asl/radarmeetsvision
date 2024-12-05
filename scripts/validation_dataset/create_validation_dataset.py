import argparse
import cv2
import laspy
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pickle
import re
import rosbag
import scipy.ndimage as ndi
import sensor_msgs.point_cloud2 as pc2

from pathlib import Path
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset")

class ValidationDataset:
    def __init__(self, input_dir):
        self.downscale = 2

        self.snr_filter = 150 if 'outdoor' in str(input_dir) else 200
        logger.info(f"Setting SNR to {self.snr_filter}")
        for file_path in input_dir.iterdir():
            if 'calibrated_camera_parameters.txt' in str(file_path):
                self.camera_parameters_file = file_path

            elif 'offset.xyz' in str(file_path):
                self.offset_file = file_path

            elif 'result_tracks3_full_2.las' in str(file_path):
                self.world_pointcloud_file = file_path

            elif '.bag' in str(file_path):
                self.bag_file = file_path

        # TODO: Don't hardcode the calibrations, what about different setups
        # Correct for aesch and maschinenhalle
        self.R_camera_radar = np.array(([[-9.99537680e-01, 2.06396371e-03, -3.03342592e-02],
                                         [ 3.03920764e-02, 3.94275324e-02, -9.98760127e-01],
                                         [8.65399670e-04, 9.99220301e-01, 3.94720324e-02]]))
        self.t_camera_radar = np.array([[0.01140952, -0.0056055,  -0.02026767]]).T
        self.K = np.array([[1200.4311769445228, 0.0, 634.1037111885645],
                           [0.0, 1201.8315992165312, 432.2169659507848],
                           [0.0, 0.0, 1.0]])
        self.distortion_coeffs_equidistant = [-0.12935247711303535, 0.06833088389569625, -0.0995869958615853, 0.04717896330175957]


    def generate(self):
        output_dir = Path(self.bag_file).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        rgb_dir = output_dir / 'rgb'
        rgb_dir.mkdir(exist_ok=True)
        depth_dir = output_dir / 'depth'
        depth_dir.mkdir(exist_ok=True)
        depth_prior_dir = output_dir / 'depth_prior'
        depth_prior_dir.mkdir(exist_ok=True)

        # Empty pose file
        pose_file = output_dir / 'pose_file.txt'
        with pose_file.open('w') as f:
            pass

        # Build the dict with depth GTs
        depth_dict_file = output_dir / 'depth_dict.pkl'
        intrinsics_dict_file = output_dir / 'intrinsics_dict.pkl'
        if not depth_dict_file.is_file() or not intrinsics_dict_file.is_file():
            logger.info("Computing depth dict")
            depth_dict, intrinsics_dict = self.get_depth_dict()
            with depth_dict_file.open('wb') as f:
                pickle.dump(depth_dict, f)
            with intrinsics_dict_file.open('wb') as f:
                pickle.dump(intrinsics_dict, f)
        else:
            with depth_dict_file.open('rb') as f:
                depth_dict = pickle.load(f)
            with intrinsics_dict_file.open('rb') as f:
                intrinsics_dict = pickle.load(f)

        topics = ['/image_raw', '/radar/cfar_detections', '/tf_static']
        points_radar_window = []
        snr_radar_window = []
        bridge = CvBridge()

        with rosbag.Bag(self.bag_file, 'r') as bag:
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=topics)):
                if topic == '/radar/cfar_detections':
                    # Transform PC to camera frame
                    points_radar, snr_radar, _ = self.pointcloud2_to_xyz_array(msg)
                    points_radar = points_radar.T

                    if len(points_radar):
                        points_radar_window.append(self.R_camera_radar @ points_radar + self.t_camera_radar)
                        snr_radar_window.append(snr_radar)
                        if len(points_radar_window) > 3:
                            points_radar_window.pop(0)
                            snr_radar_window.pop(0)

                elif topic == '/image_raw':
                    timestamp = t.to_nsec()
                    timestamp_alt = msg.header.stamp.to_nsec()

                    if timestamp in depth_dict.keys() or timestamp_alt in depth_dict.keys():
                        if timestamp not in depth_dict.keys() and timestamp_alt in depth_dict.keys():
                            timestamp = timestamp_alt

                        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                        depth_prior = self.get_depth_prior(points_radar_window, snr_radar_window, depth_dict[timestamp])
                        index = intrinsics_dict[timestamp]['index']

                        if depth_prior is not None:
                            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                            img = cv2.resize(img, (img.shape[1]//self.downscale, img.shape[0]//self.downscale))
                            img_file = rgb_dir / f'{index:05d}_rgb.jpg'
                            cv2.imwrite(str(img_file), img)

                            # Write the depth gt
                            depth_file = output_dir / 'depth' / f'{index:05d}_d.npy'
                            np.save(depth_file, depth_dict[timestamp])
                            plt.imsave(output_dir / 'depth' / f'{index:05d}_d.jpg', depth_dict[timestamp], vmin=0,vmax=15)

                            depth_prior_file = output_dir / 'depth_prior' / f'{index:05d}_ra.npy'
                            np.save(depth_prior_file, depth_prior)
                            plt.imsave(output_dir / 'depth_prior' / f'{index:05d}_ra.jpg', depth_prior, vmin=0,vmax=15)

                            # Write pose
                            pose_flat = intrinsics_dict[timestamp]['pose'].flatten()
                            with pose_file.open('a') as f:
                                f.write(f'{index}: ')
                                f.write(' '.join(map(str, pose_flat)))
                                f.write('\n')


    def pointcloud2_to_xyz_array(self, cloud_msg):
        # Convert PointCloud2 to a list of points (x, y, z)
        points = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))
        snr = np.array(list(pc2.read_points(cloud_msg, field_names=("snr"), skip_nans=True)))
        noise = np.array(list(pc2.read_points(cloud_msg, field_names=("noise"), skip_nans=True)))
        return points, snr, noise

    def get_depth_prior(self, points_radar_window, snr_radar_window, depth):
        height, width = depth.shape
        radius_prior_pixel = 5
        circular_mask = self.create_circular_mask(2 * width, 2 * height, radius=radius_prior_pixel)
        depth_prior = None

        points_radar = np.concatenate(points_radar_window, axis=1)
        points_snr = np.concatenate(snr_radar_window, axis=0)

        snr_filter = points_snr[:, 0] > self.snr_filter
        points_radar = points_radar[:, snr_filter]
        points_snr = points_snr[snr_filter, :]

        if len(points_radar) > 0:
            # TODO: Don't hardcode the calibrations, what about different setups
            radar_image_points = self.K @ points_radar
            radar_depth = radar_image_points[2, :].copy()
            radar_image_points /= radar_depth

            depth_positive_mask = radar_depth > 0.0
            if depth_positive_mask.sum() > 0:
                u = radar_image_points[0, depth_positive_mask]
                v = radar_image_points[1, depth_positive_mask]
                radar_depth = radar_depth[depth_positive_mask]
                points_snr = points_snr[depth_positive_mask]

                # TODO: Don't hardcode the calibrations, what about different setups
                u_distorted, v_distorted = self.apply_distortion_equidistant(u.copy(), v.copy())
                u_distorted, v_distorted = u_distorted / self.downscale, v_distorted / self.downscale

                inside_image_mask = (u_distorted >= 0) & (u_distorted < width) & (v_distorted >= 0) & (v_distorted < height)
                if inside_image_mask.sum() > 0:
                    u_distorted = u_distorted[inside_image_mask]
                    v_distorted = v_distorted[inside_image_mask]
                    radar_depth = radar_depth[inside_image_mask]
                    points_snr = points_snr[inside_image_mask]

                    depth_prior = np.zeros((width, height), dtype=np.float32)
                    snr_prior = np.zeros((width, height), dtype=np.float32)

                    for i, (x, y) in enumerate(zip(u_distorted, v_distorted)):
                        translated_mask = circular_mask[int(width - x):int(2 * width - x), int(height - y):int(2 * height - y)]
                        depth_prior += radar_depth[i] * translated_mask
                        snr_prior += points_snr[i] * translated_mask

                    depth_prior = depth_prior.T

                    # depth_factor = np.zeros((height, width))
                    # depth_factor[height//factor:2*height//factor, width//factor:2*width//factor] = depth
                    # fig, axs = plt.subplots(1, 2)
                    # axs.set_xlim([0, width])
                    # axs.set_ylim([height, 0])
                    # scatter = axs.scatter(u_distorted, self.transform_v_to_plt(v_distorted, height), c=radar_depth, vmin=0.0, vmax=5.0)
                    # cbar = plt.colorbar(scatter, ax=axs, cmap='viridis')
                    # cbar.set_label('Radar Depth')
                    # scatter = axs[0].scatter(u_distorted, v_distorted, c=radar_depth, vmin=0.0, vmax=15.0)
                    # cbar = plt.colorbar(scatter, ax=axs, cmap='viridis')
                    # axs[0].imshow(depth, cmap='viridis', origin='upper', vmin=0.0, vmax=15.0)
                    # axs[0].set_xlim([0, width])
                    # axs[0].set_ylim([height, 0])
                    # axs[1].imshow(depth_prior, vmin=0.0, cmap='viridis', vmax=15.0)

                    # print(radar_depth, points_snr, points_noise)
                    # plt.show()

        return depth_prior

    def process_camera_data(self, data_dict, points, offset):
        timestamp = data_dict['timestamp']
        index = data_dict['index']
        K = data_dict['K']
        R = data_dict['R']
        t = data_dict['t']
        width = data_dict['width']
        height = data_dict['height']
        radial_dist = data_dict['radial_dist']
        tangential_dist = data_dict['tangential_dist']

        t_rotated = -R @ t
        pose = np.concatenate((R, t_rotated), axis=1)
        m = K @ pose

        image_points = m @ points
        depth_flat = image_points[2, :].copy()
        depth_positive_mask = depth_flat > 0.0

        depth_dict = {}
        intrinsics_dict = {}

        if depth_positive_mask.sum() > 0:
            depth_flat = depth_flat[depth_positive_mask]
            image_points_filtered = image_points[:, depth_positive_mask] / depth_flat

            u, v = image_points_filtered[0, :], image_points_filtered[1, :]

            # Apply distortion to u, v before checking inside image bounds
            u_distorted, v_distorted = self.apply_distortion_equidistant(u.copy(), v.copy())

            # Check if distorted coordinates are within the image bounds
            inside_image_mask = (u_distorted >= 0) & (u_distorted < width) & (v_distorted >= 0) & (v_distorted < height)

            if inside_image_mask.sum() > 0:
                v_distorted = v_distorted[inside_image_mask]
                u_distorted = u_distorted[inside_image_mask]

                # Use interpolation to assign depth values instead of direct pixel indexing
                depth = self.interpolate_depth(u_distorted, v_distorted, u, v, depth_flat, height, width)
                depth_dict[timestamp] = depth
                intrinsics_dict[timestamp] = {'K': K,
                                              'radial_dist': radial_dist,
                                              'tangential_dist': tangential_dist,
                                              'index': index,
                                              'pose': pose
                                            }

        logger.info(f'Computed depth for {timestamp}, {index}')
        return depth_dict, intrinsics_dict


    def get_depth_dict(self):
        # Read the offset and point cloud data
        offset = self.read_offset(self.offset_file)
        points = self.read_pointcloud(self.world_pointcloud_file, offset)

        # Prepare a multiprocessing pool
        pool = mp.Pool(mp.cpu_count())

        # Arguments for each process
        tasks = []
        for data_dict in self.get_next_camera(self.camera_parameters_file):
            tasks.append((data_dict.copy(), points, offset))

        # Processing data in parallel using Pool.starmap
        results = pool.starmap(self.process_camera_data, tasks)

        # Close and join the pool
        pool.close()
        pool.join()

        # Combine results from all processes
        combined_depth_dict = {}
        combined_intrinsics_dict = {}

        for depth_dict, intrinsics_dict in results:
            combined_depth_dict.update(depth_dict)
            combined_intrinsics_dict.update(intrinsics_dict)

        return combined_depth_dict, combined_intrinsics_dict


    def interpolate_depth(self, u_query, v_query, u_data, v_data, depth_data, height, width):
        # Create an empty depth map
        depth_map = np.zeros((height//self.downscale, width//self.downscale))

        # Downsclae the coordinates
        u_data /= self.downscale
        v_data /= self.downscale
        u_query /= self.downscale
        v_query /= self.downscale

        # Combine u and v coordinates into pairs for efficient searching
        data_points = np.vstack((u_data, v_data)).T
        query_points = np.vstack((u_query, v_query)).T

        # Use KDTree for efficient nearest-neighbor searching
        tree = cKDTree(data_points)

        # For each distorted point, find the k nearest neighbors in the original points
        k = 10  # Number of nearest neighbors to consider
        dist, indices = tree.query(query_points, k=k)

        # Inverse distance weighting interpolation
        for i, (point, neighbors_idx, dists) in enumerate(zip(query_points, indices, dist)):
            if np.any(dists == 0):
                # If one of the neighbors has zero distance (i.e., exact match), use its depth directly
                depth_map[int(point[1]), int(point[0])] = depth_data[neighbors_idx[np.argmin(dists)]]
            else:
                # Apply inverse distance weighting
                weights = 1.0 / dists
                weights /= np.sum(weights)
                interpolated_depth = np.sum(weights * depth_data[neighbors_idx])
                depth_map[int(point[1]), int(point[0])] = interpolated_depth

        return depth_map

    def apply_distortion_equidistant(self, u, v):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) / fx
        y = (v - cy) / fy
        r = np.sqrt(x**2 + y**2)
        mask_r = r > 1e-8

        # Only apply to big enough radii
        theta = np.arctan(r[mask_r])
        thetad = theta * (1 + \
                          self.distortion_coeffs_equidistant[0] * theta**2 + \
                          self.distortion_coeffs_equidistant[1] * theta**4 + \
                          self.distortion_coeffs_equidistant[2] * theta**6 + \
                          self.distortion_coeffs_equidistant[3] * theta**8)

        scaling = thetad / r[mask_r]

        xd, yd = np.copy(x), np.copy(y)
        xd[mask_r] = x[mask_r] * scaling
        yd[mask_r] = y[mask_r] * scaling

        ud = xd * fx + cx
        vd = yd * fy + cy
        return ud, vd

    def get_next_camera(self, camera_parameters_file):
        with open(camera_parameters_file, "r") as f:
            lines = f.readlines()

        filename_mask = r'([0-9]*).jpg ([0-9]*) ([0-9]*)'

        image_count = 0
        intrinsics_index = None
        translation_index = None
        radial_dist_index = None
        tangential_dist_index = None
        data_dict = {'timestamp': None, 'index': None, 'width': None, 'height': None, 'K': None, 't': None, 'R': None}

        for i in range(len(lines)):
            line = lines[i]
            if '.jpg' in line:
                out = re.search(filename_mask, line)
                if out is not None:
                    data_dict['timestamp'] = int(out.group(1))
                    data_dict['width'] = int(out.group(2))
                    data_dict['height'] = int(out.group(3))
                    data_dict['index'] = image_count

                    intrinsics_index = i + 1
                    radial_dist_index = i + 4
                    tangential_dist_index = i + 5
                    translation_index = i + 6
                    image_count += 1

            elif intrinsics_index == i:
                intrinsics_index = None
                K = np.zeros((3, 3))
                for j in range(3):
                    K[j, :] = np.array(lines[i + j].split(), dtype=float)
                data_dict['K'] = K

            elif radial_dist_index == i:
                radial_dist_index = None
                radial_dist = np.array(line.split(), dtype=float)
                data_dict['radial_dist'] = radial_dist

            elif tangential_dist_index == i:
                tangential_dist_index = None
                tangential_dist = np.array(line.split(), dtype=float)
                data_dict['tangential_dist'] = tangential_dist

            elif translation_index == i:
                translation_index = None

                t = np.array(line.split(), dtype=float)
                data_dict['t'] = t.reshape((3, 1))

                # Next 3 lines are rotation matrix
                R = np.zeros((3, 3))
                for j in range(3):
                    R[j, :] = np.array(lines[i + j + 1].split(), dtype=float)
                data_dict['R'] = R

                yield data_dict

    def read_pointcloud(self, pointcloud_file, offset):
        points = None
        ox = offset[0, 0]
        oy = offset[0, 1]
        oz = offset[0, 2]
        with laspy.open(pointcloud_file) as fh:
            las = fh.read()
            points = np.vstack((las.x - ox, las.y - oy, las.z - oz, np.ones(fh.header.point_count)))

        return points


    def read_offset(self, offset_file):
        offset_mask = r'([\.\-0-9]*) ([\.\-0-9]*) ([\.\-0-9]*)'
        offset = None
        with offset_file.open('r') as f:
            lines = f.readlines()

        for line in lines:
            out = re.search(offset_mask, line)
            if out is not None:
                offset = np.array([[float(out.group(1)), float(out.group(2)), float(out.group(3))]])
                break

        return offset

    def create_circular_mask(self, h, w, center=None, radius=None):
        # From:
        # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

        mask = dist_from_center <= radius
        return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the camera parameters, offset, and point cloud files.')
    parser.add_argument('input_dir', type=Path, help='Path to folder containing all required files')

    args = parser.parse_args()
    dataset = ValidationDataset(args.input_dir)
    dataset.generate()

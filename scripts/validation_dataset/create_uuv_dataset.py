import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rosbag

from cv_bridge import CvBridge
from pathlib import Path
from scipy.spatial.distance import cdist
from sensor_msgs.msg import Image, CompressedImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset")

class UUVDataset:
    def __init__(self, input_dir, max_priors):
        self.max_priors = max_priors
        self.data_bag_file = None
        for file_path in input_dir.iterdir():
            if '.bag' in str(file_path):
                if not 'data' in str(file_path):
                    self.bag_file = file_path

                else:
                    print(f"Found data bag file: {file_path}")
                    self.data_bag_file = file_path

    def generate(self):
        output_dir = Path(self.bag_file).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        rgb_dir = output_dir / 'rgb'
        rgb_full_dir = output_dir / 'rgb_full'
        rgb_dir.mkdir(exist_ok=True)
        rgb_full_dir.mkdir(exist_ok=True)
        depth_dir = output_dir / 'depth'
        depth_dir.mkdir(exist_ok=True)
        depth_prior_dir = output_dir / 'depth_prior'
        depth_prior_dir.mkdir(exist_ok=True)
        self.target_width = 640
        self.target_height = 480
        self.fft_dir = output_dir / 'fft_data'

        topics = ['/ted/image', '/navigation/plane_approximation', '/sensor/dvl_position']
        image_count = 0
        bridge = CvBridge()

        net_distance_file = output_dir / 'net_distances.txt'
        prior_distances_file = output_dir / f'prior_distances_{self.max_priors}.txt'
        timestamps_file = output_dir / f'timestamps.txt'
        net_distances = []
        prior_distances = []
        timestamps = []
        last_net_distance = 0.0
        dvl_msgs = []

        # NOTE: This is for the case of two separate bag files
        if self.data_bag_file is not None:
            with rosbag.Bag(self.data_bag_file, 'r') as bag:
                for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['/navigation/plane_approximation'])):
                    if topic == '/navigation/plane_approximation':
                        dvl_msgs.append((t, msg.NetDistance))

        with rosbag.Bag(self.bag_file, 'r') as bag:
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=topics)):
                if topic == '/ted/image':
                    if len(dvl_msgs):
                        # This means we have two separate bags, try to query for
                        dvl_msg = self.find_zoh_message(dvl_msgs, t)
                        if dvl_msg is not None:
                            last_net_distance = dvl_msg[1]

                    net_distances.append(last_net_distance)

                    depth_prior = None
                    if self.max_priors is not None:
                        depth_prior = self.get_fft_depth_prior(image_count, max_priors=self.max_priors, last_dvl_distance=last_net_distance)

                    if depth_prior is not None or self.max_priors is None:
                        # Check if the image is compressed or not
                        try:
                            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                        except Exception as e:
                            np_arr = np.frombuffer(msg.data, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        flipped_image = cv2.flip(img, 0)
                        if 'ted' in topic:
                            _, width, _ = flipped_image.shape
                            width = width // 2
                            image = flipped_image[:, width:, :]
                        else:
                            flipped_image = cv2.rotate(flipped_image, cv2.ROTATE_180)
                            image = flipped_image

                        image_resized = cv2.resize(image, (self.target_width, self.target_height))

                        img_file = rgb_dir / f'{image_count:05d}_rgb.jpg'
                        cv2.imwrite(str(img_file), image_resized)

                        # Also save the full image (with just the number, no leading zeros)
                        img_file_full = rgb_full_dir / f'{image_count}.jpg'
                        cv2.imwrite(str(img_file_full), image)

                        if depth_prior is not None:
                            depth_prior_file = output_dir / 'depth_prior' / f'{image_count:05d}_ra.npy'
                            np.save(depth_prior_file, depth_prior)
                            plt.imsave(output_dir / 'depth_prior' / f'{image_count:05d}_ra.jpg', depth_prior, vmin=0, vmax=15)

                            prior_mean = depth_prior[depth_prior > 0.0]
                            if len(prior_mean):
                                prior_distances.append(prior_mean.mean())
                            else:
                                prior_distances.append(np.nan)

                        timestamps.append(t)
                        image_count += 1

                elif topic == '/navigation/plane_approximation':
                    last_net_distance = msg.NetDistance

        with net_distance_file.open('w') as f:
            for i, d in enumerate(net_distances):
                f.write(f'{i}: {d}\n')

        with prior_distances_file.open('w') as f:
            for i, d in enumerate(prior_distances):
                f.write(f'{i}: {d}\n')

        with timestamps_file.open('w') as f:
            for i, d in enumerate(timestamps):
                f.write(f'{i}: {d}\n')

    def find_zoh_message(self, msgs, query_time):
        """
        Find the zero-order hold message given a query time.

        Args:
            msgs (list of tuple): List of (rospy.Time, message) tuples sorted by rospy.Time.
            query_time (rospy.Time): The time for which the ZOH message is required.

        Returns:
            message: The message corresponding to the ZOH.
        """
        # Binary search for the closest timestamp <= query_time
        left, right = 0, len(msgs) - 1
        best_match = None

        while left <= right:
            mid = (left + right) // 2
            msg_time, msg = msgs[mid]

            if msg_time <= query_time:
                best_match = (msg_time, msg)
                left = mid + 1
            else:
                right = mid - 1

        return best_match

    def get_fft_depth_prior(self, image_count, max_priors=None, last_dvl_distance=None):
        radius_prior_pixel = 5
        fft_csv_file = self.fft_dir / f'{image_count}_features.csv'
        depth_prior = np.zeros((self.target_width, self.target_height), dtype=np.float32)
        circular_mask = self.create_circular_mask(2 * self.target_width, 2 * self.target_height, radius=radius_prior_pixel)

        try:
            data_raw = pd.read_csv(fft_csv_file, header=None, names=['x', 'y', 'z'])  # Assuming no headers in the CSV
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

        # There are some invalid points that are just at 0,0
        data = data_raw.loc[(data_raw['x'] > 0) & (data_raw['y'] > 0)]

        if max_priors is not None and len(data) > max_priors:
            points = data[['x', 'y']].values
            if max_priors == 1:
                # Select the most central point (closest to centroid)
                centroid = points.mean(axis=0)
                distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
                central_index = np.argmin(distances_to_centroid)
                selected_indices = [central_index]

            else:
                # Select max_priors points that maximize distances
                selected_indices = [0]  # Start with the first point
                for _ in range(1, max_priors):
                    remaining_indices = list(set(range(len(points))) - set(selected_indices))
                    remaining_points = points[remaining_indices]
                    selected_points = points[selected_indices]
                    distances = cdist(remaining_points, selected_points, metric='euclidean')
                    min_distances = distances.min(axis=1)
                    next_index = remaining_indices[np.argmax(min_distances)]
                    selected_indices.append(next_index)

            data = data.iloc[selected_indices]  # Keep only the selected rows

        # Process each selected point
        if len(data):
            for _, row in data.iterrows():
                v, u, z = int(row['x']), int(row['y']), float(row['z'])
                # The FFT priors are defined for 320*240 window
                u *= 2
                v *= 2
                translated_mask = circular_mask[
                    int(self.target_width - u):int(2 * self.target_width - u),
                    int(self.target_height - v):int(2 * self.target_height - v)
                ]
                depth_prior += z * translated_mask
        else:
            # NOTE: Add the DVL as a fallback mechanism
            # NOTE: This is a very crude calibration, but probably good enough for this
            IMU_TO_STEREO_DEPTH_CALIB = -0.06

            u = self.target_width // 2
            v = self.target_height // 2
            translated_mask = circular_mask[
                int(self.target_width - u):int(2 * self.target_width - u),
                int(self.target_height - v):int(2 * self.target_height - v)
            ]
            depth_prior += (last_dvl_distance + IMU_TO_STEREO_DEPTH_CALIB) * translated_mask

        depth_prior = depth_prior.T
        return depth_prior



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
    parser.add_argument('--max_priors', type=int, default=None)

    args = parser.parse_args()
    dataset = UUVDataset(args.input_dir, args.max_priors)
    dataset.generate()

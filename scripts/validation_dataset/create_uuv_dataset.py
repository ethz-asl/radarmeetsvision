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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset")

class UUVDataset:
    def __init__(self, input_dir, max_priors):
        self.max_priors = max_priors
        for file_path in input_dir.iterdir():
            if '.bag' in str(file_path):
                self.bag_file = file_path

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

        with rosbag.Bag(self.bag_file, 'r') as bag:
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=topics)):
                if topic == '/ted/image':
                    # Depth prior
                    depth_prior = self.get_fft_depth_prior(image_count, max_priors=self.max_priors)
                    if depth_prior is not None:
                        # RGB image
                        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                        flipped_image = cv2.flip(img, 0)
                        _, width, _ = flipped_image.shape
                        width = width //2
                        right_image = flipped_image[:, width:, :]
                        right_image_resized = cv2.resize(right_image, (self.target_width, self.target_height))

                        img_file = rgb_dir / f'{image_count:05d}_rgb.jpg'
                        cv2.imwrite(str(img_file), right_image_resized)

                        # Also save the full image (with just the number, no leading zeros)
                        img_file_full = rgb_full_dir / f'{image_count}.jpg'
                        cv2.imwrite(str(img_file_full), right_image)

                        depth_prior_file = output_dir / 'depth_prior' / f'{image_count:05d}_ra.npy'
                        np.save(depth_prior_file, depth_prior)
                        plt.imsave(output_dir / 'depth_prior' / f'{image_count:05d}_ra.jpg', depth_prior, vmin=0,vmax=15)
                        image_count += 1


    def get_fft_depth_prior(self, image_count, max_priors=None):
        radius_prior_pixel = 5
        fft_csv_file = self.fft_dir / f'{image_count}_features.csv'
        depth_prior = np.zeros((self.target_width, self.target_height), dtype=np.float32)
        circular_mask = self.create_circular_mask(2 * self.target_width, 2 * self.target_height, radius=radius_prior_pixel)

        try:
            data = pd.read_csv(fft_csv_file, header=None, names=['x', 'y', 'z'])  # Assuming no headers in the CSV
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

        if max_priors is not None and len(data) > max_priors:
            points = data[['x', 'y']].values  # Extract coordinates as (x, y)
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
        for _, row in data.iterrows():
            y, x, z = int(row['x']), int(row['y']), float(row['z'])
            translated_mask = circular_mask[
                int(self.target_width - x):int(2 * self.target_width - x),
                int(self.target_height - y):int(2 * self.target_height - y)
            ]
            depth_prior += z * translated_mask

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
    parser.add_argument('max_priors', type=int, default=None)

    args = parser.parse_args()
    dataset = UUVDataset(args.input_dir, args.max_priors)
    dataset.generate()

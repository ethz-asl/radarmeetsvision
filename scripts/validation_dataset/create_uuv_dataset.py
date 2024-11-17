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

class UUVDataset:
    def __init__(self, input_dir):
        for file_path in input_dir.iterdir():
            if '.bag' in str(file_path):
                self.bag_file = file_path

    def generate(self):
        output_dir = Path(self.bag_file).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        rgb_dir = output_dir / 'rgb'
        rgb_dir.mkdir(exist_ok=True)
        depth_dir = output_dir / 'depth'
        depth_dir.mkdir(exist_ok=True)
        depth_prior_dir = output_dir / 'depth_prior'
        depth_prior_dir.mkdir(exist_ok=True)
        target_width = 640
        target_height = 480

        topics = ['/ted/image', '/navigation/plane_approximation', '/sensor/dvl_position']
        image_count = 0
        depth_priors = None
        bridge = CvBridge()
        radius_prior_pixel = 5

        with rosbag.Bag(self.bag_file, 'r') as bag:
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=topics)):
                if topic == '/navigation/plane_approximation':
                    depth_priors = [msg.NetDistance]

                elif topic == '/ted/image' and depth_priors is not None:
                    # RGB image
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    flipped_image = cv2.flip(img, 0)
                    height, width, _ = flipped_image.shape
                    width = width //2
                    right_image = flipped_image[:, width:, :]
                    right_image_resized = cv2.resize(right_image, (target_width, target_height))

                    img_file = rgb_dir / f'{image_count:05d}_rgb.jpg'
                    cv2.imwrite(str(img_file), right_image_resized)

                    # Depth prior
                    depth_prior = np.zeros((target_width, target_height), dtype=np.float32)
                    circular_mask = self.create_circular_mask(2 * target_width, 2 * target_height, radius=radius_prior_pixel)
                    x, y = target_width // 2, target_height // 2
                    translated_mask = circular_mask[int(target_width - x):int(2 * target_width - x), int(target_height - y):int(2 * target_height - y)]
                    depth_prior += depth_priors[0] * translated_mask
                    depth_prior = depth_prior.T

                    depth_prior_file = output_dir / 'depth_prior' / f'{image_count:05d}_ra.npy'
                    np.save(depth_prior_file, depth_prior)
                    plt.imsave(output_dir / 'depth_prior' / f'{image_count:05d}_ra.jpg', depth_prior, vmin=0,vmax=15)

                    image_count += 1

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
    dataset = UUVDataset(args.input_dir)
    dataset.generate()

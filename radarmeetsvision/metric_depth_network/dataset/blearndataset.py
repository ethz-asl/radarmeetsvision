######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

# Class was adapted to conform to the other Dataset classes in this directory

import cv2
import logging
import numpy as np
import torch
import random
import re

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from scipy.spatial import cKDTree

from .transform import Resize, PrepareForNet, Crop

logger = logging.getLogger(__name__)

class BlearnDataset(Dataset):
    def __init__(self, dataset_dir, mode, size):
        self.mode = mode
        self.size = size

        self.transform = Compose([
            Resize(
                width=size[0],
                height=size[1],
                resize_target=True if 'train' in mode else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            # TODO: Think if we want to add this here, again, evaluate, where do these numbers come from?
            # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if 'train' in self.mode else []))

        self.dir = Path(dataset_dir)
        if not self.dir.is_dir():
            logger.error(f"Could not find: {self.dir}")
            raise FileNotFoundError

        self.rgb_dir = self.dir / "rgb"
        self.rgb_template = "{:05d}_rgb.jpg"
        self.rgb_template_alt = "{:04d}_rgb.jpg"
        self.rgb_mask = r'([0-9]*)_rgb.jpg'

        self.depth_dir = self.dir / "depth"
        self.depth_template = "{:05d}_d.npy"
        self.depth_template_alt = "{:04d}_d.npy"
        self.depth_normalized_template = "{:05d}_dn.npy"
        self.depth_min, self.depth_max, self.depth_range = self.get_depth_range()

        self.depth_prior_dir = self.dir / "depth_prior"
        self.depth_prior_template = "{:05d}_ra.npy"

        self.height = size[0]
        self.width = size[1]
        self.train_split = 0.8

        self.filelist = self.get_filelist()
        if self.filelist:
            logger.info(f"Loaded {dataset_dir} with length {len(self.filelist)}")

    def get_filelist(self):
        # Sort and find all .jpg's
        all_rgb_files = list(self.rgb_dir.glob('*.jpg'))
        all_rgb_files = sorted(all_rgb_files)
        all_indexes = []
        for rgb_file in all_rgb_files:
            out = re.search(self.rgb_mask, str(rgb_file))
            if out is not None:
                all_indexes.append(int(out.group(1)))

        train_split_len = round(len(all_indexes) * self.train_split)
        val_split_len = len(all_indexes) - train_split_len

        if self.mode == 'train':
            filelist = all_indexes[val_split_len:train_split_len + 1]

        elif self.mode == 'val':
            filelist = all_indexes[:val_split_len]

        elif 'all' in self.mode:
            filelist = all_indexes

        else:
            filelist = None

        # Default case is to use full split
        if filelist is not None:
            logger.info(f"Using {len(filelist)}/{len(all_indexes)} for task {self.mode}")

        return filelist

    def __getitem__(self, item):
        index = int(self.filelist[item])
        img_path = self.rgb_dir / self.rgb_template.format(index)

        # TODO: Fix properly, screwed up the hypersim dataset with :04d
        if not img_path.is_file():
            img_path = self.rgb_dir / self.rgb_template_alt.format(index)

        image_cv = cv2.imread(str(img_path))
        image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) / 255.0

        depth = self.get_depth(index)
        depth_prior = self.get_depth_prior(index, image_cv.copy(), depth)

        sample = self.transform({'image': image, 'depth': depth, 'depth_prior': depth_prior})
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['depth'] = torch.nan_to_num(sample['depth'], nan=0.0)

        sample['depth_prior'] = torch.from_numpy(sample['depth_prior'])
        sample['depth_prior'] = torch.nan_to_num(sample['depth_prior'], nan=0.0)

        sample['valid_mask'] = ((sample['depth'] > self.depth_min) & (sample['depth'] <= self.depth_max))
        sample['image_path'] = str(img_path)

        return sample

    def __len__(self):
        length = 0
        if self.filelist is not None:
            length = len(self.filelist)

        return length

    def get_depth(self, index):
        depth = None
        depth_path = self.depth_dir / self.depth_template.format(index)
        depth_path_alt = self.depth_dir / self.depth_template_alt.format(index)
        depth_normalized_path = self.depth_dir / self.depth_normalized_template.format(index)
        if depth_path.is_file():
            depth = np.load(depth_path)

        # TODO: Fix properly, screwed up the hypersim dataset with :04d
        elif depth_path_alt.is_file():
            depth = np.load(depth_path_alt)

        elif depth_normalized_path.is_file():
            if self.depth_range is not None and self.depth_min is not None:
                depth_normalized = np.load(depth_normalized_path)
                depth_valid_mask = (depth_normalized > 0.0) & (depth_normalized <= 1.0)
                depth = np.zeros(depth_normalized.shape, dtype='float32')
                depth[depth_valid_mask] = depth_normalized[depth_valid_mask] * self.depth_range + self.depth_min

            else:
                logger.error("Only found normalized depth, but did not find depth normalization file")

        # Relative depth prediction mode
        if self.depth_max == 1.0:
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = np.clip(depth, a_min=1e-6, a_max=None)
        return depth

    def get_approximate_dense_depth(self, item, max_dist=5):
        index = int(self.filelist[item])
        depth_data = self.get_depth(index)
        height, width = depth_data.shape
        v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        v, u = v.flatten(), u.flatten()
        query_points = np.vstack((u, v)).T

        mask = depth_data > 0.0
        v, u = np.where(mask)
        depth_data_flat = depth_data[mask].flatten()
        data_points = np.vstack((u, v)).T

        tree = cKDTree(data_points)
        dist, indices = tree.query(query_points, k=10)

        dense_depth = np.zeros_like(depth_data)

        for i, (point, neighbors_idx, dists) in enumerate(zip(query_points, indices, dist)):
            if np.any(dists == 0):
                # If one of the neighbors has zero distance (i.e., exact match), use its depth directly
                dense_depth[point[1], point[0]] = depth_data_flat[neighbors_idx[np.argmin(dists)]]
            else:
                # Apply inverse distance weighting
                dist_mask = dists < max_dist
                if dist_mask.sum() > 0:
                    weights = 1.0 / dists[dist_mask]
                    weights /= np.sum(weights)
                    interpolated_depth = np.sum(weights * depth_data_flat[neighbors_idx[dist_mask]])
                    dense_depth[point[1], point[0]] = interpolated_depth
                else:
                    dense_depth[point[1], point[0]] = np.nan

        return dense_depth

    def get_depth_range(self):
        norm_min, norm_max, norm_range = None, None, None
        depth_norm_file = self.depth_dir / 'depth_norm_map.txt'
        if depth_norm_file.is_file():
            depth_norm_mask = r'Depth min: ([.0-9]*) to 0, depth max: ([.0-9]*) to 1'

            file = depth_norm_file
            mask = depth_norm_mask

            with file.open("r") as f:
                line = f.readline()

            out = re.search(mask, line)
            if out:
                norm_min = float(out.group(1))
                norm_max = float(out.group(2))
                norm_range = norm_max - norm_min
                logger.info(f'Found norm range {norm_range:.3f} m')

        else:
            logger.error(f"Could not find: {depth_norm_file}")

        return norm_min, norm_max, norm_range

    def get_depth_prior(self, index, img_copy, depth):
        if self.depth_prior_dir.is_dir():
            depth_prior = np.load(str(self.depth_prior_dir / self.depth_prior_template.format(index))).astype(np.float32)
            if depth_prior.max() <= 1.0:
                depth_prior_valid_mask = (depth_prior > 0.0) & (depth_prior <= 1.0)
                depth_prior[depth_prior_valid_mask] *= self.depth_range + self.depth_min

        else:
            depth_prior = self.compute_depth_prior(img_copy, depth)

        return depth_prior


    def compute_depth_prior(self, img_copy, depth):
        quality_level = 0.01
        minimum_distance = 50
        number_of_points = 50
        radius_prior_pixel = 5
        maximum_number_of_corners = 5
        height, width = img_copy.shape[0], img_copy.shape[1]

        circular_mask = self.create_circular_mask(2 * height, 2 * width, radius=radius_prior_pixel)
        depth_prior = np.zeros((height, width), dtype=np.float32)
        img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img, number_of_points, quality_level, minimum_distance)

        if corners is not None:
            corners = corners.squeeze()
            random.shuffle(corners)
            num_corners_to_use = random.randint(1, min(len(corners), maximum_number_of_corners))
            good_corners = 0

            # Process the selected corners
            for c in corners:
                if isinstance(c, np.ndarray):
                    x, y = c
                    translated_mask = circular_mask[int(height - y):int(2 * height - y), int(width - x):int(2 * width - x)]
                    depth_at_prior = np.mean(depth_prior[translated_mask & ~np.isnan(depth_prior)])
                    if depth_at_prior == 0.0:
                        depth_at_corner = np.mean(depth[translated_mask & ~np.isnan(depth)])

                        if depth_at_corner > self.depth_min and depth_at_corner < self.depth_max:
                            depth_prior += depth_at_corner * translated_mask
                            good_corners += 1

                        if good_corners >= num_corners_to_use:
                            break

        if corners is None or good_corners == 0:
            logger.info(f"Using center as backup in {self.dir}")
            depth_prior = self.get_center_prior(depth_prior, depth, width, height, circular_mask)

        return depth_prior

    def get_center_prior(self, depth_prior, depth, width, height, circular_mask):
        x, y = width//2, height//2
        translated_mask = circular_mask[int(height - y):int(2 * height - y), int(width - x):int(2 * width - x)]
        depth_at_corner = np.mean(depth[translated_mask & ~np.isnan(depth)])
        if depth_at_corner > 0.0:
            depth_prior += depth_at_corner * translated_mask
        return depth_prior

    def create_circular_mask(self, h, w, center=None, radius=None):
        # Based on: https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
        if center is None:
            center = (int(w / 2), int(h / 2))

        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

        mask = dist_from_center <= radius
        return mask

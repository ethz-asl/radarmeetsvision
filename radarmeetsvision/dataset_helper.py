######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class MultiDatasetLoader(Dataset):
    def __init__(self, datasets, depth_min_max):
        self.datasets = datasets
        self.cumulative_lengths = [0]
        self.output_depth_min_max = depth_min_max

        total_length = 0
        for dataset in self.datasets:
            total_length += len(dataset)
            self.cumulative_lengths.append(total_length)

        logger.info(f"Multidataset has length {self.cumulative_lengths[-1]}")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        dataset_idx = self.get_dataset_index(index)
        local_idx = index - self.cumulative_lengths[dataset_idx]
        sample = self.datasets[dataset_idx][local_idx]
        return sample

    def get_dataset_index(self, global_index):
        for i, cum_length in enumerate(self.cumulative_lengths):
            if global_index < cum_length:
                return i - 1
        return len(self.datasets) - 1

    def get_approximate_dense_depth(self, index):
        dataset_idx = self._get_dataset_index(index)
        local_idx = index - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_approximate_dense_depth(local_idx)

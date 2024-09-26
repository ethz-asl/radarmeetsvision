import unittest

from .context import *

class BlearnDatasetTestSuite(unittest.TestCase):
    def test_train_len(self):
        # GIVEN: A testing blearndataset object with 5 samples
        dataset = BlearnDataset('tests/resources/tiny_dataset', 'train', (518, 518))

        # THEN: The dataset has length of 0.8*5
        self.assertEqual(len(dataset), 4)

    def test_val_len(self):
        # GIVEN: A testing blearndataset object with 5 samples
        dataset = BlearnDataset('tests/resources/tiny_dataset', 'val', (518, 518))

        # THEN: The validation dataset has length of 0.2*5
        self.assertEqual(len(dataset), 1)

    def test_all_len(self):
        # GIVEN: A testing blearndataset object with 5 samples
        dataset = BlearnDataset('tests/resources/tiny_dataset', 'all', (518, 518))

        # THEN: The validation dataset has length of 5
        self.assertEqual(len(dataset), 5)

    def test_invalid_task_len(self):
        # GIVEN: A testing blearndataset object with 5 samples, but an invalid task
        dataset = BlearnDataset('tests/resources/tiny_dataset', 'invalid', (518, 518))

        # THEN: The validation dataset is None
        self.assertEqual(len(dataset), 0)

    def test_getitem(self):
        # GIVEN: A testing blearndataset object with 5 samples
        dataset = BlearnDataset('tests/resources/tiny_dataset', 'train', (518, 518))

        # WHEN: Getting all samples of the dataset
        for i in range(len(dataset)):
            sample = dataset[i]

            # THEN: Each sample has an RGB image with shape (1x3x512x512)
            self.assertEqual(sample['image'].shape, (3, 518, 518))

            # AND: Each sample has a depth map
            self.assertEqual(sample['depth'].shape, (518, 518))

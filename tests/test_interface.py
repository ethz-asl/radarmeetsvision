import unittest

from .context import *

setup_global_logger()

class RadarMeetsVisionTestSuite(unittest.TestCase):
    def test_interface_load_model_none(self):
        # GIVEN: A rmv interface
        interface = Interface()

        # WHEN: Loading the model, but not setting anything
        model = interface.load_model()

        # THEN: The returned model is None
        self.assertEqual(model, None)

    def test_interface_load_model(self):
        # GIVEN: A rmv interface
        interface = Interface()

        # WHEN: Setting at least the encoder, output_channels
        interface.set_encoder('vits')
        interface.set_depth_range((0.19983673095703125, 120.49285888671875))
        interface.set_output_channels(2)
        interface.set_use_depth_prior(True)

        # AND: Loading the model, but not setting anything
        model = interface.load_model()

        # THEN: The returned model is None
        self.assertNotEqual(model, None)

    def test_get_dataset_loader(self):
        # GIVEN: A rmv interface
        interface = Interface()

        # WHEN: Setting the input/output size
        interface.set_size(518, 518)

        # AND: Setting the batch size
        interface.set_batch_size(4)

        # THEN: Get a valid loader and dataset
        loader, dataset = interface.get_dataset_loader('train', 'tests/resources', ['tiny_dataset', 'tiny_dataset_validation'])
        self.assertNotEqual(loader, None)
        self.assertNotEqual(loader, None)

        # AND: The combined dataset length is 8 entries (both have 4 training entires each)
        self.assertEqual(len(dataset), 8)

    def test_train_epoch(self):
        # GIVEN: A rmv interface
        interface = Interface()

        # AND: Set the epochs and optimizer
        interface.set_epochs(10)

        # AND: Loading the model, but not setting anything
        interface.set_encoder('vits')
        interface.set_depth_range((0.19983673095703125, 120.49285888671875))
        interface.set_output_channels(2)
        interface.set_use_depth_prior(True)
        interface.load_model()

        # AND: Setting the optimizer and criterion
        interface.set_optimizer()
        interface.set_criterion()

        # AND: A training dataset loader
        interface.set_size(518, 518)
        interface.set_batch_size(1)
        loader, _ = interface.get_dataset_loader('train', 'tests/resources', ['tiny_dataset', 'tiny_dataset_validation'])

        # WHEN: A new epoch is trained
        interface.train_epoch(0, loader)

    def test_val_epoch(self):
        # GIVEN: A rmv interface
        interface = Interface()

        # AND: Set the epochs and optimizer
        interface.set_epochs(10)

        # AND: Loading the model, but not setting anything
        interface.set_encoder('vits')
        interface.set_depth_range((0.19983673095703125, 120.49285888671875))
        interface.set_output_channels(2)
        interface.set_use_depth_prior(True)
        interface.load_model()

        # AND: Setting the optimizer
        interface.set_optimizer()

        # AND: A validation dataset loader
        interface.set_size(518, 518)
        interface.set_batch_size(1)
        loader, _ = interface.get_dataset_loader('val', 'tests/resources', ['tiny_dataset', 'tiny_dataset_validation'])

        # WHEN: A epoch is validated
        interface.validate_epoch(0, loader)

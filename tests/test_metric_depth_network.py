import cv2
import numpy as np
import unittest

from .context import *

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class MetricDepthNetworkTestSuite(unittest.TestCase):
    def test_inference(self):
        # GIVEN: A metric depth anything V2 network
        encoder = 'vits'
        model = DepthAnythingV2(**model_configs[encoder])
        device = get_device()
        model = model.to(device).eval()

        # WHEN: A raw file is passed to the model
        raw_img = cv2.imread('tests/resources/tiny_dataset/rgb/00000_rgb.jpg')
        depth = model.infer_image(raw_img, device)

        # THEN: The output depth is valid
        self.assertFalse(np.isnan(depth).any())

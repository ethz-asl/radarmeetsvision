import cv2
import numpy as np
import time
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

    def test_inference_time(self):
        # GIVEN: A metric depth anything V2 network
        model = get_model(None, True, 'vitb', 120.0, 2)
        device = get_device()
        print(f"Using device {device}")
        model = model.to(device).eval()

        # WHEN: Random matrices are inferred
        total_time = 0
        N = 10
        if device != 'cpu':
            N = 500

        for i in range(N):
            img = torch.rand((1, 4, 518, 518), device=device, requires_grad=False)
            start_time = time.monotonic()
            prediction = model.forward(img)
            total_time += (time.monotonic() - start_time)

        print(f"Average time per iteration: {total_time/float(N)}")

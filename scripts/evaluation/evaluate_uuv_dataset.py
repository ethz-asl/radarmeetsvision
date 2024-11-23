######################################################################
#
# Copyright (c) 2024 ETHZ Autonomous Systems Lab. All rights reserved.
#
######################################################################

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import radarmeetsvision as rmv

logger = logging.getLogger(__name__)

class UUVResult:
    def __init__(self):
        self.dir = None

    def set_dir(self, dir):
        self.dir = Path(dir)
        os.makedirs(self.dir, exist_ok=True)

    def save_output(self, index, depth_prediction):
        if depth_prediction is not None:
            depth_prediction_np = depth_prediction.squeeze().cpu().numpy()
            plt.imsave(str(self.dir / f"{index}.jpg"), depth_prediction_np, cmap='viridis')
            np.save(str(self.dir / f"{index}.npy"), depth_prediction_np)

def main():
    parser = argparse.ArgumentParser(description='Purely evalute a network')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--network', type=str, help='Path to the network file')
    args = parser.parse_args()
    rmv.setup_global_logger()

    interface = rmv.Interface()
    interface.set_encoder('vitb')
    depth_min = 0.19983673095703125
    depth_max = 120.49285888671875
    interface.set_depth_range((depth_min, depth_max))
    interface.set_output_channels(2)

    interface.set_size(480, 640)
    interface.set_batch_size(1)
    interface.set_criterion()

    depth_prior_base = Path('/home/asl/Downloads/case_4_data/output/depth_prior')
    depth_prior_dirs = ['1', '2', '3', '4', '5', 'all']
    output_dir = Path(args.dataset) / 'predictions'
    uuv_result = UUVResult()
    for depth_prior_dir in depth_prior_dirs:
        uuv_result.set_dir(output_dir / depth_prior_dir)
        interface.set_use_depth_prior(True, depth_prior_base / depth_prior_dir)
        interface.load_model(pretrained_from=args.network)
        loader = interface.get_single_dataset_loader(args.dataset, min_index=0, max_index=-1)
        interface.validate_epoch(0, loader, iteration_callback=uuv_result.save_output)



if __name__ == "__main__":
    main()
